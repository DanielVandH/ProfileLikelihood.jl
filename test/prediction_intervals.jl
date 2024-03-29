using OrdinaryDiffEq
using ..ProfileLikelihood
using FiniteDiff
using InteractiveUtils
using Optimization
using Random
using OptimizationNLopt
using Interpolations

# Setting up 
λ = 0.01
K = 100.0
u₀ = 10.0
t = 0:100:1000
σ = 10.0
@inline function ode_fnc(u, p, t)
    λ, K = p
    du = λ * u * (1 - u / K)
    return du
end
tspan = extrema(t)
p = (λ, K)
prob = ODEProblem(ode_fnc, u₀, tspan, p)
sol = solve(prob, Rosenbrock23(), saveat=t)
Random.seed!(2828)
uᵒ = sol.u + σ * randn(length(t))
@inline function loglik_fnc2(θ, data, integrator)
    λ, K, u₀ = θ
    uᵒ, σ = data
    integrator.p[1] = λ
    integrator.p[2] = K
    reinit!(integrator, u₀)
    solve!(integrator)
    return gaussian_loglikelihood(uᵒ, integrator.sol.u, σ, length(uᵒ))
end
lb = [0.0, 50.0, 0.0]
ub = [0.05, 150.0, 50.0]
θ₀ = [λ, K, u₀]
syms = [:λ, :K, :u₀]
prob = LikelihoodProblem(
    loglik_fnc2, θ₀, ode_fnc, u₀, maximum(t);
    syms=syms,
    data=(uᵒ, σ),
    ode_parameters=[1.0, 1.0],
    ode_kwargs=(verbose=false, saveat=t),
    f_kwargs=(adtype=Optimization.AutoFiniteDiff(),),
    prob_kwargs=(lb=lb, ub=ub),
    ode_alg=Rosenbrock23()
)
sol = mle(prob, NLopt.LN_NELDERMEAD)
prof = profile(prob, sol; alg=NLopt.LN_NELDERMEAD, conf_level=0.95, parallel=true)

function prediction_function(θ, data)
    λ, K, u₀ = θ
    t = data
    prob = ODEProblem(ode_fnc, u₀, extrema(t), (λ, K))
    sol = solve(prob, Rosenbrock23(), saveat=t)
    return sol.u::Vector{Float64}
end
function prediction_function!(u, θ, data)
    λ, K, u₀ = θ
    t = data
    prob = ODEProblem(ode_fnc, u₀, extrema(t), (λ, K))
    sol = solve(prob, Rosenbrock23(), saveat=t)
    u .= sol.u
    return nothing
end

# Test that we can build a prototype 
q_prot = ProfileLikelihood.build_q_prototype(prediction_function, prof, t)
q_prot_2 = ProfileLikelihood.build_q_prototype(prediction_function, prof[1], t)
@test q_prot == q_prot_2 == zeros(11)
@inferred ProfileLikelihood.build_q_prototype(prediction_function, prof, t)
@inferred ProfileLikelihood.build_q_prototype(prediction_function, prof[1], t)

# Test that we can spline the other MLEs 
splines = ProfileLikelihood.spline_other_mles(prof)
@inferred ProfileLikelihood.spline_other_mles(prof)
θ = zeros(3)
for i in 1:3
    spline = splines[i]
    parameter_values = get_parameter_values(prof[i])
    for (j, ψ) in enumerate(parameter_values)
        other = ProfileLikelihood.get_other_mles(prof[i])[j]
        @test spline(ψ) ≈ other
        ProfileLikelihood.build_θ!(θ, i, spline, ψ)
        if i == 1
            @test θ ≈ [ψ, other...]
        elseif i == 2
            @test θ ≈ [other[1], ψ, other[2]]
        elseif i == 3
            @test θ ≈ [other..., ψ]
        end
    end
end

# Test that we can prepare the prediction grid 
resolution = 200
prof_idx, param_ranges, splines, res = ProfileLikelihood.prepare_prediction_grid(prof, resolution)
@test prof_idx == [1, 2, 3]
@test param_ranges[1] ≈ LinRange(prof.confidence_intervals[1]..., resolution)
@test param_ranges[2] ≈ LinRange(prof.confidence_intervals[2]..., resolution)
@test param_ranges[3] ≈ LinRange(prof.confidence_intervals[3]..., resolution)
@test splines == ProfileLikelihood.spline_other_mles(prof)
@test res == 200

# Test that we can prepare the prediction cache 
θ, q_vals, q_lower_bounds, q_upper_bounds, q_union_lower_bounds, q_union_upper_bounds = ProfileLikelihood.prepare_prediction_cache(prof, prof_idx, q_prot, resolution, Val(false))
@test θ == zeros(3)
@test q_vals == Dict(prof_idx .=> [zeros(11, resolution) for _ in 1:3])
@test q_lower_bounds == Dict(prof_idx .=> [Inf * ones(11) for _ in 1:3])
@test q_upper_bounds == Dict(prof_idx .=> [-Inf * ones(11) for _ in 1:3])
@test q_union_lower_bounds == Inf * ones(11)
@test q_union_upper_bounds == -Inf * ones(11)

_θ, _q_vals, _q_lower_bounds, _q_upper_bounds, _q_union_lower_bounds, _q_union_upper_bounds = ProfileLikelihood.prepare_prediction_cache(prof, prof_idx, q_prot, resolution, Val(true))
@test _θ == Dict(1 => [zeros(3) for _ in 1:Base.Threads.nthreads()], 2 => [zeros(3) for _ in 1:Base.Threads.nthreads()], 3 => [zeros(3) for _ in 1:Base.Threads.nthreads()])
@test _q_vals == Dict(prof_idx .=> [zeros(11, resolution) for _ in 1:3])
@test _q_lower_bounds == Dict(prof_idx .=> [Inf * ones(11) for _ in 1:3])
@test _q_upper_bounds == Dict(prof_idx .=> [-Inf * ones(11) for _ in 1:3])
@test _q_union_lower_bounds == Inf * ones(11)
@test _q_union_upper_bounds == -Inf * ones(11)

# Test that we can evaluate the prediction function
θ, q_vals, q_lower_bounds, q_upper_bounds, q_union_lower_bounds, q_union_upper_bounds = ProfileLikelihood.prepare_prediction_cache(prof, prof_idx, q_prot, resolution, Val(false))
data = t
for iip in [false, true]
    for n in 1:3
        q_n = q_vals[n]
        range = param_ranges[n]
        spline = splines[n]
        if !iip
            ProfileLikelihood.evaluate_prediction_function!(q_n, range, spline, θ, n, prediction_function, data, Val(iip), Val(false))
        else
            ProfileLikelihood.evaluate_prediction_function!(q_n, range, spline, θ, n, prediction_function!, data, Val(iip), Val(false))
        end
        for i in 1:resolution
            ψ = range[i]
            if n == 1
                @test q_n[:, i] ≈ prediction_function([ψ, spline(ψ)...], data)
                @test q_vals[n][:, i] ≈ prediction_function([ψ, spline(ψ)...], data)
            elseif n == 2
                @test q_n[:, i] ≈ prediction_function([spline(ψ)[1], ψ, spline(ψ)[2]], data)
                @test q_vals[n][:, i] ≈ prediction_function([spline(ψ)[1], ψ, spline(ψ)[2]], data)
            elseif n == 3
                @test q_n[:, i] ≈ prediction_function([spline(ψ)..., ψ], data)
                @test q_vals[n][:, i] ≈ prediction_function([spline(ψ)..., ψ], data)
            end
        end
    end
end

θ, q_vals, q_lower_bounds, q_upper_bounds, q_union_bounds = ProfileLikelihood.prepare_prediction_cache(prof, prof_idx, q_prot, resolution, Val(false))
for iip in [false, true]
    if !iip
        ProfileLikelihood.evaluate_prediction_function!(q_vals, param_ranges, splines, θ, prof_idx, prediction_function, data, Val(iip), Val(false))
    else
        ProfileLikelihood.evaluate_prediction_function!(q_vals, param_ranges, splines, θ, prof_idx, prediction_function!, data, Val(iip), Val(false))
    end
    for n in 1:3
        for i in 1:resolution
            ψ = param_ranges[n][i]
            if n == 1
                @test q_vals[n][:, i] ≈ prediction_function([ψ, splines[n](ψ)...], data)
            elseif n == 2
                @test q_vals[n][:, i] ≈ prediction_function([splines[n](ψ)[1], ψ, splines[n](ψ)[2]], data)
            elseif n == 3
                @test q_vals[n][:, i] ≈ prediction_function([splines[n](ψ)..., ψ], data)
            end
        end
    end
end

_θ, q_vals, q_lower_bounds, q_upper_bounds, q_union_lower_bounds, q_union_upper_bounds = ProfileLikelihood.prepare_prediction_cache(prof, prof_idx, q_prot, resolution, Val(true))
data = t
for iip in [false, true]
    for n in 1:3
        q_n = q_vals[n]
        range = param_ranges[n]
        spline = splines[n]
        if !iip
            ProfileLikelihood.evaluate_prediction_function!(q_n, range, spline, _θ[n], n, prediction_function, data, Val(iip), Val(true))
        else
            ProfileLikelihood.evaluate_prediction_function!(q_n, range, spline, _θ[n], n, prediction_function!, data, Val(iip), Val(true))
        end
        for i in 1:resolution
            ψ = range[i]
            if n == 1
                @test q_n[:, i] ≈ prediction_function([ψ, spline(ψ)...], data)
                @test q_vals[n][:, i] ≈ prediction_function([ψ, spline(ψ)...], data)
            elseif n == 2
                @test q_n[:, i] ≈ prediction_function([spline(ψ)[1], ψ, spline(ψ)[2]], data)
                @test q_vals[n][:, i] ≈ prediction_function([spline(ψ)[1], ψ, spline(ψ)[2]], data)
            elseif n == 3
                @test q_n[:, i] ≈ prediction_function([spline(ψ)..., ψ], data)
                @test q_vals[n][:, i] ≈ prediction_function([spline(ψ)..., ψ], data)
            end
        end
    end
end

_θ, q_vals, q_lower_bounds, q_upper_bounds, q_union_bounds = ProfileLikelihood.prepare_prediction_cache(prof, prof_idx, q_prot, resolution, Val(true))
for iip in [false, true]
    if !iip
        ProfileLikelihood.evaluate_prediction_function!(q_vals, param_ranges, splines, _θ, prof_idx, prediction_function, data, Val(iip), Val(true))
    else
        ProfileLikelihood.evaluate_prediction_function!(q_vals, param_ranges, splines, _θ, prof_idx, prediction_function!, data, Val(iip), Val(true))
    end
    for n in 1:3
        q_n = q_vals[n]
        range = param_ranges[n]
        spline = splines[n]
        for i in 1:resolution
            ψ = param_ranges[n][i]
            if n == 1
                @test q_vals[n][:, i] ≈ prediction_function([ψ, splines[n](ψ)...], data)
            elseif n == 2
                @test q_vals[n][:, i] ≈ prediction_function([splines[n](ψ)[1], ψ, splines[n](ψ)[2]], data)
            elseif n == 3
                @test q_vals[n][:, i] ≈ prediction_function([splines[n](ψ)..., ψ], data)
            end
        end
    end
end

# Test that we are correctly updating all the interval bounds 
ProfileLikelihood.update_interval_bounds!(q_lower_bounds, q_upper_bounds, q_union_lower_bounds, q_union_upper_bounds, q_vals, prof_idx)
q_1 = q_vals[1]
q_2 = q_vals[2]
q_3 = q_vals[3]
q_lb_1 = minimum(q_1; dims=2)
q_lb_2 = minimum(q_2; dims=2)
q_lb_3 = minimum(q_3; dims=2)
q_ub_1 = maximum(q_1; dims=2)
q_ub_2 = maximum(q_2; dims=2)
q_ub_3 = maximum(q_3; dims=2)
@test q_lb_1 ≈ q_lower_bounds[1]
@test q_lb_2 ≈ q_lower_bounds[2]
@test q_lb_3 ≈ q_lower_bounds[3]
@test q_ub_1 ≈ q_upper_bounds[1]
@test q_ub_2 ≈ q_upper_bounds[2]
@test q_ub_3 ≈ q_upper_bounds[3]
union_lb = min.(q_lb_1, q_lb_2, q_lb_3)
union_ub = max.(q_ub_1, q_ub_2, q_ub_3)
@test q_union_lower_bounds ≈ union_lb
@test q_union_upper_bounds ≈ union_ub

# See that we are getting the intervals correct 
individual_intervals, union_intervals = get_prediction_intervals(prof_idx, q_lower_bounds, q_upper_bounds, q_union_lower_bounds, q_union_upper_bounds, 0.95)
ci_1 = individual_intervals[1]
ci_2 = individual_intervals[2]
ci_3 = individual_intervals[3]
for i in 1:11
    a1, b1 = ci_1[i]
    a2, b2 = ci_2[i]
    a3, b3 = ci_3[i]
    a, b = union_intervals[i]
    @test ci_1[i] isa ProfileLikelihood.ConfidenceInterval
    @test ci_2[i] isa ProfileLikelihood.ConfidenceInterval
    @test ci_3[i] isa ProfileLikelihood.ConfidenceInterval
    @test union_intervals[i] isa ProfileLikelihood.ConfidenceInterval
    @test a1 ≈ q_lb_1[i]
    @test a2 ≈ q_lb_2[i]
    @test a3 ≈ q_lb_3[i]
    @test b1 ≈ q_ub_1[i]
    @test b2 ≈ q_ub_2[i]
    @test b3 ≈ q_ub_3[i]
    @test a ≈ union_lb[i]
    @test b ≈ union_ub[i]
    @test a1 < b1
    @test a2 < b2
    @test a3 < b3
    @test a < b
    @test a == min(a1, a2, a3)
    @test b == max(b1, b2, b3)
    @test ProfileLikelihood.get_level(ci_1[i]) == 0.95
    @test ProfileLikelihood.get_level(ci_2[i]) == 0.95
    @test ProfileLikelihood.get_level(ci_3[i]) == 0.95
    @test ProfileLikelihood.get_level(union_intervals[i]) == 0.95
end

# Test the full procedure  
_individual_intervals, _union_intervals, _q_vals, _param_ranges = get_prediction_intervals(prediction_function, prof, t; resolution=resolution)
__individual_intervals, __union_intervals, __q_vals, __param_ranges = get_prediction_intervals(prediction_function!, prof, t; resolution=resolution, q_prototype=q_prot)
___individual_intervals, ___union_intervals, ___q_vals, ___param_ranges = get_prediction_intervals(prediction_function, prof, t; resolution=resolution, parallel=true)
____individual_intervals, ____union_intervals, ____q_vals, ____param_ranges = get_prediction_intervals(prediction_function!, prof, t; resolution=resolution, q_prototype=q_prot, parallel=true)
@test individual_intervals == _individual_intervals == __individual_intervals == ___individual_intervals == ____individual_intervals
@test union_intervals == _union_intervals == __union_intervals == ___union_intervals == ____union_intervals
@test q_vals == _q_vals == __q_vals == ___q_vals == ____q_vals
@test param_ranges == _param_ranges == __param_ranges == ___param_ranges == ____param_ranges

# Test multithreading
_t = LinRange(extrema(t)..., 1000)
_individual_intervals, _union_intervals, _q_vals, _param_ranges = get_prediction_intervals(prediction_function, prof, _t; resolution=resolution)
__individual_intervals, __union_intervals, __q_vals, __param_ranges = get_prediction_intervals(prediction_function!, prof, _t; resolution=resolution, q_prototype=zero(_t))
___individual_intervals, ___union_intervals, ___q_vals, ___param_ranges = get_prediction_intervals(prediction_function, prof, _t; resolution=resolution, parallel=true)
____individual_intervals, ____union_intervals, ____q_vals, ____param_ranges = get_prediction_intervals(prediction_function!, prof, _t; resolution=resolution, q_prototype=zero(_t), parallel=true)
@test _individual_intervals == __individual_intervals == ___individual_intervals == ____individual_intervals
@test _union_intervals == __union_intervals == ___union_intervals == ____union_intervals
@test _q_vals == __q_vals == ___q_vals == ____q_vals
@test _param_ranges == __param_ranges == ___param_ranges == ____param_ranges

# Look at the inference 
q = prediction_function
iip = isinplace(q, 3)
data = t
q_prototype = ProfileLikelihood.build_q_prototype(q, prof, data)
@inferred ProfileLikelihood.build_q_prototype(q, prof, data)
prof_idx, param_ranges, splines = ProfileLikelihood.prepare_prediction_grid(prof, resolution)
@inferred ProfileLikelihood.prepare_prediction_grid(prof, resolution)
θ, q_vals, q_lower_bounds, q_upper_bounds, q_union_lower_bounds, q_union_upper_bounds = ProfileLikelihood.prepare_prediction_cache(prof, prof_idx, q_prototype, resolution, Val(false))
@inferred ProfileLikelihood.prepare_prediction_cache(prof, prof_idx, q_prototype, resolution, Val(false))
ProfileLikelihood.evaluate_prediction_function!(q_vals, param_ranges, splines, θ, prof_idx, q, data, Val(iip))
ProfileLikelihood.evaluate_prediction_function!(q_vals[1], param_ranges[1], splines[1], θ, 1, q, data, Val(iip))
n = 1
q_n = q_vals[n]
range = param_ranges[n]
spline = splines[n]
iip = Val(false)
g = ProfileLikelihood.evaluate_prediction_function!
ProfileLikelihood.update_interval_bounds!(q_lower_bounds, q_upper_bounds, q_union_lower_bounds, q_union_upper_bounds, q_vals, prof_idx)
individual_intervals, union_intervals = ProfileLikelihood.get_prediction_intervals(prof_idx, q_lower_bounds, q_upper_bounds,
    q_union_lower_bounds, q_union_upper_bounds, 0.95)
@inferred ProfileLikelihood.get_prediction_intervals(prof_idx, q_lower_bounds, q_upper_bounds,
    q_union_lower_bounds, q_union_upper_bounds, 0.95)
@inferred get_prediction_intervals(prediction_function, prof, t; resolution=resolution)

q = prediction_function!
iip = isinplace(q, 3)
data = t
q_prototype = zeros(Float64, length(t))
prof_idx, param_ranges, splines = ProfileLikelihood.prepare_prediction_grid(prof, resolution)
@inferred ProfileLikelihood.prepare_prediction_grid(prof, resolution)
θ, q_vals, q_lower_bounds, q_upper_bounds, q_union_lower_bounds, q_union_upper_bounds = ProfileLikelihood.prepare_prediction_cache(prof, prof_idx, q_prototype, resolution, Val(false))
@inferred ProfileLikelihood.prepare_prediction_cache(prof, prof_idx, q_prototype, resolution, Val(false))
ProfileLikelihood.evaluate_prediction_function!(q_vals, param_ranges, splines, θ, prof_idx, q, data, Val(iip))
ProfileLikelihood.evaluate_prediction_function!(q_vals[1], param_ranges[1], splines[1], θ, 1, q, data, Val(iip))
n = 1
q_n = q_vals[n]
range = param_ranges[n]
spline = splines[n]
iip = Val(false)
g = ProfileLikelihood.evaluate_prediction_function!
ProfileLikelihood.update_interval_bounds!(q_lower_bounds, q_upper_bounds, q_union_lower_bounds, q_union_upper_bounds, q_vals, prof_idx)
individual_intervals, union_intervals = ProfileLikelihood.get_prediction_intervals(prof_idx, q_lower_bounds, q_upper_bounds,
    q_union_lower_bounds, q_union_upper_bounds, 0.95)
@inferred ProfileLikelihood.get_prediction_intervals(prof_idx, q_lower_bounds, q_upper_bounds,
    q_union_lower_bounds, q_union_upper_bounds, 0.95)
@inferred get_prediction_intervals(prediction_function, prof, t; resolution=resolution)

q = prediction_function!
iip = isinplace(q, 3)
data = t
q_prototype = zeros(Float64, length(t))
prof_idx, param_ranges, splines = ProfileLikelihood.prepare_prediction_grid(prof, resolution)
@inferred ProfileLikelihood.prepare_prediction_grid(prof, resolution)
θ, q_vals, q_lower_bounds, q_upper_bounds, q_union_lower_bounds, q_union_upper_bounds = ProfileLikelihood.prepare_prediction_cache(prof, prof_idx, q_prototype, resolution, Val(true))
@inferred ProfileLikelihood.prepare_prediction_cache(prof, prof_idx, q_prototype, resolution, Val(true))
ProfileLikelihood.evaluate_prediction_function!(q_vals, param_ranges, splines, θ, prof_idx, q, data, Val(iip))
ProfileLikelihood.evaluate_prediction_function!(q_vals[1], param_ranges[1], splines[1], θ[1], 1, q, data, Val(iip))
n = 1
q_n = q_vals[n]
range = param_ranges[n]
spline = splines[n]
iip = Val(false)
g = ProfileLikelihood.evaluate_prediction_function!
ProfileLikelihood.update_interval_bounds!(q_lower_bounds, q_upper_bounds, q_union_lower_bounds, q_union_upper_bounds, q_vals, prof_idx)
individual_intervals, union_intervals = ProfileLikelihood.get_prediction_intervals(prof_idx, q_lower_bounds, q_upper_bounds,
    q_union_lower_bounds, q_union_upper_bounds, 0.95)
@inferred ProfileLikelihood.get_prediction_intervals(prof_idx, q_lower_bounds, q_upper_bounds,
    q_union_lower_bounds, q_union_upper_bounds, 0.95)
@inferred get_prediction_intervals(prediction_function, prof, t; resolution=resolution)

## Test the bivariate problem 
prof = bivariate_profile(prob, sol, ((1, 2), (2, 3)); alg=NLopt.LN_NELDERMEAD, conf_level=0.95, parallel=true)

# Test that we can build a prototype 
q_prot = ProfileLikelihood.build_q_prototype(prediction_function, prof, t)
q_prot_2 = ProfileLikelihood.build_q_prototype(prediction_function, prof[1, 2], t)
@test q_prot == q_prot_2 == zeros(11)
@inferred ProfileLikelihood.build_q_prototype(prediction_function, prof, t)
@inferred ProfileLikelihood.build_q_prototype(prediction_function, prof[(1, 2)], t)

# Test that we can spline the other MLEs 
splines = ProfileLikelihood.spline_other_mles(prof)
@inferred ProfileLikelihood.spline_other_mles(prof)
θ = zeros(3)
for i in ((1, 2), (2, 3))
    spline = splines[i]
    grid_1, grid_2 = get_parameter_values(prof[i])
    for (j, ψ) in pairs(grid_1)
        for (k, ϕ) in pairs(grid_2)
            other = ProfileLikelihood.get_other_mles(prof[i], j, k)
            @test spline(ψ, ϕ) ≈ other
            @inferred spline(ψ, ϕ)
            ProfileLikelihood.build_θ!(θ, i, spline, (ψ, ϕ))
            if i == (1, 2)
                @test θ ≈ [ψ, ϕ, other[1]]
            elseif i == (2, 3)
                @test θ ≈ [other[1], ψ, ϕ]
            end
        end
    end
end

# Test that we can prepare the prediction grid 
Random.seed!(929292929) 
resolution = 200
prof_idx, param_ranges, splines, res = ProfileLikelihood.prepare_prediction_grid(prof, resolution)
@test prof_idx == [(1, 2), (2, 3)]
a, b, c, d = ProfileLikelihood.get_bounding_box(prof[(1, 2)])
grid_1 = LinRange(a, b, resolution)
grid_2 = LinRange(c, d, resolution)
full_grid = vec([(x, y) for x in grid_1, y in grid_2])
points_in_cr = full_grid ∈ get_confidence_regions(prof[(1, 2)])
reduced_grid = full_grid[points_in_cr]
old_grid = full_grid[.!points_in_cr]
@test all(reduced_grid ∈ ProfileLikelihood.get_confidence_regions(prof[1, 2]))
for (ψ, ϕ) in reduced_grid
    spl = splines[1, 2]
    η = spl(ψ, ϕ)[1]
    θ = [ψ, ϕ, η]
    ℓ = prof.likelihood_problem.log_likelihood_function(θ, prof.likelihood_problem.data) - sol.maximum
    @test ℓ > ProfileLikelihood.get_chisq_threshold(0.95, 2)
end
for (ψ, ϕ) in old_grid
    spl = splines[1, 2]
    η = spl(ψ, ϕ)[1]
    θ = [ψ, ϕ, η]
    ℓ = prof.likelihood_problem.log_likelihood_function(θ, prof.likelihood_problem.data) - sol.maximum
    @test ℓ < ProfileLikelihood.get_chisq_threshold(0.95, 2) + 1e-3
end
len_1 = length(reduced_grid)

a, b, c, d = ProfileLikelihood.get_bounding_box(prof[(2, 3)])
grid_1 = LinRange(a, b, resolution)
grid_2 = LinRange(c, d, resolution)
full_grid = vec([(x, y) for x in grid_1, y in grid_2])
points_in_cr = full_grid ∈ get_confidence_regions(prof[(2, 3)])
reduced_grid = full_grid[points_in_cr]
old_grid = full_grid[.!points_in_cr]
@test all(reduced_grid ∈ ProfileLikelihood.get_confidence_regions(prof[2, 3]))
for (ψ, ϕ) in reduced_grid
    spl = splines[2, 3]
    η = spl(ψ, ϕ)[1]
    θ = [η, ψ, ϕ]
    ℓ = prof.likelihood_problem.log_likelihood_function(θ, prof.likelihood_problem.data) - sol.maximum
    @test ℓ > ProfileLikelihood.get_chisq_threshold(0.95, 2)
end
for (ψ, ϕ) in old_grid
    spl = splines[2, 3]
    η = spl(ψ, ϕ)[1]
    θ = [η, ψ, ϕ]
    ℓ = prof.likelihood_problem.log_likelihood_function(θ, prof.likelihood_problem.data) - sol.maximum
    @test ℓ < ProfileLikelihood.get_chisq_threshold(0.95, 2) + 1e-3
end
len_2 = length(reduced_grid)
@test res == min(len_1, len_2)
@test splines == ProfileLikelihood.spline_other_mles(prof)

# Test that we can prepare the prediction cache 
θ, q_vals, q_lower_bounds, q_upper_bounds, q_union_lower_bounds, q_union_upper_bounds = ProfileLikelihood.prepare_prediction_cache(prof, prof_idx, q_prot, res, Val(false))
@test θ == zeros(3)
@test q_vals == Dict(prof_idx .=> [zeros(11, res) for _ in 1:2])
@test q_lower_bounds == Dict(prof_idx .=> [Inf * ones(11) for _ in 1:2])
@test q_upper_bounds == Dict(prof_idx .=> [-Inf * ones(11) for _ in 1:2])
@test q_union_lower_bounds == Inf * ones(11)
@test q_union_upper_bounds == -Inf * ones(11)

_θ, _q_vals, _q_lower_bounds, _q_upper_bounds, _q_union_lower_bounds, _q_union_upper_bounds = ProfileLikelihood.prepare_prediction_cache(prof, prof_idx, q_prot, res, Val(true))
@test _θ == Dict((1, 2) => [zeros(3) for _ in 1:Base.Threads.nthreads()], (2, 3) => [zeros(3) for _ in 1:Base.Threads.nthreads()])
@test _q_vals == Dict(prof_idx .=> [zeros(11, res) for _ in 1:2])
@test _q_lower_bounds == Dict(prof_idx .=> [Inf * ones(11) for _ in 1:2])
@test _q_upper_bounds == Dict(prof_idx .=> [-Inf * ones(11) for _ in 1:2])
@test _q_union_lower_bounds == Inf * ones(11)
@test _q_union_upper_bounds == -Inf * ones(11)

# Test that we can evaluate the prediction function
θ, q_vals, q_lower_bounds, q_upper_bounds, q_union_lower_bounds, q_union_upper_bounds = ProfileLikelihood.prepare_prediction_cache(prof, prof_idx, q_prot, res, Val(false))
data = t
for iip in [false, true]
    for n in ((1, 2), (2, 3))
        q_n = q_vals[n]
        range = param_ranges[n]
        spline = splines[n]
        if !iip
            ProfileLikelihood.evaluate_prediction_function!(q_n, range, spline, θ, n, prediction_function, data, Val(iip), Val(false))
        else
            ProfileLikelihood.evaluate_prediction_function!(q_n, range, spline, θ, n, prediction_function!, data, Val(iip), Val(false))
        end
        for i in 1:res
            ψ, ϕ = range[i]
            if n == (1, 2)
                @test q_n[:, i] ≈ prediction_function([ψ, ϕ, spline(ψ, ϕ)[1]], data)
                @test q_vals[n][:, i] ≈ prediction_function([ψ, ϕ, spline(ψ, ϕ)[1]], data)
            elseif n == (2, 3)
                @test q_n[:, i] ≈ prediction_function([spline(ψ, ϕ)[1], ψ, ϕ], data)
                @test q_vals[n][:, i] ≈ prediction_function([spline(ψ, ϕ)[1], ψ, ϕ], data)
            end
        end
    end
end

θ, q_vals, q_lower_bounds, q_upper_bounds, q_union_bounds = ProfileLikelihood.prepare_prediction_cache(prof, prof_idx, q_prot, res, Val(false))
for iip in [false, true]
    if !iip
        ProfileLikelihood.evaluate_prediction_function!(q_vals, param_ranges, splines, θ, prof_idx, prediction_function, data, Val(iip), Val(false))
    else
        ProfileLikelihood.evaluate_prediction_function!(q_vals, param_ranges, splines, θ, prof_idx, prediction_function!, data, Val(iip), Val(false))
    end
    for n in ((1, 2), (2, 3))
        q_n = q_vals[n]
        range = param_ranges[n]
        spline = splines[n]
        for i in 1:res
            ψ, ϕ = range[i]
            if n == (1, 2)
                @test q_n[:, i] ≈ prediction_function([ψ, ϕ, spline(ψ, ϕ)[1]], data)
                @test q_vals[n][:, i] ≈ prediction_function([ψ, ϕ, spline(ψ, ϕ)[1]], data)
            elseif n == (2, 3)
                @test q_n[:, i] ≈ prediction_function([spline(ψ, ϕ)[1], ψ, ϕ], data)
                @test q_vals[n][:, i] ≈ prediction_function([spline(ψ, ϕ)[1], ψ, ϕ], data)
            end
        end
    end
end

_θ, q_vals, q_lower_bounds, q_upper_bounds, q_union_lower_bounds, q_union_upper_bounds = ProfileLikelihood.prepare_prediction_cache(prof, prof_idx, q_prot, res, Val(true))
data = t
for iip in [false, true]
    for n in ((1, 2), (2, 3))
        q_n = q_vals[n]
        range = param_ranges[n]
        spline = splines[n]
        if !iip
            ProfileLikelihood.evaluate_prediction_function!(q_n, range, spline, _θ[n], n, prediction_function, data, Val(iip), Val(true))
        else
            ProfileLikelihood.evaluate_prediction_function!(q_n, range, spline, _θ[n], n, prediction_function!, data, Val(iip), Val(true))
        end
        for i in 1:res
            ψ, ϕ = range[i]
            if n == (1, 2)
                @test q_n[:, i] ≈ prediction_function([ψ, ϕ, spline(ψ, ϕ)[1]], data)
                @test q_vals[n][:, i] ≈ prediction_function([ψ, ϕ, spline(ψ, ϕ)[1]], data)
            elseif n == (2, 3)
                @test q_n[:, i] ≈ prediction_function([spline(ψ, ϕ)[1], ψ, ϕ], data)
                @test q_vals[n][:, i] ≈ prediction_function([spline(ψ, ϕ)[1], ψ, ϕ], data)
            end
        end
    end
end

_θ, q_vals, q_lower_bounds, q_upper_bounds, q_union_bounds = ProfileLikelihood.prepare_prediction_cache(prof, prof_idx, q_prot, res, Val(true))
for iip in [false, true]
    if !iip
        ProfileLikelihood.evaluate_prediction_function!(q_vals, param_ranges, splines, _θ, prof_idx, prediction_function, data, Val(iip), Val(true))
    else
        ProfileLikelihood.evaluate_prediction_function!(q_vals, param_ranges, splines, _θ, prof_idx, prediction_function!, data, Val(iip), Val(true))
    end
    for n in ((1, 2), (2, 3))
        q_n = q_vals[n]
        range = param_ranges[n]
        spline = splines[n]
        for i in 1:res
            ψ, ϕ = range[i]
            if n == (1, 2)
                @test q_n[:, i] ≈ prediction_function([ψ, ϕ, spline(ψ, ϕ)[1]], data)
                @test q_vals[n][:, i] ≈ prediction_function([ψ, ϕ, spline(ψ, ϕ)[1]], data)
            elseif n == (2, 3)
                @test q_n[:, i] ≈ prediction_function([spline(ψ, ϕ)[1], ψ, ϕ], data)
                @test q_vals[n][:, i] ≈ prediction_function([spline(ψ, ϕ)[1], ψ, ϕ], data)
            end
        end
    end
end

# Test that we are correctly updating all the interval bounds 
ProfileLikelihood.update_interval_bounds!(q_lower_bounds, q_upper_bounds, q_union_lower_bounds, q_union_upper_bounds, q_vals, prof_idx)
q_1 = q_vals[(1, 2)]
q_2 = q_vals[(2, 3)]
q_lb_1 = minimum(q_1; dims=2)
q_lb_2 = minimum(q_2; dims=2)
q_ub_1 = maximum(q_1; dims=2)
q_ub_2 = maximum(q_2; dims=2)
@test q_lb_1 ≈ q_lower_bounds[(1, 2)]
@test q_lb_2 ≈ q_lower_bounds[(2, 3)]
@test q_ub_1 ≈ q_upper_bounds[(1, 2)]
@test q_ub_2 ≈ q_upper_bounds[(2, 3)]
union_lb = min.(q_lb_1, q_lb_2)
union_ub = max.(q_ub_1, q_ub_2)
@test q_union_lower_bounds ≈ union_lb
@test q_union_upper_bounds ≈ union_ub

# See that we are getting the intervals correct 
individual_intervals, union_intervals = get_prediction_intervals(prof_idx, q_lower_bounds, q_upper_bounds, q_union_lower_bounds, q_union_upper_bounds, 0.95)
ci_1 = individual_intervals[(1, 2)]
ci_2 = individual_intervals[(2, 3)]
for i in 1:11
    a1, b1 = ci_1[i]
    a2, b2 = ci_2[i]
    a, b = union_intervals[i]
    @test ci_1[i] isa ProfileLikelihood.ConfidenceInterval
    @test ci_2[i] isa ProfileLikelihood.ConfidenceInterval
    @test union_intervals[i] isa ProfileLikelihood.ConfidenceInterval
    @test a1 ≈ q_lb_1[i]
    @test a2 ≈ q_lb_2[i]
    @test b1 ≈ q_ub_1[i]
    @test b2 ≈ q_ub_2[i]
    @test a ≈ union_lb[i]
    @test b ≈ union_ub[i]
    @test a1 < b1
    @test a2 < b2
    @test a < b
    @test a == min(a1, a2)
    @test b == max(b1, b2)
    @test ProfileLikelihood.get_level(ci_1[i]) == 0.95
    @test ProfileLikelihood.get_level(ci_2[i]) == 0.95
    @test ProfileLikelihood.get_level(union_intervals[i]) == 0.95
end

# Test the full procedure 
Random.seed!(929292929)
@time _individual_intervals, _union_intervals, _q_vals, _param_ranges = get_prediction_intervals(prediction_function, prof, t; resolution=resolution)
Random.seed!(929292929)
@time __individual_intervals, __union_intervals, __q_vals, __param_ranges = get_prediction_intervals(prediction_function!, prof, t; resolution=resolution, q_prototype=q_prot)
Random.seed!(929292929)
@time ___individual_intervals, ___union_intervals, ___q_vals, ___param_ranges = get_prediction_intervals(prediction_function, prof, t; resolution=resolution, parallel=true)
Random.seed!(929292929)
@time ____individual_intervals, ____union_intervals, ____q_vals, ____param_ranges = get_prediction_intervals(prediction_function!, prof, t; resolution=resolution, q_prototype=q_prot, parallel=true)
@test individual_intervals == _individual_intervals == __individual_intervals == ___individual_intervals == ____individual_intervals
@test union_intervals == _union_intervals == __union_intervals == ___union_intervals == ____union_intervals
@test q_vals == _q_vals == __q_vals == ___q_vals == ____q_vals
@test param_ranges == _param_ranges == __param_ranges == ___param_ranges == ____param_ranges

# Test multithreading
_t = LinRange(extrema(t)..., 1000)
Random.seed!(929292929) 
@time _individual_intervals, _union_intervals, _q_vals, _param_ranges = get_prediction_intervals(prediction_function, prof, _t; resolution=resolution)
Random.seed!(929292929) 
@time __individual_intervals, __union_intervals, __q_vals, __param_ranges = get_prediction_intervals(prediction_function!, prof, _t; resolution=resolution, q_prototype=zero(_t))
Random.seed!(929292929) 
@time ___individual_intervals, ___union_intervals, ___q_vals, ___param_ranges = get_prediction_intervals(prediction_function, prof, _t; resolution=resolution, parallel=true)
Random.seed!(929292929) 
@time ____individual_intervals, ____union_intervals, ____q_vals, ____param_ranges = get_prediction_intervals(prediction_function!, prof, _t; resolution=resolution, q_prototype=zero(_t), parallel=true)
@test _individual_intervals == __individual_intervals == ___individual_intervals == ____individual_intervals
@test _union_intervals == __union_intervals == ___union_intervals == ____union_intervals
@test _q_vals == __q_vals == ___q_vals == ____q_vals
@test _param_ranges == __param_ranges == ___param_ranges == ____param_ranges