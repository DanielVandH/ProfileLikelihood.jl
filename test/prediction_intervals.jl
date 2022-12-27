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
sol = mle(prob, NLopt.LD_LBFGS)
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
prof_idx, param_ranges, splines = ProfileLikelihood.prepare_prediction_grid(prof, resolution)
@test prof_idx == [1, 2, 3]
@test param_ranges[1] ≈ LinRange(0.006400992274213644, 0.01786032876226762, resolution)
@test param_ranges[2] ≈ LinRange(90.81154862835605, 109.54214763511888, resolution)
@test param_ranges[3] ≈ LinRange(1.5919805025139593, 19.070831536649305, resolution)
@test splines == ProfileLikelihood.spline_other_mles(prof)

# Test that we can prepare the prediction cache 
θ, q_vals, q_lower_bounds, q_upper_bounds, q_union_lower_bounds, q_union_upper_bounds = ProfileLikelihood.prepare_prediction_cache(prof, prof_idx, q_prot, resolution)
@test θ == zeros(3)
@test q_vals == Dict(prof_idx .=> [zeros(11, resolution) for _ in 1:3])
@test q_lower_bounds == Dict(prof_idx .=> [Inf * ones(11) for _ in 1:3])
@test q_upper_bounds == Dict(prof_idx .=> [-Inf * ones(11) for _ in 1:3])
@test q_union_lower_bounds == Inf * ones(11)
@test q_union_upper_bounds == -Inf * ones(11)

# Test that we can evaluate the prediction function
data = t
for iip in [false, true]
    for n in 1:3
        q_n = q_vals[n]
        range = param_ranges[n]
        spline = splines[n]
        if !iip
            ProfileLikelihood.evaluate_prediction_function!(q_n, range, spline, θ, n, prediction_function, data, Val(iip))
        else
            ProfileLikelihood.evaluate_prediction_function!(q_n, range, spline, θ, n, prediction_function!, data, Val(iip))
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

θ, q_vals, q_lower_bounds, q_upper_bounds, q_union_bounds = ProfileLikelihood.prepare_prediction_cache(prof, prof_idx, q_prot, resolution)
for iip in [false, true]
    if !iip
        ProfileLikelihood.evaluate_prediction_function!(q_vals, param_ranges, splines, θ, prof_idx, prediction_function, data, Val(iip))
    else
        ProfileLikelihood.evaluate_prediction_function!(q_vals, param_ranges, splines, θ, prof_idx, prediction_function!, data, Val(iip))
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
@test _individual_intervals == individual_intervals
@test _union_intervals == union_intervals
@test _q_vals == q_vals
@test _param_ranges == param_ranges
__individual_intervals, __union_intervals, __q_vals, __param_ranges = get_prediction_intervals(prediction_function!, prof, t; resolution=resolution, q_prototype=q_prot)
@test _individual_intervals == individual_intervals == __individual_intervals
@test _union_intervals == union_intervals == __union_intervals
@test _q_vals == q_vals == __q_vals
@test _param_ranges == param_ranges == __param_ranges

# Look at the inference 
q = prediction_function
iip = isinplace(q, 3)
data = t
@code_warntype isinplace(q, 3)
q_prototype = ProfileLikelihood.build_q_prototype(q, prof, data)
@code_warntype ProfileLikelihood.build_q_prototype(q, prof, data)
@inferred ProfileLikelihood.build_q_prototype(q, prof, data)
prof_idx, param_ranges, splines = ProfileLikelihood.prepare_prediction_grid(prof, resolution)
@code_warntype ProfileLikelihood.prepare_prediction_grid(prof, resolution)
@inferred ProfileLikelihood.prepare_prediction_grid(prof, resolution)
θ, q_vals, q_lower_bounds, q_upper_bounds, q_union_lower_bounds, q_union_upper_bounds = ProfileLikelihood.prepare_prediction_cache(prof, prof_idx, q_prototype, resolution)
@inferred ProfileLikelihood.prepare_prediction_cache(prof, prof_idx, q_prototype, resolution)
@code_warntype ProfileLikelihood.prepare_prediction_cache(prof, prof_idx, q_prototype, resolution)
ProfileLikelihood.evaluate_prediction_function!(q_vals, param_ranges, splines, θ, prof_idx, q, data, Val(iip))
@code_warntype ProfileLikelihood.evaluate_prediction_function!(q_vals, param_ranges, splines, θ, prof_idx, q, data, Val(iip))
ProfileLikelihood.evaluate_prediction_function!(q_vals[1], param_ranges[1], splines[1], θ, 1, q, data, Val(iip))
n = 1
q_n = q_vals[n]
range = param_ranges[n]
spline = splines[n]
iip = Val(false)
g = ProfileLikelihood.evaluate_prediction_function!
@code_warntype g(q_n, range, spline, θ, n, q, data, iip)
ProfileLikelihood.update_interval_bounds!(q_lower_bounds, q_upper_bounds, q_union_lower_bounds, q_union_upper_bounds, q_vals, prof_idx)
@code_warntype ProfileLikelihood.update_interval_bounds!(q_lower_bounds, q_upper_bounds, q_union_lower_bounds, q_union_upper_bounds, q_vals, prof_idx)
individual_intervals, union_intervals = ProfileLikelihood.get_prediction_intervals(prof_idx, q_lower_bounds, q_upper_bounds,
    q_union_lower_bounds, q_union_upper_bounds, 0.95)
@inferred ProfileLikelihood.get_prediction_intervals(prof_idx, q_lower_bounds, q_upper_bounds,
    q_union_lower_bounds, q_union_upper_bounds, 0.95)
@code_warntype ProfileLikelihood.get_prediction_intervals(prof_idx, q_lower_bounds, q_upper_bounds,
    q_union_lower_bounds, q_union_upper_bounds, 0.95)
@inferred get_prediction_intervals(prediction_function, prof, t; resolution=resolution)
@code_warntype get_prediction_intervals(prediction_function, prof, t; resolution=resolution)