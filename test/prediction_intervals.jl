using OrdinaryDiffEq 
using ..ProfileLikelihood 
using FiniteDiff 
using Optimization
using Random
using OptimizationNLopt 

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

# spline 
splines = ProfileLikelihood.spline_other_mles(prof)
cache = zeros(2)
θ = zeros(3)
for i in keys(splines)
    spl = splines[i]
    for (j, ψ) in pairs(get_parameter_values(prof[i]))
        other_mles = [spl[k](ψ) for k in eachindex(spl)]
        @test other_mles ≈ ProfileLikelihood.get_other_mles(prof[i])[j]
        ProfileLikelihood.eval_other_mles_spline!(cache, spl, ψ)
        @test cache ≈ other_mles
        ProfileLikelihood.build_θ!(θ, i, spl, ψ)
        if i == 1
            @test θ ≈ [ψ, cache...]
        elseif i == 2
            @test θ ≈ [cache[1], ψ, cache[2]]
        elseif i == 3
            @test θ ≈ [cache..., ψ]
        end
    end
end
@inferred ProfileLikelihood.spline_other_mles(prof)

# Prediction 
function prediction_function(θ, data)
    λ, K, u₀ = θ
    t = data
    prob = ODEProblem(ode_fnc, u₀, extrema(t), (λ, K))
    sol = solve(prob, Rosenbrock23(), saveat=t)
    return sol.u::Vector{Float64}
end
λ_rng = LinRange(ProfileLikelihood.get_confidence_intervals(prof[:λ])..., 250)
q_vals = ProfileLikelihood.eval_prediction_function(prediction_function, prof[:λ], λ_rng, t)
@inferred ProfileLikelihood.eval_prediction_function(prediction_function, prof[:λ], λ_rng, t)
for (j, q) in pairs(q_vals)
    local θ
    ψ = λ_rng[j]
    spl = splines[1]
    other_mles = [spl[k](ψ) for k in eachindex(spl)]
    θ = [ψ, other_mles...]
    _q = prediction_function(θ, t)
    @test q ≈ _q
end

q = ProfileLikelihood.eval_prediction_function(prediction_function, prof, t)
@inferred ProfileLikelihood.eval_prediction_function(prediction_function, prof, t)
for i in 1:3
    @test q[i] ≈ ProfileLikelihood.eval_prediction_function(prediction_function, prof[i], LinRange(ProfileLikelihood.get_confidence_intervals(prof[i])..., 250), t)
end

# Prediction intervals
t_many_pts = LinRange(extrema(t)..., 1000)
intervals, union_intervals,  all_curves, param_ranges = get_prediction_intervals(prediction_function, prof, t_many_pts; q_type=Vector{Float64})
@test length(intervals) == 3
@test length(union_intervals) == length(t_many_pts)
for i in 1:3
    for CI in intervals[i]
        @test ProfileLikelihood.get_lower(CI) ≤ ProfileLikelihood.get_upper(CI)
        @test ProfileLikelihood.get_level(CI) == 0.95
    end
end

a = reduce(hcat, [[ProfileLikelihood.get_lower(ci) for ci in intervals[i]] for i in 1:3])
b = reduce(hcat, [[ProfileLikelihood.get_upper(ci) for ci in intervals[i]] for i in 1:3])
min_a = minimum(a; dims=2)
max_b = maximum(b; dims=2)
@test all(a[:, 1] .< b[:, 1])
@test all(a[:, 2] .< b[:, 2])
@test all(a[:, 3] .< b[:, 3])
@test all(min_a .< max_b)
for (j, CI) in pairs(union_intervals)
    @test ProfileLikelihood.get_lower(CI) ≤ ProfileLikelihood.get_upper(CI)
    @test ProfileLikelihood.get_level(CI) == 0.95
    @test ProfileLikelihood.get_lower(CI) == min_a[j] 
    @test ProfileLikelihood.get_upper(CI) == max_b[j]
end