######################################################
## Example II: Logistic ODE
######################################################
## Step 1: Generate some data for the problem and define the likelihood
Random.seed!(2929911002)
u₀, λ, K, n, T = 0.5, 1.0, 1.0, 100, 10.0
t = LinRange(0, T, n)
u = @. K * u₀ * exp(λ * t) / (K - u₀ + u₀ * exp(λ * t))
σ = 0.1
uᵒ = u .+ [0.0, σ * randn(length(u) - 1)...] # add some noise 
@inline function ode_fnc(u, p, t)
    local λ, K
    λ, K = p
    du = λ * u * (1 - u / K)
    return du
end
@inline function loglik_fnc2(θ, data, integrator)
    local uᵒ, n, λ, K, σ, u0
    uᵒ, n = data
    λ, K, σ, u0 = θ
    integrator.p[1] = λ
    integrator.p[2] = K
    reinit!(integrator, u0)
    solve!(integrator)
    return gaussian_loglikelihood(uᵒ, integrator.sol.u, σ, n)
end

## Step 2: Define the problem
θ₀ = [0.7, 2.0, 0.15, 0.4]
lb = [0.0, 1e-6, 1e-6, 0.0]
ub = [10.0, 10.0, 10.0, 10.0]
syms = [:λ, :K, :σ, :u₀]
prob = LikelihoodProblem(
    loglik_fnc2, θ₀, ode_fnc, u₀, (0.0, T); # u₀ is just a placeholder IC in this case
    syms=syms,
    data=(uᵒ, n),
    ode_parameters=[1.0, 1.0], # temp values for [λ, K],
    ode_kwargs=(verbose=false, saveat=t),
    f_kwargs=(adtype=Optimization.AutoFiniteDiff(),),
    prob_kwargs=(lb=lb, ub=ub),
    ode_alg=Tsit5()
)

## Step 3: Compute the MLE 
sol = mle(prob, NLopt.LN_BOBYQA; abstol=1e-16, reltol=1e-16)
@test PL.get_maximum(sol) ≈ 86.54963187188535
@test PL.get_mle(sol, 1) ≈ 0.7751485360202867 rtol = 1e-4
@test sol[2] ≈ 1.0214251327023145 rtol = 1e-4
@test sol[3] ≈ 0.10183154994808913 rtol = 1e-4
@test sol[4] ≈ 0.5354121514863078 rtol = 1e-4

## Step 4: Profile 
_prob = deepcopy(prob)
_sol = deepcopy(sol)
prof = profile(prob, sol; conf_level=0.9, parallel=false)
@test sol.mle == _sol.mle
@test sol.maximum == _sol.maximum # checking aliasing 
@test _prob.θ₀ == prob.θ₀
@test λ ∈ get_confidence_intervals(prof, :λ)
@test K ∈ prof.confidence_intervals[2]
@test σ ∈ get_confidence_intervals(prof[:σ])
@test u₀ ∈ get_confidence_intervals(prof, 4)

prof1 = profile(prob, sol; conf_level=0.9, parallel=false)
prof2 = profile(prob, sol; conf_level=0.9, parallel=true)

#b1 = @benchmark profile($prob, $sol; conf_level=$0.9, parallel=$true)
#b2 = @benchmark profile($prob, $sol; conf_level=$0.9, parallel=$false)

@test prof1.confidence_intervals[1].lower ≈ prof2.confidence_intervals[1].lower rtol = 1e-3
@test prof1.confidence_intervals[2].lower ≈ prof2.confidence_intervals[2].lower rtol = 1e-3
@test prof1.confidence_intervals[3].lower ≈ prof2.confidence_intervals[3].lower rtol = 1e-3
@test prof1.confidence_intervals[4].lower ≈ prof2.confidence_intervals[4].lower rtol = 1e-3
@test prof1.confidence_intervals[1].upper ≈ prof2.confidence_intervals[1].upper rtol = 1e-2
@test prof1.confidence_intervals[2].upper ≈ prof2.confidence_intervals[2].upper rtol = 1e-3
@test prof1.confidence_intervals[3].upper ≈ prof2.confidence_intervals[3].upper rtol = 1e-3
@test prof1.confidence_intervals[4].upper ≈ prof2.confidence_intervals[4].upper rtol = 1e-3
@test prof1.other_mles[1] ≈ prof2.other_mles[1] rtol = 1e-0 atol = 1e-2
@test prof1.other_mles[2] ≈ prof2.other_mles[2] rtol = 1e-1 atol = 1e-2
@test prof1.other_mles[3] ≈ prof2.other_mles[3] rtol = 1e-3 atol = 1e-2
@test prof1.other_mles[4] ≈ prof2.other_mles[4] rtol = 1e-0 atol = 1e-2
@test prof1.parameter_values[1] ≈ prof2.parameter_values[1] rtol = 1e-3
@test prof1.parameter_values[2] ≈ prof2.parameter_values[2] rtol = 1e-3
@test prof1.parameter_values[3] ≈ prof2.parameter_values[3] rtol = 1e-3
@test prof1.parameter_values[4] ≈ prof2.parameter_values[4] rtol = 1e-3
@test issorted(prof1.parameter_values[1])
@test issorted(prof1.parameter_values[2])
@test issorted(prof1.parameter_values[3])
@test issorted(prof1.parameter_values[4])
@test issorted(prof2.parameter_values[1])
@test issorted(prof2.parameter_values[2])
@test issorted(prof2.parameter_values[3])
@test issorted(prof2.parameter_values[4])
@test prof1.profile_values[1] ≈ prof2.profile_values[1] rtol = 1e-1
@test prof1.profile_values[2] ≈ prof2.profile_values[2] rtol = 1e-1
@test prof1.profile_values[3] ≈ prof2.profile_values[3] rtol = 1e-1
@test prof1.profile_values[4] ≈ prof2.profile_values[4] rtol = 1e-1
@test prof1.splines[1].itp.knots ≈ prof2.splines[1].itp.knots
@test prof1.splines[2].itp.knots ≈ prof2.splines[2].itp.knots
@test prof1.splines[3].itp.knots ≈ prof2.splines[3].itp.knots
@test prof1.splines[4].itp.knots ≈ prof2.splines[4].itp.knots

prof = prof1

## Step 5: Visualise 
using CairoMakie, LaTeXStrings
fig = plot_profiles(prof;
    latex_names=[L"\lambda", L"K", L"\sigma", L"u_0"],
    show_mles=true,
    shade_ci=true,
    true_vals=[λ, K, σ, u₀],
    fig_kwargs=(fontsize=30, resolution=(1450.0f0, 880.0f0)),
    axis_kwargs=(width=600, height=300))
SAVE_FIGURE && save("figures/logistic_example.png", fig)