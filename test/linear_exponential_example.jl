using OrdinaryDiffEq
using ..ProfileLikelihood
using Optimization 
using CairoMakie 
using Random
using Distributions
using MuladdMacro
using LoopVectorization
using LatinHypercubeSampling 
using OptimizationOptimJL
using OptimizationNLopt
using StableRNGs
using ReferenceTests

######################################################
## Example III: Linear Exponential ODE with Grid Search
######################################################
## Step 1: Generate some data for the problem and define the likelihood
rng = StableRNG(2992999)
λ = -0.5
y₀ = 15.0
σ = 0.5
T = 5.0
n = 450
Δt = T / n
t = [j * Δt for j in 0:n]
y = y₀ * exp.(λ * t)
yᵒ = y .+ [0.0, rand(rng, Normal(0, σ), n)...]
@inline function ode_fnc(u, p, t)
    λ = p
    du = λ * u
    return du
end
function _loglik_fnc(θ::AbstractVector{T}, data, integrator) where {T}
    yᵒ, n = data
    λ, σ, u0 = θ
    integrator.p = λ
    ## Now solve the problem 
    reinit!(integrator, u0)
    solve!(integrator)
    if !SciMLBase.successful_retcode(integrator.sol)
        return typemin(T)
    end
    ℓ = -0.5(n + 1) * log(2π * σ^2)
    s = zero(T)
    @turbo @muladd for i in eachindex(yᵒ, integrator.sol.u)
        s = s + (yᵒ[i] - integrator.sol.u[i]) * (yᵒ[i] - integrator.sol.u[i])
    end
    ℓ = ℓ - 0.5s / σ^2
end

## Step 2: Define the problem
θ₀ = [-1.0, 0.5, 19.73] # will be replaced anyway
lb = [-10.0, 1e-6, 0.5]
ub = [10.0, 10.0, 25.0]
syms = [:λ, :σ, :y₀]
prob = LikelihoodProblem(
    _loglik_fnc, θ₀, ode_fnc, y₀, (0.0, T);
    syms=syms,
    data=(yᵒ, n),
    ode_parameters=1.0, # temp value for λ
    ode_kwargs=(verbose=false, saveat=t),
    f_kwargs=(adtype=Optimization.AutoFiniteDiff(),),
    prob_kwargs=(lb=lb, ub=ub),
    ode_alg=Tsit5()
)

## Step 3: Prepare the parameter grid  
regular_grid = RegularGrid(lb, ub, 50) # resolution can also be given as a vector for each parameter
@time gs = grid_search(prob, regular_grid)
@test gs[:λ] ≈ -0.612244897959183
@test gs[:σ] ≈ 0.816327448979592
@test gs[:y₀] ≈ 16.5

@time _gs = grid_search(prob, regular_grid; parallel=Val(true))
@test _gs[:λ] ≈ -0.612244897959183
@test _gs[:σ] ≈ 0.816327448979592
@test _gs[:y₀] ≈ 16.5

gs1, L1 = grid_search(prob, regular_grid; parallel=Val(true), save_vals=Val(true))
gs2, L2 = grid_search(prob, regular_grid; parallel=Val(false), save_vals=Val(true))
@test L1 ≈ L2

# Can also use LatinHypercubeSampling to avoid the dimensionality issue, although 
# you may have to be more particular with choosing the bounds to get good coverage of 
# the parameter space. An example is below.
d = 3
gens = 1000
plan, _ = LHCoptim(500, d, gens; rng)
new_lb = [-2.0, 0.05, 10.0]
new_ub = [2.0, 0.2, 20.0]
bnds = [(new_lb[i], new_ub[i]) for i in 1:d]
parameter_vals = Matrix(scaleLHC(plan, bnds)') # transpose so that a column is a parameter set 
irregular_grid = IrregularGrid(lb, ub, parameter_vals)
gs_ir, loglik_vals_ir = grid_search(prob, irregular_grid; save_vals=Val(true))
max_lik, max_idx = findmax(loglik_vals_ir)
@test max_lik == ProfileLikelihood.get_maximum(gs_ir)
@test parameter_vals[:, max_idx] ≈ ProfileLikelihood.get_mle(gs_ir)

_gs_ir, _loglik_vals_ir = grid_search(prob, irregular_grid; parallel=Val(true), save_vals=Val(true))
_max_lik, _max_idx = findmax(_loglik_vals_ir)
@test _max_lik == ProfileLikelihood.get_maximum(gs_ir)
@test parameter_vals[:, _max_idx] ≈ ProfileLikelihood.get_mle(_gs_ir)
@test loglik_vals_ir ≈ _loglik_vals_ir
@test _max_idx == max_idx

# Also see MultistartOptimization.jl

## Step 4: Compute the MLE, starting at the grid search solution 
prob = ProfileLikelihood.update_initial_estimate(prob, gs)
sol = mle(prob, Optim.LBFGS())
@test ProfileLikelihood.get_mle(sol, 1) ≈ -0.4994204745412974 rtol = 1e-2
@test sol[2] ≈ 0.5287809 rtol = 1e-2
@test sol[:y₀] ≈ 15.145175459094732 rtol = 1e-2

## Step 5: Profile 
prof = profile(prob, sol; alg=NLopt.LN_NELDERMEAD, parallel=true)
@test λ ∈ get_confidence_intervals(prof, :λ)
@test σ ∈ get_confidence_intervals(prof[:σ])
@test y₀ ∈ get_confidence_intervals(prof, 3)

prof1 = profile(prob, sol; alg=NLopt.LN_NELDERMEAD, parallel=true)
prof2 = profile(prob, sol; alg=NLopt.LN_NELDERMEAD, parallel=false)

@test prof1.confidence_intervals[1].lower ≈ prof2.confidence_intervals[1].lower rtol = 1e-3
@test prof1.confidence_intervals[2].lower ≈ prof2.confidence_intervals[2].lower rtol = 1e-3
@test prof1.confidence_intervals[3].lower ≈ prof2.confidence_intervals[3].lower rtol = 1e-3
@test prof1.confidence_intervals[1].upper ≈ prof2.confidence_intervals[1].upper rtol = 1e-2
@test prof1.confidence_intervals[2].upper ≈ prof2.confidence_intervals[2].upper rtol = 1e-3
@test prof1.confidence_intervals[3].upper ≈ prof2.confidence_intervals[3].upper rtol = 1e-3
@test prof1.other_mles[1] ≈ prof2.other_mles[1] rtol = 1e-0 atol = 1e-2
@test prof1.other_mles[2] ≈ prof2.other_mles[2] rtol = 1e-1 atol = 1e-2
@test prof1.other_mles[3] ≈ prof2.other_mles[3] rtol = 1e-3 atol = 1e-2
@test prof1.parameter_values[1] ≈ prof2.parameter_values[1] rtol = 1e-3
@test prof1.parameter_values[2] ≈ prof2.parameter_values[2] rtol = 1e-3
@test prof1.parameter_values[3] ≈ prof2.parameter_values[3] rtol = 1e-3
@test issorted(prof1.parameter_values[1])
@test issorted(prof1.parameter_values[2])
@test issorted(prof1.parameter_values[3])
@test issorted(prof2.parameter_values[1])
@test issorted(prof2.parameter_values[2])
@test issorted(prof2.parameter_values[3])
@test prof1.profile_values[1] ≈ prof2.profile_values[1] rtol = 1e-1
@test prof1.profile_values[2] ≈ prof2.profile_values[2] rtol = 1e-1
@test prof1.profile_values[3] ≈ prof2.profile_values[3] rtol = 1e-1
@test prof1.splines[1].itp.knots ≈ prof2.splines[1].itp.knots
@test prof1.splines[2].itp.knots ≈ prof2.splines[2].itp.knots
@test prof1.splines[3].itp.knots ≈ prof2.splines[3].itp.knots

#bpar = @benchmark profile($prob, $sol; alg=$NLopt.LN_NELDERMEAD, parallel=$true)
#bser = @benchmark profile($prob, $sol; alg=$NLopt.LN_NELDERMEAD, parallel=$false)

prof = prof1

## Step 5: Visualise 
using CairoMakie
fig = plot_profiles(prof; nrow=1, ncol=3,
    latex_names=[L"\lambda", L"\sigma", L"y_0"],
    true_vals=[λ, σ, y₀],
    fig_kwargs=(fontsize=41,),
    axis_kwargs=(width=600, height=300))
resize_to_layout!(fig)
fig_path = normpath(@__DIR__, "..", "docs", "src", "figures")
@test_reference joinpath(fig_path, "linear_exponential_example.png") fig

