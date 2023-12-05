# Example III: Linear exponential ODE and grid searching 

Now we consider $\mathrm dy/\mathrm dt = \lambda y$, $y(0) = y_0$. This has solution $y(t) = y_0\mathrm{e}^{\lambda t}$. First, load the packages we'll be using:

```julia
using OrdinaryDiffEq
using ProfileLikelihood
using Optimization 
using CairoMakie 
using Random
using Distributions
using MuladdMacro
using LoopVectorization
using LatinHypercubeSampling 
using OptimizationOptimJL
using OptimizationNLopt
using Test
using StableRNGs
```

## Setting up the problem 

Let us start by defining the data and the likelihood problem:

```julia
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
using LoopVectorization, MuladdMacro
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
    loglik_fnc, θ₀, ode_fnc, y₀, (0.0, T);
    syms=syms,
    data=(yᵒ, n),
    ode_parameters=1.0, # temp value for λ
    ode_kwargs=(verbose=false, saveat=t),
    f_kwargs=(adtype=Optimization.AutoFiniteDiff(),),
    prob_kwargs=(lb=lb, ub=ub),
    ode_alg=Tsit5()
)
```

## Grid searching 

Let us now give an alternative way of exploring this likelihood function. We have been using `mle`, but we also provide some capability for using a grid search, which can sometimes be useful for e.g. visualising a likelihood function or obtaining initial estmiates for parameters (although it scales terribly for problems with more than even three parameters). Below we define a `RegularGrid`, a regular grid for each parameter:

```julia
regular_grid = RegularGrid(lb, ub, 50) # resolution can also be given as a vector for each parameter
```

We can now use this grid to evaluate the likelihood function at each point, and then return the maximum values (use `save_vals=Val(true)` if you want all the computed values as an array, given as a second argument; also see `?grid_search`). (You can also set `parallel = Val(true)` so that the computation is done with multithreading.)

```julia
gs = grid_search(prob, regular_grid)
LikelihoodSolution. retcode: Success
Maximum likelihood: -548.3068396174556
Maximum likelihood estimates: 3-element Vector{Float64}
     λ: -0.612244897959183
     σ: 0.816327448979592
     y₀: 16.5
```

You could also use an irregular grid, defining some grid as a matrix where each column is a set of parameter values, or a vector of vectors. Here is an example using LatinHypercubeSampling.jl to avoid the dimensionality issue (although in practice we would have to be more careful with choosing the parameter bounds to get good coverage of the parameter space).

```julia
using LatinHypercubeSampling
d = 3
gens = 1000
plan, _ = LHCoptim(500, d, gens; rng)
new_lb = [-2.0, 0.05, 10.0]
new_ub = [2.0, 0.2, 20.0]
bnds = [(new_lb[i], new_ub[i]) for i in 1:d]
parameter_vals = Matrix(scaleLHC(plan, bnds)') # transpose so that a column is a parameter set 
irregular_grid = IrregularGrid(lb, ub, parameter_vals)
gs_ir, loglik_vals_ir = grid_search(prob, irregular_grid; save_vals=Val(true), parallel = Val(true))
```
```julia
julia> gs_ir
LikelihoodSolution. retcode: Success
Maximum likelihood: -2611.078183576969
Maximum likelihood estimates: 3-element Vector{Float64}
     λ: -0.5170340681362726
     σ: 0.18256513026052107
     y₀: 14.348697394789578
```
```julia
max_lik, max_idx = findmax(loglik_vals_ir)
@test max_lik == PL.get_maximum(gs_ir)
@test parameter_vals[:, max_idx] ≈ PL.get_mle(gs_ir)
```

(If you just want to try many points for starting your optimiser, see e.g. the optimiser in MultistartOptimization.jl.)

## Parameter estimation 

Now let's use `mle`. We will restart the initial guess to use the estimates from our grid search.

```julia
prob = update_initial_estimate(prob, gs)
sol = mle(prob, Optim.LBFGS())
```

Now we profile.

```julia
prof = profile(prob, sol; alg=NLopt.LN_NELDERMEAD, parallel = true)
```
```julia
ProfileLikelihoodSolution. MLE retcode: Success
Confidence intervals:
     95.0% CI for λ: (-0.5092192953535792, -0.49323747169071175)
     95.0% CI for σ: (0.4925813447124647, 0.5612815283609663)
     95.0% CI for y₀: (14.856528827532468, 15.173375766524025)
```
```julia
@test λ ∈ get_confidence_intervals(prof, :λ)
@test σ ∈ get_confidence_intervals(prof[:σ])
@test y₀ ∈ get_confidence_intervals(prof, 3)
```

## Visualisation

Finally, we can visualise the profiles:

```julia
fig = plot_profiles(prof; nrow=1, ncol=3,
    latex_names=[L"\lambda", L"\sigma", L"y_0"],
    true_vals=[λ, σ, y₀],
    fig_kwargs=(fontsize=41,),
    axis_kwargs=(width=600, height=300))
resize_to_layout!(fig)
```

```@raw html
<figure>
    <img src='../figures/linear_exponential_example.png', alt'Linear exponential profiles'><br>
</figure>
```