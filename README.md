# ProfileLikelihood.jl

This module defines the routines required for computing maximum likelihood estimates and profile likelihoods. The optimisation routines are built around the [Optimization.jl](https://github.com/SciML/Optimization.jl) interface. To illustrate how this module works, it is best to give examples (more examples are in the tests). The tests are also useful to see what other methods we provide for working with solutions and problems for these likelihood problems. Note that in the documentation that follows, we are really computing normalised profile log-likelihoods (the profile log-likelihood subtracted by the maximum log-likelihood).

## Example I: Multiple linear regression

Let us start with a multiple linear regression example; this example is also given [here](https://github.com/DanielVandH/TissueMechanics.jl/blob/8384ac0191892e911d1e2e97cc2a1464abeae141/test/ProfileLikelihood/regression.jl). We want to consider some regression problem 

$$
y_i = \beta_0 + \beta_1x_{1i} + \beta_2x_{2i} + \beta_3x_{1i}x_{2i} + \varepsilon_i,\quad i=1,\ldots,n,
$$

where $\varepsilon_i\sim \mathcal N(0, \sigma^2)$. Let us start by simulating some data for this problem.
```Julia
using Random, Distributions
Random.seed!(98871)
n = 300
β = [-1.0, 1.0, 0.5, 3.0]
σ = 0.05
x₁ = rand(Uniform(-1, 1), n)
x₂ = rand(Normal(1.0, 0.5), n)
X = hcat(ones(n), x₁, x₂, x₁ .* x₂)
ε = rand(Normal(0.0, σ), n)
y = X * β + ε
```
We want to re-obtain these coefficients using maximum likelihood estimation. We start by defining our log-likelihood function, remembering that the density function for a Gaussian distribution $\mathcal N(\mu, \sigma^2)$ is $$ \frac{1}{\sqrt{2\mathrm{\pi}\sigma^2}}\exp\left(-\frac{1}{2\sigma^2}\left(x-\mu\right)^2\right). $$
Here is the code we use.
```Julia
using LoopVectorization, LinearAlgebra
function loglik(θ, data)
    σ, β₀, β₁, β₂, β₃ = θ
    y, X, sse, n, β = data
    sse = get_tmp(sse, θ)
    β = get_tmp(β, θ)
    β[1] = β₀
    β[2] = β₁
    β[3] = β₂
    β[4] = β₃
    ℓℓ = -0.5n * log(2π * σ^2)
    mul!(sse, X, β)
    @turbo for i in 1:n
        ℓℓ = ℓℓ - 0.5 / σ^2 * (y[i] - sse[i])^2
    end
    return ℓℓ
end
```
We define the log-likelihood function in terms of the parameters `θ` and some known parameters `data`. We define the `data` by
```Julia
using PreallocationTools 
sse = dualcache(zeros(n))
β_cache = dualcache(similar(β))
dat = (y, X, sse, n, β_cache)
```
Notice that we use `dualcache` from [PreallocationTools.jl](https://github.com/SciML/PreallocationTools.jl). This allows us to pre-allocate an array which can use dual numbers for automatic differentiation. These cache arrays are obtained using the `get_tmp` function in `loglik` above. 

To now use this function, we define a `LikelihoodProblem`.
```Julia
using Optimization, ForwardDiff, LaTeXStrings
using TissueMechanics
θ₀ = ones(5) # initial guess
prob = LikelihoodProblem(loglik, 5;
    θ₀,
    data=dat,
    adtype=Optimization.AutoForwardDiff(),
    lb = [1e-12, -Inf, -Inf, -Inf, -Inf],
    ub = Inf * ones(5),
    names=[L"\sigma", L"\beta_0", L"\beta_1", L"\beta_2", L"\beta_3"],#optional for plotting
    syms = [:σ, :β₀, :β₁, :β₂, :β₃])#optional
```
```Julia
julia> prob
LikelihoodProblem. In-place: true
θ₀: 5-element Vector{Float64}
     σ: 1.0
     β₀: 1.0
     β₁: 1.0
     β₂: 1.0
     β₃: 1.0
```

We can now find the maximum likelihood estimates using `mle`, which will return a `LikelihoodSolution`. By default, this uses the `PolyOpt()` algorithm from [OptimizationPolyAlgorithms.jl](https://github.com/SciML/Optimization.jl/tree/master/lib/OptimizationPolyalgorithms), but we can use any optimiser from [Optimization.jl](https://github.com/SciML/Optimization.jl). Some examples:
```Julia
using OptimizationNLopt, OptimizationOptimJL
sol = mle(prob)                         ## Default 
sol = mle(prob, NLopt.LN_NELDERMEAD())  ## Supplied NLopt optimiser 
sol = mle(prob, Optim.LBFGS())          ## Supplied Optim optimiser
```
```Julia
julia> sol
LikelihoodSolution. retcode: true
Algorithm: LBFGS
Maximum likelihood: 499.05465303642393
Maximum likelihood estimates: 5-element Vector{Float64}
     σ: 0.04584660891973036
     β₀: -0.9957434748868981
     β₁: 0.9980500713585472
     β₂: 0.4970440103699667
     β₃: 3.000932109142093
julia> mle(sol) # = sol.θ
5-element Vector{Float64}:
  0.04584660891973036
 -0.9957434748868981
  0.9980500713585472
  0.4970440103699667
  3.000932109142093
julia> maximum(sol) # = sol.maximum
499.05465303642393     
```

Now let's build some profile likelihoods. This is as simple as calling `profile` on the `LikelihoodProblem`. If we want to profile a single parameter, say `β₂` (the fourth parameter), we could call
```Julia
profile(prob, sol, 4)   # If sol is not provided, it is recomputed
```
To profile all parameters:
```Julia
julia> prof = profile(prob, sol; conf_level = 0.95) # If sol is not provided, it is recomputed
ProfileLikelihoodSolution. MLE retcode: true
Algorithm: LBFGS
Confidence intervals:
     95.0% CI for σ: (0.04240916385931529, 0.04977479434917498)
     95.0% CI for β₀: (-1.0074914601069906, -0.9839954896668065)
     95.0% CI for β₁: (0.9781028298519581, 1.017997312865135)
     95.0% CI for β₂: (0.4866463054614728, 0.5074417152784672)
     95.0% CI for β₃: (2.983268581677337, 3.018595636606847)
```
These confidence intervals can be looked at individual using `confidence_interval`; see the tests. We can call this `ProfileLikelihoodSolution` structure directly to compute the profile likelihood function, which is represented as a spline through the data obtained while computing `prof`.
```Julia
julia> prof(0.05, 1)
-2.1053078622996964
julia> prof([-0.8, -0.9, 0.5, 0.3, 1.0], 2)
5-element Vector{Float64}:
 -5.4545339201225715
 -5.4545339201225715
 -5.4545339201225715
 -5.4545339201225715
 -5.4545339201225715
 ```
 We can plot the profile likelihoods as follows:
```Julia
julia> plot_profiles(prof; fontsize=20, resolution=(1600, 800))
```
![Profile log-likelihood](https://github.com/DanielVandH/TissueMechanics.jl/blob/main/images/profile_likelihood.png)

## Example II: Logistic ODE

Let us now give a logistic ODE example; this example is also given [here](https://github.com/DanielVandH/TissueMechanics.jl/blob/main/test/ProfileLikelihood/logistic_ode.jl). We consider the ODE

$$
\dfrac{\mathrm du}{\mathrm dt} = \lambda u \left(1 - \dfrac{u}{K}\right),\quad t>0,\quad \lambda>0,\,K>0,
$$

with some initial condition  $u(0) =u_0 > 0 $. We will suppose that we have some data  $u_i^o $,  $ i = 1, \ldots, n $, that comes from this ODE, with some noise added so that  $u_i^o = u_i + \varepsilon_i $, where  $\varepsilon_i \sim \mathcal N(0, \sigma^2)  $ and $u_i$ exactly solves the ODE. To start, let us generate some data from this ODE and add some noise:
```Julia
using Random 
Random.seed!(2929911002)
u₀ = 0.5
λ = 1.0
K = 1.0
n = 100
T = 10.0
t = LinRange(0, T, n)
u = @. K * u₀ * exp(λ * t) / (K - u₀ + u₀ * exp(λ * t)) ## exact solution
σ = 0.1
uᵒ = u .+ [0.0, σ * randn(length(u) - 1)...]
```

To now define the likelihood problem, now represented as an `ODELikelihoodProblem`, we need to define (1) the function that computes the right-hand side of the ODE for use with [DifferentialEquations.jl](https://diffeq.sciml.ai/stable/), and (2) the log-likelihood function. The ODE function is
```Julia
function ode_fnc(u, p, t)
    λ, K = p
    du = λ * u * (1 - u / K)
    return du
end
```
(See the [DifferentialEquations.jl documentation](https://diffeq.sciml.ai/stable/) for more details.) For the log-likelihood function, we need an extra parameter `integrator` that contains a pre-defined `integrator` for solving the ODE; this is the same `integrator` as in the [integrator interface from DifferentialEquations.jl](https://diffeq.sciml.ai/stable/basics/integrator/). Here is the function.
```Julia
using TissueMechanics, DifferentialEquations
function loglik(θ, data, integrator)
    ## Extract the parameters
    uᵒ, n = data
    λ, K, σ, u0 = θ
    ## What do you want to do with the integrator?
    integrator.p[1] = λ
    integrator.p[2] = K
    ## Now solve the problem 
    reinit!(integrator, u0)
    solve!(integrator)
    return gaussian_loglikelihood(uᵒ, integrator.sol.u, σ, n)
end
```
The `gaussian_loglikelihood` computes the log-likelihood for Gaussian data. The function `reinit!` updates the initial condition with `u0` and restarts the integrator back to $t = 0$, and `solve!` solves the problem in-place.

Now we define the `ODELikelihoodProblem`.
```Julia
θ₀ = [0.7, 2.0, 0.15, 0.4]
lb = [0.0, 1e-6, 1e-6, 0.0]
ub = [10.0, 10.0, 10.0, 10.0]
param_names = [L"\lambda", L"K", L"\sigma", L"u_0"]
prob = ODELikelihoodProblem(loglik, 4, ode_fnc, u₀, (0.0, T), [1.0, 1.0], t;
    data=(uᵒ, n), θ₀, lb, ub, ode_kwargs=(verbose=false,),
    names=param_names, syms=[:λ, :K, :σ, :u₀])
```
```Julia
julia> prob
ODELikelihoodProblem. In-place: true
θ₀: 4-element Vector{Float64}
     λ: 0.7
     K: 2.0
     σ: 0.15
     u₀: 0.4
```
Note that even though we are estimating the initial condition, we do still need to provide an initial estimate for the initial condition directly in the fourth argument; this ensures that the integrator can be defined. Keyword arguments to the integrator are provided from `ode_kwargs`. The solver that is used to solve the ODE is chosen automatically, but we can provide one using the `ode_alg` keyword argument, e.g.
```Julia
ODELikelihoodProblem(loglik, 4, ode_fnc, u₀, (0.0, T), [1.0, 1.0], t;
    data=(uᵒ, n), θ₀, lb, ub, ode_kwargs=(verbose=false,),
    names=param_names, syms=[:λ, :K, :σ, :u₀], ode_alg = Tsit5())
```
We note that the interface currently does not allow for automatic differentiation, at least until we can figure out how to pre-allocate dual numbers to the integrator interface for a nested function (mentioned in [#33](https://github.com/DanielVandH/TissueMechanics.jl/issues/33)).

Just as with the `LikelihoodProblem` in the first example, we can use `mle` to find the maximum likelihood estimates:
```Julia
julia> using OptimizationNLopt
julia> sol = mle(prob, NLopt.LN_NELDERMEAD())
ODELikelihoodSolution. retcode: XTOL_REACHED
Algorithm: Nelder-Mead simplex algorithm (local, no-derivative)
Maximum likelihood: 86.54963187456266
Maximum likelihood estimates: 4-element Vector{Float64}
     λ: 0.7751368526425868
     K: 1.021426226596144
     σ: 0.10183158749452081
     u₀: 0.5354136334876759
```
We can also profile:
```Julia
julia> prof = profile(prob, sol; alg=NLopt.LN_NELDERMEAD(), conf_level=0.95)
ProfileLikelihoodSolution. MLE retcode: XTOL_REACHED
Algorithm: Nelder-Mead simplex algorithm (local, no-derivative)
Confidence intervals: 
     95.0% CI for λ: (0.5046604984922813, 1.1230565616168797)
     95.0% CI for K: (0.991136221977401, 1.0590950616484003)
     95.0% CI for σ: (0.08919700221648974, 0.11775663655486768)
     95.0% CI for u₀: (0.44852361582663736, 0.6204495782922566)
julia> plot_profiles(prof; fontsize=20, resolution=(1600, 800))
```
![Profile log-likelihood](https://github.com/DanielVandH/TissueMechanics.jl/blob/main/images/profile_likelihood_2.png)

