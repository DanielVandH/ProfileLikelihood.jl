# ProfileLikelihood 

This module defines the routines required for computing maximum likelihood estimates and profile likelihoods. The optimisation routines are built around the [Optimization.jl](https://github.com/SciML/Optimization.jl) interface, allowing us to e.g. easily switch between algorithms, between finite differences and automatic differentiation, and it allows for constraints to be defined with ease. Below we list the definitions we are using for likelihoods and profile likelihoods. This code only works for scalar parameters of interest (i.e. out of a vector $\boldsymbol \theta$, you can profile a single scalar parameter $\theta_i \in \boldsymbol\theta$) for now.

**Definition: Likelihood function** (see Casella & Berger, 2002): Let $f(\boldsymbol x \mid \boldsymbol \theta)$ denote the joint probability density function (PDF) of the sample $\boldsymbol X = (X_1,\ldots,X_n)^{\mathsf T}$, where $\boldsymbol \theta \in \Theta$ is some set of parameters and $\Theta$ is the parameter space. We define the _likelihood function_ $\mathcal L \colon \Theta \to [0, \infty)$ by $\mathcal L(\boldsymbol \theta \mid \boldsymbol x) = f(\boldsymbol x \mid \boldsymbol \theta)$ for some realisation $\boldsymbol x = (x_1,\ldots,x_n)^{\mathsf T}$ of $\boldsymbol X$. The _log-likelihood function_ $\ell\colon\Theta\to\mathbb R$ is defined by $\ell(\boldsymbol \theta \mid \boldsymbol x) =  \log\mathcal L(\boldsymbol\theta \mid \boldsymbol x)$.The _maximum likelihood estimate_ (MLE) $\hat{\boldsymbol\theta}$ is the parameter $\boldsymbol\theta$ that maximises the likelihood function, $\hat{\boldsymbol{\theta}} = argmax_{\boldsymbol{\theta} \in \Theta} \mathcal{L}(\boldsymbol{\theta} \mid \boldsymbol x) = argmax_{\boldsymbol\theta \in \Theta} \ell(\boldsymbol\theta \mid \boldsymbol x)$.

**Definition: Profile likelihood function** (see Pawitan, 2001): Suppose we have some parameters of interest, $\boldsymbol \theta \in \Theta$, and some nuisance parameters, $\boldsymbol \phi \in \Phi$, and some data $\boldsymbol x = (x_1,\ldots,x_n)^{\mathsf T}$, giving smoe joint likelihood $\mathcal L \colon \Theta \cup \Phi \to [0, \infty)$ defined by $\mathcal L(\boldsymbol\theta, \boldsymbol\phi \mid \boldsymbol x)$. We define the _profile likelihood_ $\mathcal L_p \colon \Theta \to [0, \infty)$ of $\boldsymbol\theta$ by $\mathcal L_p(\boldsymbol\theta \mid \boldsymbol x) = \sup_{\boldsymbol \phi \in \Phi \mid \boldsymbol \theta} \mathcal L(\boldsymbol \theta, \boldsymbol \phi \mid \boldsymbol x)$. The _profile log-likelihood_ $\ell_p \colon \Theta \to \mathbb R$ of $\boldsymbol\theta$ is defined by $\ell_p(\boldsymbol \theta \mid \boldsymbol x) = \log \mathcal L_p(\boldsymbol\theta \mid \boldsymbol x)$. The _normalised profile likelihood_ is defined by $\hat{\mathcal L}_p(\boldsymbol\theta \mid \boldsymbol x) = \mathcal L_p(\boldsymbol \theta \mid \boldsymbol x) - \mathcal L_p(\hat{\boldsymbol\theta} \mid \boldsymbol x)$, where $\hat{\boldsymbol\theta}$ is the MLE of $\boldsymbol\theta$, and similarly for the normalised profile log-likelihood.

From Wilk's theorem, we know that $2\hat{\ell}\_p(\boldsymbol\theta \mid \boldsymbol x) \geq -\chi_{p, 1-\alpha}^2$ is an approximate $100(1-\alpha)\%$ confidence region for $\boldsymbol \theta$, and this allows us to obtain confidence intervals for parameters by considering only their profile likelihood, where $\chi_{p,1-\alpha}^2$ is the $1-\alpha$ quantile of the $\chi_p^2$ distribution and $p$ is the length of $\boldsymbol\theta$. For the case of a scalar parameter of interest, $-\chi_{1, 0.95}^2 \approx -1.92$.

We compute the profile log-likelihood in this package by starting at the MLE, and stepping left/right until we reach a given threshold. The code is iterative to not waste time in so much of the parameter space.

# Interface

The interface for defining a likelihood problem builds on top of [Optimization.jl](https://github.com/SciML/Optimization.jl). Below we list the three main structs that we use, with `LikelihoodProblem` the most important one and the only one that needs to be directly defined. Examples of how we use these structs are given later, and much extra functionality is given in the tests.

## Defining the problem: LikelihoodProblem

The `LikelihoodProblem` is the definition of a likelihood function, and provides the following constructor:

```julia
LikelihoodProblem(loglik::Function, θ₀;
    syms=eachindex(θ₀), data=SciMLBase.NullParameters(),
    f_kwargs=nothing, prob_kwargs=nothing)
```

Here, `loglik` is a function for the log-likelihood, taking the form `ℓ(θ, p)`. The second argument, `θ₀`, is the initial estimate for the parameter values. You can provide symbolic names for the parameters via `syms`, so that e.g. `prob[:α]` (where `prob` is a `LikelihoodProblem` with `:α ∈ syms`) returns the initial estimate for `:α`. The argument `p` in the likelihood function can be used to pass data or other parameters into the argument, and the keyword argument `data` can be used for this. Lastly, `f_kwargs` and `prob_kwargs` are additional keyword arguments for the `OptimizationFunction` and `OptimizationProblem`, respectively; see the [Optimization.jl](https://github.com/SciML/Optimization.jl) documentation for more detail here.

We also provide a simple interface for defining a log-likelihood that requires the solution of a differential equation:

```julia 
LikelihoodProblem(loglik::Function, θ₀,
    ode_function, u₀, tspan;
    syms=eachindex(θ₀), data=SciMLBase.NullParameters(),
    ode_parameters=SciMLBase.NullParameters(), ode_alg,
    ode_kwargs=nothing, f_kwargs=nothing, prob_kwargs=nothing)
```

Importantly, `loglik` in this case is now a function of the form `ℓ(θ, p, integrator)`, where `integrator` is the same integrator as in the integrator interface from DifferentialEquations.jl; see the documentation at DifferentialEquations.jl for more detail on using the integrator. Furthermore, `ode_function` is the function for the ODE, `u₀` its initial condition, and `tspan` its time span. Additionally, the parameters for the `ode_function` (e.g. the `p` in `ode_function(du, u, p, t)` or `ode_function(u, p, t)`) can be passed using the keyword argument `ode_parameters`. The algorithm used to solve the differential equation is passed with `ode_alg`, and lastly any additional keyword arguments for solving the problem are to be passed through `ode_kwargs`. 

## Solving the problem: mle and LikelihoodSolution 

The MLEs for a given `LikelihoodProblem` are found using the function `mle`, e.g. `mle(prob, Optim.LBFGS())` will optimise the likelihood function using the LBFGS algorithm from Optim.jl (see also `?mle`). This function returns a `LikelihoodSolution`, defined by:

```julia
struct LikelihoodSolution{Θ,P,M,R,A} <: AbstractLikelihoodSolution
    mle::Θ
    problem::P
    optimiser::A
    maximum::M
    retcode::R
end
```

If `sol isa LikelihoodSolution`, then you can use the `syms` from your original problem to access a specific MLE, e.g. `sol[:α]` would return the MLE for the paramter `:α`.

## Profiling the parameters: profile and ProfileLikelihoodSolution 

The results for a profile likelihood, obtained from `profile(prob, sol)` (see also `?profile`), are stored in a `ProfileLikelihoodSolution` struct:

```julia
struct ProfileLikelihoodSolution{I,V,LP,LS,Spl,CT,CF,OM}
    parameter_values::Dict{I,V}
    profile_values::Dict{I,V}
    likelihood_problem::LP
    likelihood_solution::LS
    splines::Dict{I,Spl}
    confidence_intervals::Dict{I,ConfidenceInterval{CT,CF}}
    other_mles::OM
end
```

Here, the parameter values used for each parameter are given in `parameter_values`, with parameter indices (or symbols) are mapped to these values. Similarly, the values of the profile log-likelihood are stored in `profile_values`. We use a spline (see Interpolations.jl) to make the profile log-likelihood a continuous function, and these splines are given by `splines`. Next, the computed confidence intervals are given in `confidence_intervals`, with a confidence interval represented by a `ConfidenceInterval` struct. Lastly, since computing the profile log-likelihood function requires an optimisation problem with one variable fixed and the others free, we obtain for each profile log-likelihood value a set of optimised parameters -- these parameters are given in `other_mles`.

If `prof` is a `ProfileLikelihoodSolution`, then you can also call it as e.g. `prof(0.5, 1)` to evaluate the profile log-likelihood function of the first parameter at the point `0.5`. Alternatively, `prof(0.7, :α)` does the same but for the parameter `:α` at the point `0.7`. You can also index `prof` at a specific index (or symbol) to see the results only for that parameter, e.g. `prof[1]` or `prof[:α]`; this returns a `ProfileLikelihoodSolutionView`.

# Examples 

Let us now give some examples. More detail is given in the tests. 

## Multiple linear regression

Let us start with a linear regression example. We perform a simulation study where we try and estimate the parameters in a regression of the form 

$$
y_i = \beta_0 + \beta_1x_{1i} + \beta_2x_{2i} + \beta_3x_{1i}x_{3i} + \varepsilon_i, \quad \varepsilon_i \sim \mathcal N(0, \sigma^2), \quad i=1,2,\ldots, n. 
$$

We also try and estimate $\sigma$. Let us start by simulating the data:

```julia 
using Random, Distributions 
Random.seed!(98871)
n = 600
β = [-1.0, 1.0, 0.5, 3.0]
σ = 0.05
x₁ = rand(Uniform(-1, 1), n)
x₂ = rand(Normal(1.0, 0.5), n)
X = hcat(ones(n), x₁, x₂, x₁ .* x₂)
ε = rand(Normal(0.0, σ), n)
y = X * β + ε
```

The data `y` is now our noisy data. The likelihood function in this is $\ell(\sigma, \boldsymbol \beta \mid \boldsymbol y) = -(n/2)\log(2\mathrm{\pi}\sigma^2) - (1/2\sigma^2)\sum_i (y_i - \beta_0 - \beta_1x_{1i} - \beta_2x_{2i} - \beta_3x_{1i}x_{2i})^2$. We now define our likelihood function. To allow for automatic differentiation, we use `PreallocationTools.DiffCache` to define our cache vectors.

```julia 
sse = DiffCache(zeros(n))
β_cache = DiffCache(similar(β), 10)
dat = (y, X, sse, n, β_cache)
@inline function loglik_fnc(θ, data)
    σ, β₀, β₁, β₂, β₃ = θ
    y, X, sse, n, β = data
    _sse = get_tmp(sse, θ)
    _β = get_tmp(β, θ)
    _β[1] = β₀
    _β[2] = β₁
    _β[3] = β₂
    _β[4] = β₃
    ℓℓ = -0.5n * log(2π * σ^2)
    mul!(_sse, X, _β)
    for i in eachindex(y)
        ℓℓ = ℓℓ - 0.5 / σ^2 * (y[i] - _sse[i])^2
    end
    return ℓℓ
end
```

Now having defined our likelihood, we can define the likelihood problem. We let the problem be unconstrained, except for $\sigma > 0$. We start at the value $1$ for each parameter. To use automatic differentiation, we use `Optimization.AutoForwardDiff` for the `adtype`.

```julia 
using Optimization, ForwardDiff
θ₀ = ones(5)
prob = LikelihoodProblem(loglik_fnc, θ₀;
    data=dat,
    f_kwargs=(adtype=Optimization.AutoForwardDiff(),),
    prob_kwargs=(
        lb=[0.0, -Inf, -Inf, -Inf, -Inf],
        ub=Inf * ones(5)
    ),
    syms=[:σ, :β₀, :β₁, :β₂, :β₃]
)
LikelihoodProblem. In-place: true
θ₀: 5-element Vector{Float64}
     σ: 1.0
     β₀: 1.0
     β₁: 1.0
     β₂: 1.0
     β₃: 1.0
```

Now we can compute the MLEs.

```julia 
using OptimizationOptimJL
sol = mle(prob, Optim.LBFGS())
LikelihoodSolution. retcode: Success
Maximum likelihood: 957.6376683220673
Maximum likelihood estimates: 5-element Vector{Float64}
     σ: 0.049045771053511954
     β₀: -1.0041730424101303
     β₁: 1.006051999753723
     β₂: 0.5041343138021581
     β₃: 2.9922041467801934
```

We can compare these MLEs to the true MLES $\hat{\beta} = (\boldsymbol X^{\mathsf T}\boldsymbol X)^{-1}\boldsymbol X^{\mathsf T}\boldsymbol y$ and $\hat\sigma^2 = (1/n_d)(\boldsymbol y - \boldsymbol X\boldsymbol \beta)^{\mathsf T}(\boldsymbol y - \boldsymbol X\boldsymbol \beta)$, where $n_d$ is the degrees of freedom, as follows (note the indexing):

```julia
using Test, LinearAlgebra
df = n - (length(β) + 1)
resids = y .- X * sol[2:5]
@test sol[2:5] ≈ inv(X' * X) * X' * y # sol[i] = sol.mle[i] 
@test sol[:σ]^2 ≈ 1 / df * sum(resids .^ 2) atol = 1e-4 # symbol indexing
```

We can now profile the results. In this case, since the problem has no bounds for some parameters we need to manually define the parameter bounds used for profiling. The function `construct_profile_ranges` is used for this.

```julia
lb = [1e-12, -5.0, -5.0, -5.0, -5.0]
ub = [15.0, 15.0, 15.0, 15.0, 15.0]
resolutions = [600, 200, 200, 200, 200] # use many points for σ
param_ranges = construct_profile_ranges(sol, lb, ub, resolutions)
prof = profile(prob, sol; param_ranges)
ProfileLikelihoodSolution. MLE retcode: Success
Confidence intervals: 
     95.0% CI for σ: (0.04639652142575396, 0.05196200098682017)
     95.0% CI for β₀: (-1.0133286782651982, -0.9950163004240635)
     95.0% CI for β₁: (0.9906172772151969, 1.021486501403717)
     95.0% CI for β₂: (0.4960199617761438, 0.5122490969333844)
     95.0% CI for β₃: (2.9786181979880935, 3.0057902255444287)
```

These confidence intervals can be compared to the true confidence intervals as follows, noting that the variance-covariance matrix for the $\beta_i$ coefficients is $\boldsymbol\Sigma = \sigma^2(\boldsymbol X^{\mathsf T}\boldsymbol X)^{-1}$ so that their confidence interval is $\hat\beta_i \pm 1.96\sqrt{\boldsymbol\Sigma_{ii}}$. Additionally, a confidence interval for $\sigma$ is $\sqrt{(\boldsymbol y - \boldsymbol X\boldsymbol \beta)^{\mathsf T}(\boldsymbol y - \boldsymbol X\boldsymbol \beta)}(1/\sqrt{\chi_{0.975, n_d}}, 1/\sqrt{\chi_{0.025, n_d}})$.

```julia
vcov_mat = sol[:σ]^2 * inv(X' * X)
for i in 1:4
    @test prof.confidence_intervals[i+1][1] ≈ sol.mle[i+1] - 1.96sqrt(vcov_mat[i, i]) atol = 1e-3
    @test prof.confidence_intervals[i+1][2] ≈ sol.mle[i+1] + 1.96sqrt(vcov_mat[i, i]) atol = 1e-3
end
rss = sum(resids .^ 2)
χ²_up = quantile(Chisq(df), 0.975)
χ²_lo = quantile(Chisq(df), 0.025)
σ_CI_exact = sqrt.(rss ./ (χ²_up, χ²_lo))
@test get_confidence_intervals(prof, :σ).lower ≈ σ_CI_exact[1] atol = 1e-3
@test ProfileLikelihood.get_upper(get_confidence_intervals(prof, :σ)) ≈ σ_CI_exact[2] atol = 1e-3
```

You can use `prof` to view a single parameter's results, e.g.

```julia
prof[:β₂]
Profile likelihood for parameter β₂. MLE retcode: Success
MLE: 0.5041343138021581
95.0% CI for β₂: (0.4960199617761438, 0.5122490969333844)
```

You can also evaluate the profile at a point inside its confidence interval. (If you want to evaluate outside the confidence interval, you need to use a non-`Throw` `extrap` in the `profile` function's keyword argument [see also Interpolations.jl].) The following are all the same, evaluating the profile for $\beta_2$ at $\beta_2=0.5$:

```julia
prof[:β₂](0.50)
prof(0.50, :β₂)
prof(0.50, 4)
```

We can now also visualise the results. In the plot below, the red line is at the threshold for the confidence region, so that the parameters between these values define the confidence interval. The red lines are at the MLEs, and the black lines are at the true values. 

```julia 
using CairoMakie, LaTeXStrings
fig = plot_profiles(prof;
    latex_names=[L"\sigma", L"\beta_0", L"\beta_1", L"\beta_2", L"\beta_3"], # default names would be of the form θᵢ
    show_mles=true,
    shade_ci=true,
    true_vals=[σ, β...],
    fig_kwargs=(fontsize=30, resolution=(2134.0f0, 906.0f0)),
    axis_kwargs=(width=600, height=300))
xlims!(fig.content[1], 0.045, 0.055) # fix the ranges
xlims!(fig.content[2], -1.025, -0.975)
xlims!(fig.content[4], 0.475, 0.525)
```

![Regression profiles](https://github.com/DanielVandH/ProfileLikelihood.jl/blob/main/test/figures/regression_profiles.png?raw=true)

You could also plot individual or specific parameters:

```julia
plot_profiles(prof, [1, 3]) # plot σ and β₁
plot_profiles(prof, [:σ, :β₁, :β₃]) # can use symbols 
plot_profiles(prof, 1) # can just provide an integer 
plot_profiles(prof, :β₂) # symbols work
```

## Logistic ordinary differential equation

Now let us consider the logistic ordinary differential equation (ODE). For ODEs, our treatment is as follows: Let us have some ODE $\mathrm dy/\mathrm dt = f(y, t; \boldsymbol \theta)$ for some parameters $\boldsymbol\theta$ of interest. We will suppose that we have some data $y_i^o$ at time $t_i$, $i=1,\ldots,n$, with initial condition $y_0^o$ at time $t_0=0$, which we model according to a normal distribution $y_i^o \mid \boldsymbol \theta \sim \mathcal N(y_i(\boldsymbol \theta), \sigma^2)$, $i=0,1,\ldots,n$, where $y_i$ is a solution of the ODE at time $t_i$. This defines a likelihood that we can use for estimating the parameters.

Let us now proceed with our example. We are considering $\mathrm du/\mathrm dt = \lambda u(1-u/K)$, $u(0)=u_0$, and our interest is in estimating $(\lambda, K, u_0)$, and also the variance of the noise $\sigma$. The exact solution to this problem is $u(t) = Ku_0/[(K-u_0)\mathrm{e}^{-\lambda t} + u_0]$ which we use for generating noisy data.

```julia
Random.seed!(2929911002)
u₀, λ, K, n, T = 0.5, 1.0, 1.0, 100, 10.0
t = LinRange(0, T, n)
u = @. K * u₀ * exp(λ * t) / (K - u₀ + u₀ * exp(λ * t))
σ = 0.1
uᵒ = u .+ [0.0, σ * randn(length(u) - 1)...] # add some noise 
```

Now having our data, we define the ODE and the likelihood function.

```julia 
@inline function ode_fnc(u, p, t)
    local λ, K
    λ, K = p
    du = λ * u * (1 - u / K)
    return du
end
@inline function loglik_fnc(θ, data, integrator)
    local uᵒ, n, λ, K, σ, u0
    uᵒ, n = data
    λ, K, σ, u0 = θ
    integrator.p[1] = λ
    integrator.p[2] = K
    reinit!(integrator, u0)
    solve!(integrator)
    return gaussian_loglikelihood(uᵒ, integrator.sol.u, σ, n)
end
```

Now we can define our problem. We constrain the problem so that $0 \leq \lambda \leq 10$, $10^{-6} \leq K \leq 10$, $10^{-6} \leq \sigma \leq 10$, and $0 \leq u_0 \leq 10$.

```julia
using FiniteDiff, OrdinaryDiffEq
θ₀ = [0.7, 2.0, 0.15, 0.4]
lb = [0.0, 1e-6, 1e-6, 0.0]
ub = [10.0, 10.0, 10.0, 10.0]
syms = [:λ, :K, :σ, :u₀]
prob = LikelihoodProblem(
    loglik_fnc, θ₀, ode_fnc, u₀, (0.0, T); # u₀ is just a placeholder IC in this case
    syms=syms,
    data=(uᵒ, n),
    ode_parameters=[1.0, 1.0], # temp values for [λ, K],
    ode_kwargs=(verbose=false, saveat=t),
    f_kwargs=(adtype=Optimization.AutoFiniteDiff(),),
    prob_kwargs=(lb=lb, ub=ub),
    ode_alg=Tsit5()
)
```

Now we find the MLEs.

```julia
using OptimizationNLopt
sol = mle(prob, NLopt.LN_BOBYQA; abstol=1e-16, reltol=1e-16)
LikelihoodSolution. retcode: Failure
Maximum likelihood: 86.54963187188535
Maximum likelihood estimates: 4-element Vector{Float64}
     λ: 0.7751485360202867
     K: 1.0214251327023145
     σ: 0.10183154994808913
     u₀: 0.5354121514863078
```

We can now profile. This time, we use a 90\% confidence interval.

```julia
prof = profile(prob, sol; conf_level=0.9)
ProfileLikelihoodSolution. MLE retcode: Failure
Confidence intervals: 
     90.0% CI for λ: (0.5447291688702172, 1.0597476103940162)
     90.0% CI for K: (0.9958363404632838, 1.0519310547041105)
     90.0% CI for σ: (0.09104488237974673, 0.11493628491475444)
     90.0% CI for u₀: (0.46273795601666323, 0.6067706857612565)
```

```julia
@test λ ∈ get_confidence_intervals(prof, :λ)
@test K ∈ prof.confidence_intervals[2]
@test σ ∈ get_confidence_intervals(prof[:σ])
@test u₀ ∈ get_confidence_intervals(prof, 4)
```

We can visualise as we did before:

```julia
fig = plot_profiles(prof;
    latex_names=[L"\lambda", L"K", L"\sigma", L"u_0"],
    show_mles=true,
    shade_ci=true,
    true_vals=[λ, K, σ, u₀],
    fig_kwargs=(fontsize=30, resolution=(1410.0f0, 880.0f0)),
    axis_kwargs=(width=600, height=300))
```

![Logistic profiles](https://github.com/DanielVandH/ProfileLikelihood.jl/blob/main/test/figures/logistic_example.png?raw=true)

## Linear exponential ODE and grid searching

Now we consider $\mathrm dy/\mathrm dt = \lambda y$, $y(0) = y_0$. This has solution $y(t) = y_0\mathrm{e}^{\lambda t}$. Let us start by defining the data and the likelihood problem:

```julia
## Step 1: Generate some data for the problem and define the likelihood
Random.seed!(2992999)
λ = -0.5
y₀ = 15.0
σ = 0.5
T = 5.0
n = 200
Δt = T / n
t = [j * Δt for j in 0:n]
y = y₀ * exp.(λ * t)
yᵒ = y .+ [0.0, rand(Normal(0, σ), n)...]
@inline function ode_fnc(u, p, t)
    local λ
    λ = p
    du = λ * u
    return du
end
@inline function loglik_fnc(θ, data, integrator)
    local yᵒ, n, λ, σ, u0
    yᵒ, n = data
    λ, σ, u0 = θ
    integrator.p = λ
    ## Now solve the problem 
    reinit!(integrator, u0)
    solve!(integrator)
    return gaussian_loglikelihood(yᵒ, integrator.sol.u, σ, n)
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

Let us now give an alternative way of exploring this likelihood function. We have been using `mle`, but we also provide some capability for using a grid search, which can sometimes be useful for e.g. visualising a likelihood function or obtaining initial estmiates for parameters (although it scales terribly for problems with more than even three parameters). Below we define a `RegularGrid`, a regular grid for each parameter:

```julia
regular_grid = RegularGrid(lb, ub, 50) # resolution can also be given as a vector for each parameter
```

We can now use this grid to evaluate the likelihood function at each point, and then return the maximum values (use `save_vals=Val(true)` if you want all the computed values as an array, given as a second argument; also see `?grid_search`).

```julia
gs = grid_search(prob, regular_grid)
LikelihoodSolution. retcode: Success
Maximum likelihood: -237.80342827421936
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
plan, _ = LHCoptim(500, d, gens)
new_lb = [-2.0, 10.0, 0.05]
new_ub = [2.0, 20.0, 0.2]
bnds = [(new_lb[i], new_ub[i]) for i in 1:d]
parameter_vals = Matrix(scaleLHC(plan, bnds)') # transpose so that a column is a parameter set 
irregular_grid = IrregularGrid(lb, ub, parameter_vals)
gs_ir, loglik_vals_ir = grid_search(prob, irregular_grid; save_vals=Val(true))
```
```julia
LikelihoodSolution. retcode: Success
Maximum likelihood: -687.9525589767204
Maximum likelihood estimates: 3-element Vector{Float64}
     λ: 0.6693386773547094
     σ: 10.040080160320642
     y₀: 0.08907815631262525
```
```julia
max_lik, max_idx = findmax(loglik_vals_ir)
@test max_lik == PL.get_maximum(gs_ir)
@test parameter_vals[:, max_idx] ≈ PL.get_mle(gs_ir)
```

(If you just want to try many points for starting your optimiser, see the optimiser in MultistartOptimization.jl.)

Now let's use `mle`. We will restart the initial guess to use the estimates from our grid search.

```julia
prob = update_initial_estimate(prob, gs)
sol = mle(prob, Optim.LBFGS())
```

Now we profile.

```julia
prof = profile(prob, sol; f_abstol=1e-6)
```
```julia
ProfileLikelihoodSolution. MLE retcode: Success
Confidence intervals: 
     95.0% CI for λ: (-0.5138791517539231, -0.4919599635444335)
     95.0% CI for σ: (0.44145726606891256, 0.5371526631048306)
     95.0% CI for y₀: (14.987116289320653, 15.42457150183812)
```
```julia
@test λ ∈ get_confidence_intervals(prof, :λ)
@test σ ∈ get_confidence_intervals(prof[:σ])
@test y₀ ∈ get_confidence_intervals(prof, 3)
```

Finally, we can visualise the profiles:

```julia
fig = plot_profiles(prof; nrow=1, ncol=3,
    latex_names=[L"\lambda", L"\sigma", L"y_0"],
    true_vals=[λ, σ, y₀],
    fig_kwargs=(fontsize=30, resolution=(2109.644f0, 444.242f0)),
    axis_kwargs=(width=600, height=300))
```

![Linear exponential profiles](https://github.com/DanielVandH/ProfileLikelihood.jl/blob/main/test/figures/linear_exponential_example.png?raw=true)