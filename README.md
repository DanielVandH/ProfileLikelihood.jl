# ProfileLikelihood 

- [ProfileLikelihood](#profilelikelihood)
- [Interface](#interface)
  - [Defining the problem: LikelihoodProblem](#defining-the-problem-likelihoodproblem)
  - [Solving the problem: mle and LikelihoodSolution](#solving-the-problem-mle-and-likelihoodsolution)
  - [Profiling the parameters: profile and ProfileLikelihoodSolution](#profiling-the-parameters-profile-and-profilelikelihoodsolution)
  - [Propagating uncertainty: Prediction intervals](#propagating-uncertainty-prediction-intervals)
- [Examples](#examples)
  - [Multiple linear regression](#multiple-linear-regression)
  - [Logistic ordinary differential equation](#logistic-ordinary-differential-equation)
    - [Prediction intervals](#prediction-intervals)
  - [Linear exponential ODE and grid searching](#linear-exponential-ode-and-grid-searching)
  - [Diffusion equation on a square plate](#diffusion-equation-on-a-square-plate)
    - [Building the FVMProblem](#building-the-fvmproblem)
    - [Defining a summary statistic](#defining-a-summary-statistic)
    - [Defining the LikelihoodProblem](#defining-the-likelihoodproblem)
    - [Parameter estimation](#parameter-estimation)
    - [Reducing to two parameters and grid searching](#reducing-to-two-parameters-and-grid-searching)
  - [Prediction intervals for the mass](#prediction-intervals-for-the-mass)
- [Mathematical and Implementation Details](#mathematical-and-implementation-details)
  - [Computing the profile likelihood function](#computing-the-profile-likelihood-function)
  - [Computing prediction intervals](#computing-prediction-intervals)

This module defines the routines required for computing maximum likelihood estimates and profile likelihoods. The optimisation routines are built around the [Optimization.jl](https://github.com/SciML/Optimization.jl) interface, allowing us to e.g. easily switch between algorithms, between finite differences and automatic differentiation, and it allows for constraints to be defined with ease. Below we list the definitions we are using for likelihoods and profile likelihoods. This code only works for scalar parameters of interest (i.e. out of a vector $\boldsymbol \theta$, you can profile a single scalar parameter $\theta_i \in \boldsymbol\theta$) for now.

**Definition: Likelihood function** (see Casella & Berger, 2002): Let $f(\boldsymbol x \mid \boldsymbol \theta)$ denote the joint probability density function (PDF) of the sample $\boldsymbol X = (X_1,\ldots,X_n)^{\mathsf T}$, where $\boldsymbol \theta \in \Theta$ is some set of parameters and $\Theta$ is the parameter space. We define the _likelihood function_ $\mathcal L \colon \Theta \to [0, \infty)$ by $\mathcal L(\boldsymbol \theta \mid \boldsymbol x) = f(\boldsymbol x \mid \boldsymbol \theta)$ for some realisation $\boldsymbol x = (x_1,\ldots,x_n)^{\mathsf T}$ of $\boldsymbol X$. The _log-likelihood function_ $\ell\colon\Theta\to\mathbb R$ is defined by $\ell(\boldsymbol \theta \mid \boldsymbol x) =  \log\mathcal L(\boldsymbol\theta \mid \boldsymbol x)$.The _maximum likelihood estimate_ (MLE) $\hat{\boldsymbol\theta}$ is the parameter $\boldsymbol\theta$ that maximises the likelihood function, $\hat{\boldsymbol{\theta}} = argmax_{\boldsymbol{\theta} \in \Theta} \mathcal{L}(\boldsymbol{\theta} \mid \boldsymbol x) = argmax_{\boldsymbol\theta \in \Theta} \ell(\boldsymbol\theta \mid \boldsymbol x)$.

**Definition: Profile likelihood function** (see Pawitan, 2001): Suppose we have some parameters of interest, $\boldsymbol \theta \in \Theta$, and some nuisance parameters, $\boldsymbol \phi \in \Phi$, and some data $\boldsymbol x = (x_1,\ldots,x_n)^{\mathsf T}$, giving some joint likelihood $\mathcal L \colon \Theta \cup \Phi \to [0, \infty)$ defined by $\mathcal L(\boldsymbol\theta, \boldsymbol\phi \mid \boldsymbol x)$. We define the _profile likelihood_ $\mathcal L_p \colon \Theta \to [0, \infty)$ of $\boldsymbol\theta$ by $\mathcal L_p(\boldsymbol\theta \mid \boldsymbol x) = \sup_{\boldsymbol \phi \in \Phi \mid \boldsymbol \theta} \mathcal L(\boldsymbol \theta, \boldsymbol \phi \mid \boldsymbol x)$. The _profile log-likelihood_ $\ell_p \colon \Theta \to \mathbb R$ of $\boldsymbol\theta$ is defined by $\ell_p(\boldsymbol \theta \mid \boldsymbol x) = \log \mathcal L_p(\boldsymbol\theta \mid \boldsymbol x)$. The _normalised profile likelihood_ is defined by $\hat{\mathcal L}_p(\boldsymbol\theta \mid \boldsymbol x) = \mathcal L_p(\boldsymbol \theta \mid \boldsymbol x) - \mathcal L_p(\hat{\boldsymbol\theta} \mid \boldsymbol x)$, where $\hat{\boldsymbol\theta}$ is the MLE of $\boldsymbol\theta$, and similarly for the normalised profile log-likelihood.

From Wilk's theorem, we know that $2\hat{\ell}\_p(\boldsymbol\theta \mid \boldsymbol x) \geq -\chi_{p, 1-\alpha}^2$ is an approximate $100(1-\alpha)\%$ confidence region for $\boldsymbol \theta$, and this enables us to obtain confidence intervals for parameters by considering only their profile likelihood, where $\chi_{p,1-\alpha}^2$ is the $1-\alpha$ quantile of the $\chi_p^2$ distribution and $p$ is the length of $\boldsymbol\theta$. For the case of a scalar parameter of interest, $-\chi_{1, 0.95}^2 \approx -1.92$.

We compute the profile log-likelihood in this package by starting at the MLE, and stepping left/right until we reach a given threshold. The code is iterative to not waste time in so much of the parameter space.

More detail about the methods we use in this package is given at the very end in the Mathematical and Implementation Details section.

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
struct LikelihoodSolution{N,Θ,P,M,R,A} <: AbstractLikelihoodSolution{N,P}
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

Here, the parameter values used for each parameter are given in `parameter_values`, with parameter indices (or symbols) mapped to these values. Similarly, the values of the profile log-likelihood are stored in `profile_values`. We use a spline (see Interpolations.jl) to make the profile log-likelihood a continuous function, and these splines are given by `splines`. Next, the computed confidence intervals are given in `confidence_intervals`, with a confidence interval represented by a `ConfidenceInterval` struct. Lastly, since computing the profile log-likelihood function requires an optimisation problem with one variable fixed and the others free, we obtain for each profile log-likelihood value a set of optimised parameters -- these parameters are given in `other_mles`.

If `prof` is a `ProfileLikelihoodSolution`, then you can also call it as e.g. `prof(0.5, 1)` to evaluate the profile log-likelihood function of the first parameter at the point `0.5`. Alternatively, `prof(0.7, :α)` does the same but for the parameter `:α` at the point `0.7`. You can also index `prof` at a specific index (or symbol) to see the results only for that parameter, e.g. `prof[1]` or `prof[:α]`; this returns a `ProfileLikelihoodSolutionView`.

## Propagating uncertainty: Prediction intervals 

The confidence intervals obtained from profiling can be used to obtain approximate prediction intervals via *profile-wise profile likelihoods*, as defined e.g. in [Simpson and Maclaren (2022)](https://doi.org/10.1101/2022.12.14.520367), for a prediction function $\boldsymbol q(\boldsymbol\theta)$. These intervals can be based on varying a single parameter, or by taking the union of individual prediction intervals. The main function for this is `get_prediction_intervals`. Rather than explain in full detail here, please refer to the second example below (the logistic ODE example), where we reproduce the first case study of [Simpson and Maclaren (2022)](https://doi.org/10.1101/2022.12.14.520367).

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

The data `y` is now our noisy data. The likelihood function in this example is 

$$
\ell(\sigma, \boldsymbol \beta \mid \boldsymbol y) = -(n/2)\log(2\mathrm{\pi}\sigma^2) - (1/2\sigma^2)\sum_i (y_i - \beta_0 - \beta_1x_{1i} - \beta_2x_{2i} - \beta_3x_{1i}x_{2i})^2. 
$$ 

We now define our likelihood function. To allow for automatic differentiation, we use `PreallocationTools.DiffCache` to define our cache vectors.

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

We can now profile the results. In this case, since the problem has no bounds for some parameters we need to manually define the parameter bounds used for profiling. The function `construct_profile_ranges` is used for this. Note that we use `parallel = true` below to allow for multithreading, allowing multiple parameters to be profiled at the same time.

```julia
lb = [1e-12, -5.0, -5.0, -5.0, -5.0]
ub = [15.0, 15.0, 15.0, 15.0, 15.0]
resolutions = [600, 200, 200, 200, 200] # use many points for σ
param_ranges = construct_profile_ranges(sol, lb, ub, resolutions)
prof = profile(prob, sol; param_ranges, parallel=true)
ProfileLikelihoodSolution. MLE retcode: Success
Confidence intervals: 
     95.0% CI for σ: (0.04639652142575396, 0.05196200098682017)
     95.0% CI for β₀: (-1.013328678265197, -0.9950163004240635)
     95.0% CI for β₁: (0.9906172772152076, 1.0214865014037124)
     95.0% CI for β₂: (0.4960199617761395, 0.5122490969333844)
     95.0% CI for β₃: (2.978618197988093, 3.0057902255444136)
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

The following example comes from the first case study of [Simpson and Maclaren (2022)](https://doi.org/10.1101/2022.12.14.520367).

Now let us consider the logistic ordinary differential equation (ODE). For ODEs, our treatment is as follows: Let us have some ODE $\mathrm dy/\mathrm dt = f(y, t; \boldsymbol \theta)$ for some parameters $\boldsymbol\theta$ of interest. We will suppose that we have some data $y_i^o$ at time $t_i$, $i=1,\ldots,n$, with initial condition $y_0^o$ at time $t_0=0$, which we model according to a normal distribution $y_i^o \mid \boldsymbol \theta \sim \mathcal N(y_i(\boldsymbol \theta), \sigma^2)$, $i=0,1,\ldots,n$, where $y_i$ is a solution of the ODE at time $t_i$. This defines a likelihood that we can use for estimating the parameters.

Let us now proceed with our example. We are considering $\mathrm du/\mathrm dt = \lambda u(1-u/K)$, $u(0)=u_0$, and our interest is in estimating $(\lambda, K, u_0)$, we will fix the standard deviation of the noise, $\sigma$, at $\sigma=10$. Note that the exact solution to this ODE is $u(t) = Ku_0/[(K-u_0)\mathrm{e}^{-\lambda t} + u_0]$.

The first step is to generate the data.
```julia 
using OrdinaryDiffEq, Random
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
# Initial data is obtained by solving the ODE 
tspan = extrema(t)
p = (λ, K)
prob = ODEProblem(ode_fnc, u₀, tspan, p)
sol = solve(prob, Rosenbrock23(), saveat=t)
Random.seed!(2828)
uᵒ = sol.u + σ * randn(length(t))
```

Now having our data, we define the likelihood function.

```julia 
@inline function loglik_fnc2(θ, data, integrator)
    λ, K, u₀ = θ
    uᵒ, σ = data
    integrator.p[1] = λ
    integrator.p[2] = K
    reinit!(integrator, u₀)
    solve!(integrator)
    return gaussian_loglikelihood(uᵒ, integrator.sol.u, σ, length(uᵒ))
end
```

Now we can define our problem. We constrain the problem so that $0 \leq \lambda \leq 0.05$, $50 \leq K \leq 150$, and $0 \leq u_0 \leq 50$.

```julia
lb = [0.0, 50.0, 0.0] # λ, K, u₀
ub = [0.05, 150.0, 50.0]
θ₀ = [λ, K, u₀]
syms = [:λ, :K, :u₀]
prob = LikelihoodProblem(
    loglik_fnc2, θ₀, ode_fnc, u₀, maximum(t); # Note that u₀ is just a placeholder IC in this case since we are estimating it
    syms=syms,
    data=(uᵒ, σ),
    ode_parameters=[1.0, 1.0], # temp values for [λ, K]
    ode_kwargs=(verbose=false, saveat=t),
    f_kwargs=(adtype=Optimization.AutoFiniteDiff(),),
    prob_kwargs=(lb=lb, ub=ub),
    ode_alg=Rosenbrock23()
)
```

Now we find the MLEs.

```julia
using OptimizationNLopt
sol = mle(prob, NLopt.LD_LBFGS)
LikelihoodSolution. retcode: Failure
Maximum likelihood: -38.99053694428977
Maximum likelihood estimates: 3-element Vector{Float64}
     λ: 0.010438031266786045
     K: 99.59921873132551
     u₀: 8.098422110755225
```

We can now profile. 

```julia
prof = profile(prob, sol; alg=NLopt.LN_NELDERMEAD, parallel=false)
ProfileLikelihoodSolution. MLE retcode: Failure
Confidence intervals: 
     95.0% CI for λ: (0.006400992274213644, 0.01786032876226762)
     95.0% CI for K: (90.81154862835605, 109.54214763511888)
     95.0% CI for u₀: (1.5919805025139593, 19.070831536649305)
```

```julia
@test λ ∈ get_confidence_intervals(prof, :λ)
@test K ∈ prof.confidence_intervals[2]
@test u₀ ∈ get_confidence_intervals(prof, 3)
```

We can visualise as we did before:

```julia
using CairoMakie, LaTeXStrings
fig = plot_profiles(prof;
    latex_names=[L"\lambda", L"K", L"u_0"],
    show_mles=true,
    shade_ci=true,
    nrow=1,
    ncol=3,
    true_vals=[λ, K, u₀],
    fig_kwargs=(fontsize=30, resolution=(2109.644f0, 444.242f0)),
    axis_kwargs=(width=600, height=300))
```

![Logistic profiles](https://github.com/DanielVandH/ProfileLikelihood.jl/blob/main/test/figures/logistic_example.png?raw=true)

### Prediction intervals 

Let us now use these results to compute prediction intervals for $u(t)$. Following [Simpson and Maclaren (2022)](https://doi.org/10.1101/2022.12.14.520367), the idea is to use the profile likelihood to construct another profile likelihood, called the *profile-wise profile likelihood*, that allows us to obtain prediction intervals for some prediction function $q(\boldsymbol \theta)$. More detail is given in the mathematical details section.

The first step is to define a function $q(\boldsymbol\theta)$ that comptues our prediction given some parameters $\boldsymbol\theta$. The function in this case is simply:

```julia
function prediction_function(θ, data)
    λ, K, u₀ = θ
    t = data
    prob = ODEProblem(ode_fnc, u₀, extrema(t), (λ, K))
    sol = solve(prob, Rosenbrock23(), saveat=t)
    return sol.u
end
```

Note that the second argument `data` allows for extra parameters to be passed. To now obtain prediction intervals for `sol.u`, for each `t`, we define a large grid for `t` and use `get_prediction_intervals`:

```julia
t_many_pts = LinRange(extrema(t)..., 1000)
parameter_wise, union_intervals, all_curves, param_range =
    get_prediction_intervals(prediction_function, prof,
        t_many_pts; q_type=Vector{Float64})
# t_many_pts is the `data` argument, it doesn't have to be time for other problems
```

This function `get_prediction_intervals` has four outputs:

- `parameter_wise`: These are prediction intervals for the prediction at each point $t$, coming from the profile likelihood of each respective parameter:

```julia 
julia> parameter_wise
Dict{Int64, Vector{ConfidenceInterval{Float64, Float64}}} with 3 entries:
  2 => [ConfidenceInterval{Float64, Float64}(6.45444, 12.0694, 0.95), ConfidenceInterval{Float64, Float64}(6.52931, 12.1545, 0.95), ConfidenceInterval{Float64, Float64}(6.60498, 12.24, 0.95), ConfidenceInterva…  3 => [ConfidenceInterval{Float64, Float64}(1.59389, 19.0709, 0.95), ConfidenceInterval{Float64, Float64}(1.621, 19.1773, 0.95), ConfidenceInterval{Float64, Float64}(1.64856, 19.2842, 0.95), ConfidenceInterva…  1 => [ConfidenceInterval{Float64, Float64}(1.86302, 17.5828, 0.95), ConfidenceInterval{Float64, Float64}(1.89596, 17.6768, 0.95), ConfidenceInterval{Float64, Float64}(1.92948, 17.7712, 0.95), ConfidenceInter…
``` 

For example, `parameter_wise[1]` comes from varying $\lambda$, with the parameters $K$ and $u_0$ coming from optimising the likelihood function with $\lambda$ fixed.

- `union_intervals`: These are prediction intervals at each point $t$ coming from taking the union of the intervals from the corresponding elements of `parameter_wise`.

- `all_curves`: The intervals come from taking extrema over many curves. This is a `Dict` mapping parameter indices to the curves that were used, with `all_curves[i][j]` being the set of curves for the `i`th parameter (e.g. `i=1` is for $\lambda$) and the `j`th parameter.

- `param_range`: The curves come from evaluating the prediction function between the bounds of the confidence intervals for each parameter, and this output gives the parameters used, so that e.g. `all_curves[i][j]` uses `param_range[i][j]` for the value of the `i`th parameter.

Let us now use these outputs to visualise the prediction intervals. First, let us extract the solution with the true parameter values and with the MLEs.

```julia 
exact_soln = prediction_function([λ, K, u₀], t_many_pts)
mle_soln = prediction_function(get_mle(sol), t_many_pts)
```

Now let us plot the prediction intervals coming from each parameter, and from the union of all intervals (not shown yet, see below).

```julia
fig = Figure(fontsize=38, resolution=(1402.7681f0, 848.64404f0))
alp = join('a':'z')
latex_names = [L"\lambda", L"K", L"u_0"]
for i in 1:3
    ax = Axis(fig[i < 3 ? 1 : 2, i < 3 ? i : 1], title=L"(%$(alp[i])): Profile-wise PI for %$(latex_names[i])",
        titlealign=:left, width=600, height=300)
    [lines!(ax, t_many_pts, all_curves[i][j], color=:grey) for j in eachindex(param_range[1])]
    lines!(ax, t_many_pts, exact_soln, color=:red)
    lines!(ax, t_many_pts, mle_soln, color=:blue, linestyle=:dash)
    lines!(ax, t_many_pts, getindex.(parameter_wise[i], 1), color=:black, linewidth=3)
    lines!(ax, t_many_pts, getindex.(parameter_wise[i], 2), color=:black, linewidth=3)
end
ax = Axis(fig[2, 2], title=L"(d):$ $ Union of all intervals",
    titlealign=:left, width=600, height=300)
band!(ax, t_many_pts, getindex.(union_intervals, 1), getindex.(union_intervals, 2), color=:grey)
lines!(ax, t_many_pts, getindex.(union_intervals, 1), color=:black, linewidth=3)
lines!(ax, t_many_pts, getindex.(union_intervals, 2), color=:black, linewidth=3)
lines!(ax, t_many_pts, exact_soln, color=:red)
lines!(ax, t_many_pts, mle_soln, color=:blue, linestyle=:dash)
```

To now assess the coverage of these intervals, we want to compare them to the interval coming from the full likelihood. We find this interval by taking a large number of parameters, and finding all of them for which the normalised log-likelihood exceeds the threshold $-1.92$. We then take the parameters that give a value exceeding this threshold, compute the prediction function at these values, and then take the extrema. The code below uses the function `grid_search` that evaluates the function at many points, and we describe this function in more detail in the next example.

```julia
lb = get_lower_bounds(prob)
ub = get_upper_bounds(prob)
N = 1e5
grid = [[lb[i] + (ub[i] - lb[i]) * rand() for _ in 1:N] for i in 1:3]
grid = permutedims(reduce(hcat, grid), (2, 1))
ig = IrregularGrid(lb, ub, grid)
gs, lik_vals = grid_search(prob, ig; parallel=Val(true), save_vals=Val(true))
lik_vals .-= get_maximum(sol) # normalised 
feasible_idx = findall(lik_vals .> ProfileLikelihood.get_chisq_threshold(0.95)) # values in the confidence region 
parameter_evals = grid[:, feasible_idx]
q = [prediction_function(θ, t_many_pts) for θ in eachcol(parameter_evals)]
q_mat = reduce(hcat, q)
q_lwr = minimum(q_mat; dims=2) |> vec
q_upr = maximum(q_mat; dims=2) |> vec
lines!(ax, t_many_pts, q_lwr, color=:magenta, linewidth=3)
lines!(ax, t_many_pts, q_upr, color=:magenta, linewidth=3)
```

![Logistic prediction intervals](https://github.com/DanielVandH/ProfileLikelihood.jl/blob/main/test/figures/logistic_example_prediction.png?raw=true)

The first plot shows that the profile-wise prediction interval for $\lambda$ is quite large when $t$ is small, and then small for large time. This makes sense since the large time solution is independent of $\lambda$ (the large time solution is $u_s(t)=K$). For $K$, we see that the profile-wise interval only becomes large for large time, which again makes sense. For $u_0$ we see similar behaviour as for $\lambda$. Finally, taking the union over all the nitervals, as is done in (d), shows that we fully enclose the solution coming from the MLE, as well as the true curve. The magenta curve shows the results from the full likelihood function, and is reasonably close to the approximate interval obtained from the union.

## Linear exponential ODE and grid searching

Now we consider $\mathrm dy/\mathrm dt = \lambda y$, $y(0) = y_0$. This has solution $y(t) = y_0\mathrm{e}^{\lambda t}$. Let us start by defining the data and the likelihood problem:

```julia
## Step 1: Generate some data for the problem and define the likelihood
Random.seed!(2992999)
λ = -0.5
y₀ = 15.0
σ = 0.5
T = 5.0
n = 450
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
using LoopVectorization, MuladdMacro
@inline function _loglik_fnc(θ::AbstractVector{T}, data, integrator) where {T}
    local yᵒ, n, λ, σ, u0
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

Let us now give an alternative way of exploring this likelihood function. We have been using `mle`, but we also provide some capability for using a grid search, which can sometimes be useful for e.g. visualising a likelihood function or obtaining initial estmiates for parameters (although it scales terribly for problems with more than even three parameters). Below we define a `RegularGrid`, a regular grid for each parameter:

```julia
regular_grid = RegularGrid(lb, ub, 50) # resolution can also be given as a vector for each parameter
```

We can now use this grid to evaluate the likelihood function at each point, and then return the maximum values (use `save_vals=Val(true)` if you want all the computed values as an array, given as a second argument; also see `?grid_search`). (You can also set `parallel = Val(true)` so that the computation is done with multithreading.)

```julia
gs = grid_search(prob, regular_grid)
LikelihoodSolution. retcode: Success
Maximum likelihood: -547.9579886200935
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
new_lb = [-2.0, 0.05, 10.0]
new_ub = [2.0, 0.2, 20.0]
bnds = [(new_lb[i], new_ub[i]) for i in 1:d]
parameter_vals = Matrix(scaleLHC(plan, bnds)') # transpose so that a column is a parameter set 
irregular_grid = IrregularGrid(lb, ub, parameter_vals)
gs_ir, loglik_vals_ir = grid_search(prob, irregular_grid; save_vals=Val(true), parallel = Val(true))
```
```julia
LikelihoodSolution. retcode: Success
Maximum likelihood: -1729.7407123603484
Maximum likelihood estimates: 3-element Vector{Float64}
     λ: -0.5090180360721444
     σ: 0.19368737474949904
     y₀: 15.791583166332664
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
prof = profile(prob, sol; alg=NLopt.LN_NELDERMEAD, parallel = true)
```
```julia
ProfileLikelihoodSolution. MLE retcode: Success
Confidence intervals: 
     95.0% CI for λ: (-0.51091362373969, -0.49491369219060505)
     95.0% CI for σ: (0.49607205632240814, 0.5652591835193789)
     95.0% CI for y₀: (14.98587355568687, 15.305179849533756)
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

## Diffusion equation on a square plate 

*Warning*: Much of the code in this example takes a very long time, e.g. the MLEs take just under an hour.

Let us now consider the problem of estimating parameters defining a diffusion equation on a square plate. In particular, consider 

$$
\begin{equation*}
\begin{array}{rcll}
\displaystyle
\frac{\partial u(x, y, t)}{\partial t} &=& \dfrac{1}{k}\boldsymbol{\nabla}^2 u(x, y, t) & (x, y) \in \Omega,t>0, \\
u(x, y, t) &= & 0 & (x, y) \in \partial \Omega,t>0, \\
u(x, y, 0) &= & u_0\mathbb{I}(y \leq c) &(x,y)\in\Omega,
\end{array}
\end{equation*}
$$

where $\Omega = [0, 2]^2$. This problem extends the corresponding example given in FiniteVolumeMethod.jl, namely [this example](https://github.com/DanielVandH/FiniteVolumeMethod.jl#diffusion-equation-on-a-square-plate), and so not all the code used in defining this PDE will be explained here; refer to the FiniteVolumeMethod.jl documentation. We will take the true values $k = 9$, $c = 1$, $u_0 = 50$, and let the standard deviation of the noise, $\sigma$, in the data be $0.1$. We are interested in recovering $(k, c, u_0)$; we do not consider estimating $\sigma$ here, estimating it leads to identifiability issues that distract from the main point of our example here, i.e. to just show how to setup a problem.

### Building the FVMProblem 

Let us start by defining the PDE problem, and then we will discuss profiling.

```julia 
using FiniteVolumeMethod, DelaunayTriangulation, LinearSolve
a, b, c, d = 0.0, 2.0, 0.0, 2.0
n = 500
x₁ = LinRange(a, b, n)
x₂ = LinRange(b, b, n)
x₃ = LinRange(b, a, n)
x₄ = LinRange(a, a, n)
y₁ = LinRange(c, c, n)
y₂ = LinRange(c, d, n)
y₃ = LinRange(d, d, n)
y₄ = LinRange(d, c, n)
x = reduce(vcat, [x₁, x₂, x₃, x₄])
y = reduce(vcat, [y₁, y₂, y₃, y₄])
xy = [[x[i], y[i]] for i in eachindex(x)]
unique!(xy)
x = getx.(xy)
y = gety.(xy)
r = 0.022
GMSH_PATH = "./gmsh-4.9.4-Windows64/gmsh.exe"
T, adj, adj2v, DG, points, BN = generate_mesh(x, y, r; gmsh_path=GMSH_PATH)
mesh = FVMGeometry(T, adj, adj2v, DG, points, BN)
bc = ((x, y, t, u::T, p) where {T}) -> zero(T)
type = :D
BCs = BoundaryConditions(mesh, bc, type, BN)
c = 1.0
u₀ = 50.0
f = (x, y) -> y ≤ c ? u₀ : 0.0
D = (x, y, t, u, p) -> p[1]
flux = (q, x, y, t, α, β, γ, p) -> (q[1] = -α / p[1]; q[2] = -β / p[1])
R = ((x, y, t, u::T, p) where {T}) -> zero(T)
initc = @views f.(points[1, :], points[2, :])
iip_flux = true
final_time = 0.1
k = [9.0]
prob = FVMProblem(mesh, BCs; iip_flux,
    flux_function=flux, reaction_function=R,
    initial_condition=initc, final_time,
    flux_parameters=k)
```

Our problem has now been defined. Notice that we wrap `k` in a vector so that we can easily mutate the `flux_parameters` field of `prob`; if `k` were a scalar, we could not mutate it.

Now let's generate some data. We start by solving the PDE.

```julia
alg = TRBDF2(linsolve=KLUFactorization(; reuse_symbolic=false))
sol = solve(prob, alg; specialization=SciMLBase.FullSpecialize, saveat=0.01)
```

(We use `reuse_symbolic=false` due to https://github.com/JuliaSparse/KLU.jl/issues/12 causing issues with multithreading later.) 

### Defining a summary statistic

Now, one complication with a PDE compared to the scalar ODE cases that we considered previously is that we have data at $(x_i, y_j, t_k)$ for many indices $(i, j, k)$. Rather than defining our objective function in terms of these data points, we will instead use a summary statistic. The summary statistic we use in this example is the average density,

$$
\tilde M(t) = \frac{1}{\mathrm{Area}(\Omega)}\iint_\Omega u(x, y, t)\mathrm{dA}. 
$$

We need to be able to compute this integral efficiently and accurately. For this, recall that the finite volume method discretises the domain into triangles. If $\mathcal T$ is this set of triangles, then 

$$ 
\tilde M(t) = \frac{1}{\mathrm{Area}(\Omega)}\sum_{T_k \in \mathcal T} \iint_{T_k} u(x, y, t)\mathrm{dA}. 
$$ 

Then, recall that $u$ is represented as a linear function $\alpha_k x + \beta_k y + \gamma_k$ inside the triangle $T_k$, thus 

$$ 
\tilde M(t) \approx \frac{1}{\mathrm{Area}(\Omega)}\sum_{T_k \in \mathcal T} \left[\alpha_k \iint_{T_k} x\mathrm{dA} + \beta_k \iint_{T_k} y\mathrm{dA} + \gamma_k\iint_{T_k}\mathrm{dA}\right] 
$$  

Now factoring out an $\mathrm{Area}(T_k) = \iint_{T_k}\mathrm{dA}$, 

$$ 
\tilde M(t) \approx \sum_{T_k \in \mathcal T} \frac{\mathrm{Area}(T_k)}{\mathrm{Area}(\Omega)}\left[\alpha_k \dfrac{\iint_{T_k} x\mathrm{dA}}{\iint_{T_k} \mathrm{dA}} + \beta_k \dfrac{\iint_{T_k} y\mathrm{dA}}{\iint_{T_k} \mathrm{dA}} + \gamma_k\right]. 
$$ 

Notice that the two ratios of integrals shown are simply $\hat x_k$ and $\hat y_k$, where $(\hat x_k, \hat y_k)$ is the centroid of $T_k$. Thus, the term in brackets is $\alpha_k \hat x_k + \beta_k \hat y_k + \gamma_k$, which is the approximation to $u$ at the centroid, $\tilde u(\hat x_k, \hat y_k, t)$. Thus, our approximation to the average density is 

$$ 
\tilde M(t) \approx \sum_{T_k \in \mathcal T} w_k \tilde u(\hat x_k, \hat y_k, t), \qquad w_k = \frac{\mathrm{Area}(T_k)}{\mathrm{Area}(\Omega)}. 
$$ 

The following function provides a method for computing this mass. 

```julia 
function compute_mass!(M::AbstractVector{T}, αβγ, sol, prob) where {T}
    mesh_area = prob.mesh.mesh_information.total_area
    fill!(M, zero(T))
    for i in eachindex(M)
        for V in FiniteVolumeMethod.get_elements(prob)
            element = FiniteVolumeMethod.get_element_information(prob.mesh, V)
            cx, cy = FiniteVolumeMethod.get_centroid(element)
            element_area = FiniteVolumeMethod.get_area(element)
            interpolant_val = eval_interpolant!(αβγ, prob, cx, cy, V, sol.u[i])
            M[i] += (element_area / mesh_area) * interpolant_val
        end
    end
    return nothing
end 
``` 

Let's now compute this mass and add some noise onto it. 

```julia 
using Random 
M = zeros(length(sol.t))
αβγ = zeros(3)
compute_mass!(M, αβγ, sol, prob)
true_M = deepcopy(M)
Random.seed!(29922881)
σ = 0.1
true_M .+= σ * randn(length(M))
``` 

### Defining the LikelihoodProblem

We now need to define the likelihood problem. We need to use the method for `LikelihoodProblem` that takes the `integrator` as an argument explicitly, so we must somehow construct an integrator from an `FVMProblem`. Here is one way that this can be done. Notice that we use `parallel=true` so that the PDE is solved with multithreading. For an isolated solution, this seems to solve the PDE twice as fast on my machine (eight threads).

```julia 
function ProfileLikelihood.construct_integrator(prob::FVMProblem, alg; ode_problem_kwargs, kwargs...)
    ode_problem = ODEProblem(prob; no_saveat=false, ode_problem_kwargs...)
    return ProfileLikelihood.construct_integrator(ode_problem, alg; kwargs...)
end
jac = float.(FiniteVolumeMethod.jacobian_sparsity(prob))
fvm_integrator = construct_integrator(prob, alg; ode_problem_kwargs=(jac_prototype=jac, saveat=0.01, parallel=true))
``` 

Now we define the likelihood function. 

```julia 
function loglik_fvm(θ::AbstractVector{T}, param, integrator) where {T}
    _k, _c, _u₀ = θ
    ## Update and solve
    (; prob) = param
    prob.flux_parameters[1] = _k
    pts = FiniteVolumeMethod.get_points(prob)
    for i in axes(pts, 2)
        pt = get_point(pts, i)
        prob.initial_condition[i] = gety(pt) ≤ _c ? _u₀ : zero(T)
    end
    reinit!(integrator, prob.initial_condition)
    solve!(integrator)
    if !SciMLBase.successful_retcode(integrator.sol)
        return typemin(T)
    end
    ## Compute the mass
    (; mass_data, mass_cache, shape_cache, sigma) = param
    compute_mass!(mass_cache, shape_cache, integrator.sol, prob)
    if any(isnan, mass_cache)
        return typemin(T)
    end
    ## Done 
    ℓ = @views gaussian_loglikelihood(mass_data, mass_cache, sigma, length(mass_data))
    return ℓ
end
``` 

Finally, here is the `LikelihoodProblem`.

```julia 
likprob = LikelihoodProblem(
    loglik_fvm,
    [8.54, 0.98, 29.83],
    fvm_integrator;
    syms=[:k, :c, :u₀],
    data=(prob=prob, mass_data=true_M, mass_cache=zeros(length(true_M)), shape_cache=zeros(3), sigma=σ),
    f_kwargs=(adtype=Optimization.AutoFiniteDiff(),),
    prob_kwargs=(lb=[3.0, 0.0, 0.0],
        ub=[15.0, 2.0, 250.0])
)
```

### Parameter estimation

Now that we have the problem completely setup, we are in a position for maximum likelihood estimation and profiling. For the maximum likelihood estimates, we first use a global optimiser and then we refine the solution with a local optimiser.

```julia 
mle_sol = mle(likprob, (NLopt.GN_DIRECT_L_RAND(), NLopt.LN_BOBYQA); ftol_abs=1e-8, ftol_rel=1e-8, xtol_abs=1e-8, xtol_rel=1e-8) # global, and then refine with a local algorithm
LikelihoodSolution. retcode: Failure
Maximum likelihood: 11.046014040624534
Maximum likelihood estimates: 3-element Vector{Float64}
     k: 7.847020395441574
     c: 1.1944331289720689
     u₀: 41.667309553688305
``` 
 
Next, let us profile. For interest, we show the difference in runtime when we use multithreading for profiling vs. when we do not use multithreading. I am using eight threads.

```julia 
@time prof = profile(likprob, mle_sol; alg=NLopt.LN_BOBYQA,
    ftol_abs=1e-4, ftol_rel=1e-4, xtol_abs=1e-4, xtol_rel=1e-4,
    resolution=60)
5131.960778 seconds (133.61 M allocations: 948.495 GiB, 0.13% gc time, 0.04% compilation time)
```

```julia 
@time _prof = profile(likprob, mle_sol; alg=NLopt.LN_BOBYQA,
    ftol_abs=1e-4, ftol_rel=1e-4, xtol_abs=1e-4, xtol_rel=1e-4,
    resolution=60, parallel=true)
3324.605865 seconds (131.24 M allocations: 948.598 GiB, 0.40% gc time, 0.01% compilation time)
```

The results are about twice as fast in this example. The reason it's not even faster is because we are also using multithreading in solving the PDE. If we had no used multithreading in solving the PDE, these results would take a significantly longer time. Here are the results from `prof` (same for `_prof`):

```julia 
ProfileLikelihoodSolution. MLE retcode: Failure
Confidence intervals: 
     95.0% CI for k: (7.4088716591304715, 8.574442050142432)
     95.0% CI for c: (0.6478281377475628, 2.0)
     95.0% CI for u₀: (33.78499567791489, 79.47955668442242)
```

See that all the true parameter intervals are inside these confidence intervals except for $k$, although $c$'s upper bound is right at the bounds we gave it in the problem. Let's now view the profile curves.

```julia 
using CairoMakie, LaTeXStrings
fig = plot_profiles(prof; nrow=1, ncol=3,
    latex_names=[L"k", L"c", L"u_0"],
    true_vals=[k[1], c, u₀],
    fig_kwargs=(fontsize=38, resolution=(2109.644f0, 444.242f0)),
    axis_kwargs=(width=600, height=300))
scatter!(fig.content[1], get_parameter_values(prof, :k), get_profile_values(prof, :k), color=:black, markersize=9)
scatter!(fig.content[2], get_parameter_values(prof, :c), get_profile_values(prof, :c), color=:black, markersize=9)
scatter!(fig.content[3], get_parameter_values(prof, :u₀), get_profile_values(prof, :u₀), color=:black, markersize=9)
xlims!(fig.content[1], 7.0, 9.5)
```

![PDE profiles](https://github.com/DanielVandH/ProfileLikelihood.jl/blob/main/test/figures/heat_pde_example.png?raw=true)

See that the profile curves for $c$ and $u_0$ are very flat, and we have not recovered $k$. This means that the parameters $c$ and $u_0$ are not *identifiable*, essentially meaning the data is not enough to recover these parameters. This is most likely because the mass $\tilde M(t)$ alone is not enough to uniquely define the solution. We could consider a summary statistic like 

$$ 
\mathcal S(t) = w\tilde M(t) + (1-w)\tilde A(t),
$$

for some $0 \leq w \leq 1$, where $\tilde A(t)$ is the area of the region below the leading edge of the solution, i.e. the area of the non-zero part of the solution. We do not consider this here. What we do consider is fixing $c$, keeping the summary statistic $\tilde M(t)$, and seeing what we can do with only two parameters $k$ and $u_0$.

### Reducing to two parameters and grid searching 

Let us now fix $c$ at its true value, $c = 1$, and consider estimating only $k$ and $u_0$. Since we have only $k$ and $u_0$ to estimate, it may be worthwhile to perform a grid search over our likelihood function so that we can (1) visualise the likelihood surface and (2) see reasonable estimates for $k$ and $u_0$. 

First, we redefine the problem.

```julia 
using StaticArraysCore
@inline function loglik_fvm_2(θ::AbstractVector{T}, param, integrator) where {T}
    _k, _u₀, = θ
    (; c) = param
    new_θ = SVector{3,T}((_k, c, _u₀))
    return loglik_fvm(new_θ, param, integrator)

end
likprob_2 = LikelihoodProblem(
    loglik_fvm_2,
    [8.54, 29.83],
    fvm_integrator;
    syms=[:k, :u₀],
    data=(prob=prob, mass_data=true_M, mass_cache=zeros(length(true_M)), shape_cache=zeros(3), sigma=σ, c=c),
    f_kwargs=(adtype=Optimization.AutoFiniteDiff(),),
    prob_kwargs=(lb=[3.0, 0.0],
        ub=[15.0, 250.0])
)
```

Now let's do our grid search. We show the timing when we use a multithreaded grid search vs. a serial grid search. 

```julia 
grid = RegularGrid(get_lower_bounds(likprob_2), get_upper_bounds(likprob_2), 50)
@time gs, lik_vals = grid_search(likprob_2, grid; save_vals = Val(true), parallel=Val(true))
1529.393520 seconds (91.55 M allocations: 606.223 GiB, 2.10% gc time)
```

```julia
@time _gs, _lik_vals = grid_search(likprob_2, grid; save_vals = Val(true), parallel=Val(false))
3454.357503 seconds (86.48 M allocations: 605.468 GiB, 0.14% gc time)
```

Here are the results from the grid search.

```julia
LikelihoodSolution. retcode: Success
Maximum likelihood: -24.399451875029165
Maximum likelihood estimates: 2-element Vector{Float64}
     k: 7.408163265306122
     u₀: 51.0204081632653
```

Let us now visualise the likelihood function.

```julia
fig = Figure(fontsize=38)
k_grid = get_range(grid, 1)
u₀_grid = get_range(grid, 2)
ax = Axis(fig[1, 1],
    xlabel=L"k", ylabel=L"u_0",
    xticks=0:3:15,
    yticks=0:50:250)
co = heatmap!(ax, k_grid, u₀_grid, lik_vals, colormap=Reverse(:matter))
contour!(ax, k_grid, u₀_grid, lik_vals, levels=40, color=:black, linewidth=1 / 4)
scatter!(ax, [k[1]], [u₀], color=:white, markersize=14)
scatter!(ax, [gs[:k]], [gs[:u₀]], color=:blue, markersize=14)
clb = Colorbar(fig[1, 2], co, label=L"\ell(k, u_0)", vertical=true)
```

![Likelihood function for the PDE](https://github.com/DanielVandH/ProfileLikelihood.jl/blob/main/test/figures/heat_pde_contour_example.png?raw=true)

The true parameter values are shown at the white marker, while the results from the grid search are shown in blue, and these two markers are reasonably close. We see that the likelihood function is quite flat around these values, so this might be an indicator of further identifiability issues. Let us now use the grid search results to update our initial guess and compute the MLEs, and then we profile.

```julia
likprob_2 = update_initial_estimate(likprob_2, gs)
mle_sol = mle(likprob_2, NLopt.LN_BOBYQA; ftol_abs=1e-8, ftol_rel=1e-8, xtol_abs=1e-8, xtol_rel=1e-8)
LikelihoodSolution. retcode: Failure
Maximum likelihood: 11.016184577792082
Maximum likelihood estimates: 2-element Vector{Float64}
     k: 9.40527352240195
     u₀: 49.741093700294336
```

```julia
@time prof = profile(likprob_2, mle_sol; ftol_abs=1e-4, ftol_rel=1e-4, xtol_abs=1e-4, xtol_rel=1e-4, parallel=true)
612.723061 seconds (25.45 M allocations: 155.874 GiB, 0.41% gc time)
ProfileLikelihoodSolution. MLE retcode: Failure
Confidence intervals: 
     95.0% CI for k: (8.788003299163778, 10.094019297587579)
     95.0% CI for u₀: (49.44377511158833, 50.03883730450469)
```

The confidence intervals contain the true values. We can now visualise.

```julia
fig = plot_profiles(prof; nrow=1, ncol=3,
    latex_names=[L"k", L"u_0"],
    true_vals=[k[1], u₀],
    fig_kwargs=(fontsize=38, resolution=(1441.9216f0, 470.17322f0)),
    axis_kwargs=(width=600, height=300))
scatter!(fig.content[1], get_parameter_values(prof, :k), get_profile_values(prof, :k), color=:black, markersize=9)
scatter!(fig.content[2], get_parameter_values(prof, :u₀), get_profile_values(prof, :u₀), color=:black, markersize=9)
```

![Second set of profiles](https://github.com/DanielVandH/ProfileLikelihood.jl/blob/main/test/figures/heat_pde_example_2.png?raw=true)

See that we've recovered the parameters in the confidence intervals, and the profiles are smooth -- the identifiability issues are gone. So, it seems like $c$ was the problematic parameter, since our summary statistic does not really give us any information about it. Our idea of using the summary statistic $\mathcal S(t)$ from above would likely ameliorate this issue, since it will give information directly relating to $c$.

## Prediction intervals for the mass

Let us now consider propagating the uncertainty in $k$ and $u_0$ into computing prediction intervals for $\tilde M(t)$ at each $t$. This is done using the `get_prediction_intervals` function introduced in the second example. First, we must define our prediction function.








# Mathematical and Implementation Details

We now give some of the mathematical and implementation details used in this package, namely for computing the profile likelihood function and for computing prediction intervals.

## Computing the profile likelihood function 

Let us start by giving a mathematical description of the method that we use for computing the profile log-likelihood function. Suppose that we have a parameter vector $\boldsymbol\theta$ that we partition as $\boldsymbol \theta = (\psi, \boldsymbol \omega)$ ($\psi$ is a scalar in this description, since we only support univariate profiles currently). We suppose that we have a likelihood function $\mathcal L(\boldsymbol \theta) \equiv \mathcal L(\psi, \boldsymbol \omega)$ so that the normalised profile log-likelihood function for $\psi$ is defined as 

$$ 
\hat\ell_p(\psi) = \sup_{\boldsymbol \omega \in \Omega \mid \psi} \left[\ell(\psi, \boldsymbol\omega) - \ell^*\right],
$$ 

where $\Omega$ is the parameter space for $\boldsymbol \omega$, $\ell(\psi,\boldsymbol\omega) = \mathcal L(\psi, \boldsymbol \omega)$, and $\ell^* = \ell(\hat{\boldsymbol \theta})$, where $\boldsymbol \theta$ are the MLEs for $\boldsymbol \theta$. This definition of $\hat\ell_p(\psi)$ induces a function $\boldsymbol\omega^*(\psi)$ depending on $\psi$ that gives the values of $\boldsymbol \omega$ leading to the supremum above, i.e. 

$$ 
\ell(\psi, \boldsymbol\omega^{\star}(\psi)) = \sup_{\boldsymbol \omega \in \Omega \mid \psi} \left[\ell(\psi, \boldsymbol\omega) - \ell^{\star}\right]. 
$$ 

To compute $\hat\ell_p(\psi)$, then, requires a way to efficiently compute the $\omega^*(\psi)$, and requires knowing where to stop computing. Where we stop computing the profile likelihood is simply when $\hat\ell_p(\psi) < -\chi_{1,1-\alpha}^2/2$, where $\alpha$ is the significance level (e.g. $\alpha=0.05$, in which case $\chi_{1,1-0.05}^2/2 \approx 1.92$). This motivates a iterative algorithm, where we start at the MLE and then step left and right.

We describe how we evaluate the function to the right of the MLE -- the case of going to the left is identical. First, we define $\psi_1 = \hat\psi$, where $\hat\psi$ is the MLE for $\psi$. This defines $\boldsymbol{\omega}\_{1} = \boldsymbol{\omega}^{\star}(\psi\_{1})$, which in this case just gives the MLE $\hat{\boldsymbol\theta} = (\hat\psi, \boldsymbol\omega_1)$ by definition. The value of the normalised profile log-likelihood here is simply $\hat\ell_1 = \hat\ell(\psi_1) = 0$. Then, defining some step size $\Delta\psi$, we define $\psi_2 = \psi_1 + \Delta \psi$, and in general $\psi_{j+1} = \psi_j + \Delta \psi$, we need to estimate $\boldsymbol\omega_2 = \boldsymbol \omega^*(\psi\_2)$. We do this by starting an optimiser at the initial estimate $\boldsymbol\omega_2 = \boldsymbol\omega_1$ and then using this initial estimate to produce a refined value of $\boldsymbol\omega_2$ that we take as its true value. In particular, each $\boldsymbol\omega_j$ comes from starting the optimiser at the previous $\boldsymbol\omega_{j-1}$, and the value for $\hat\ell_j = \hat\ell(\psi_j)$ comes from the value of the likelihood at $(\psi_j, \boldsymbol\omega_j)$. The same holds when going to the left except with $\hat\ell_{j+1} = \psi_j - \Delta\psi$, and then rearranging the indices $j$ when combining the results to the left and to the right.  At each step, we check if $\hat\ell_j < -\chi_{1,1-\alpha}^2/2$ and, if so, we terminate. 

Once we have terminated the algorithm, we need to obtain the confidence intervals. To do this, we fit a spline to the data $(\psi_j, \hat\ell_j)$, and use a bisection algorithm over the two intervals $(\min_j \psi_j, \hat\psi)$ and $(\hat\psi, \max_j\psi_j)$, to find where $\hat\ell_j = -\chi_{1-\alpha}^2/2$. This leads to two solutions $(L, U)$ that we take together to give the confidence interval for $\psi$. 

This is all done for each parameter.

Note that a better method for initialising the optimisation for $\boldsymbol\omega_j$ may be to use e.g. linear interpolation for the previous two values, $\boldsymbol\omega_{j-1}$ and $\boldsymbol\omega_{j-2}$ (with special care for the bounds of the parameters), but this is not done in this package (yet?).

## Computing prediction intervals

Our method for computing prediction intervals follows [Simpson and Maclaren (2022)](https://doi.org/10.1101/2022.12.14.520367), as does our description that follows. This method is nice as it provides a means for sensitivity analysis, enabling the attribution of components of uncertainty in some prediction function $q(\psi, \boldsymbol \omega)$ (with $\psi$ the interest parameter and $\boldsymbol\omega$ the nuisance parameters as above) to individual parameters. The resulting intervals are called *profile-wise intervals*, with the predictions themselves called *parameter-based, profile-wise predictions* or *profile-wise predictions*.

The idea is to take a set of profile likelihoods and the confidence intervals obtained from each, and then pushing those into a prediction function that we then use to obtain prediction intervals, making heavy use of the transformation invariance property of MLEs.

So, let us start with some prediction function $q(\psi, \boldsymbol \omega)$, and recall that the profile likelihood function for $\psi$ induces a function $\boldsymbol\omega^{\star}(\psi)$. The profile-wise likelihood for $q$, given the set of values $(\psi, \boldsymbol\omega^{\star}(\psi))$, is defined by 

$$ 
\hat\ell_p\left(q\left(\psi, \boldsymbol\omega^{\star}(\psi)\right) = q\right) = \sup_{\psi \mid  q\left(\psi, \boldsymbol\omega^{\star}(\psi)\right) = q} \hat\ell_p(\psi). 
$$ 

Note that if $q(\psi, \boldsymbol\omega^*(\psi))$ is injective, there is only one such $\psi$ such that $q\left(\psi, \boldsymbol\omega^*(\psi)\right) = q'$ for any given $q'$ in which case the profile-wise likelihood for $q$ (based on $\psi$) is simply $\hat\ell_p(\psi)$. This definition is intuitive, recalling that the profile likelihood comes from a definition like the above except with the likelihood function on the right, so profile-wise likelihoods come from profile likelihoods.  Using this definition, and using the transformation invariance property of the MLE, confidence sets for $\psi$ directly translate into confidence sets for $q$, in particular to find a $100(1-\alpha)%$ prediction intervals for $q$ we need only evaluate $q$ for $\psi$ inside its confidence interval.

Let us now describe the extra details involved in obtaining these prediction intervals, in particular what we are doing in the `get_prediction_intervals` function. For this, we imagine that $q$ is scalar valued, but the description below can be easily extended to the vector case (just apply the idea to each component -- see the logistic ODE example). We also only explain this for a single parameter $\psi$, but we describe how we use the results for each parameter to obtain a more conservative interval.

The first step is to evaluate the family of curves. If we suppose that the confidence interval for $\psi$ is $(\psi_L, \psi_U)$, we define $\psi_j = \psi_L + (j-1)(\psi_U - \psi_L)(n_\psi - 1)$, $j=1,\ldots,n_\psi$ -- this is a set of $n_\psi$ equally spaced points between the interval limits. For each $\psi_j$ we need to then compute $\boldsymbol\omega^*(\psi_j)$. Rather than re-optimise, we use the data from our profile likelihoods, where we have stored values for $(\psi, \boldsymbol\omega^*(\psi))$ to define a continuous function $\boldsymbol\omega^*(\psi)$ via linear interpolation. Using this linear interpolant we can thus compute $\boldsymbol\omega^*(\psi_j)$ for each gridpoint $\psi_j$. We can therefore compute $\boldsymbol\theta_j = (\psi_j, \boldsymbol\omega^*(\psi_j))$ so that we can evaluate the prediction function at each $\psi_j$, $q_j = q(\boldsymbol\theta_j)$. 

We now have a sample $\{q_1, \ldots, q_{n_\psi}\}$. If we let $q_L = \min_{j=1}^{n_\psi} q_j$ and $q_U = \max_{j=1}^{n_\psi} q_j$, then our prediction interval is $(q_L, q_U)$. To be more specific, this is the profile-wise interval for $q$ given the basis $(\psi, \boldsymbol\omega^*(\psi))$.

We have now described how prediction intervals are obtained gased on a single parameter. Suppose we do this for a collection of parameters $\{\psi^1, \ldots, \psi^d\}$ (e.g. if $\boldsymbol\theta = (D, \lambda, K)$, then we might have computed profiles for $\psi^1=D$, $\psi^2=\lambda$, and $\psi^3=K$), giving $d$ different intervals for each $\psi^i$, say $\{(q_L^i, q_U^i)\}_{i=1}^d$. We can take the union of these intervals to get a more conservative interval for the prediction, giving the new interval $(\min_{i=1}^d q_L^i, \max_{i=1}^d q_U^i)$.