# Example V: Lotka-Volterra ODE, GeneralLazyBufferCache, and computing bivarate profile likelihoods

This example comes from the second case study of [Simpson and Maclaren (2022)](https://doi.org/10.1101/2022.12.14.520367). First, load the packages we'll be using:

```julia
using Random
using Optimization
using OrdinaryDiffEq
using CairoMakie
using LaTeXStrings
using ProfileLikelihood
using OptimizationNLopt
using PreallocationTools
using OptimizationOptimJL
using LoopVectorization
using AbbreviatedStackTraces
```

In this example, we will be considering the Lotka-Volterra ODE, and we will also demonstrate how the `GeneralLazyBufferCache` from PreallocationTools.jl can be used for supporting automatic differentiation for similar problems. In addition, we now also show how bivariate profiles can be computed, along with prediction intervals from a bivariate profile. 
The Lotka-Volterra ODE is given by 

```math 
\begin{align*}
\frac{\mathrm da(t)}{\mathrm dt} &= \alpha a(t) - a(t)b(t), \\
\frac{\mathrm db(t)}{\mathrm dt} &= \beta a(t)b(t)-b(t),
\end{align*}
```

and we suppose that $a(0) = a_0$ and $b(0) = b_0$. For this problem, we are interested in estimating $\boldsymbol = (\alpha,\beta,a_0,b_0)$. We suppose that we have measures of the prey and predicator populations, given respectively by $a(t)$ and $b(t)$, at times $t_i$, $i=1,\ldots,m$. Letting $a_i^
o = a(t_i)$ and $b_i^o = b(t_i)$, $i=1,\ldots,m$, this means that we have the time series $\{(a_i^o, b_i^o)\}_{i=1}^m$. Moreover, just as we did in the logistic ODE example, we suppose that the data $(a_i^o, b_i^o)$ are normally distributed about the solution curve $\boldsymbol z(t; \boldsymbol\theta) = (a(t; \boldsymbol \theta), b(t; \boldsymbol \theta))$. In particular, letting $\boldsymbol z_i(\boldsymbol \theta)$ denote the value of $(a(t_i; \boldsymbol\theta), b(t_i; \boldsymbol \theta))$ at $t=t_i$, we are supposing that 

```math 
(a_i^o, b_i^o) \sim \mathcal N\left(\boldsymbol z_i(\boldsymbol \theta), \sigma^2 \boldsymbol I\right), \quad i=1,2,\ldots,m,
```

and this is what defines our likelihood ($\boldsymbol I$ is the $2$-square identity matrix). We use values $0 \leq t \leq 7$ for estimation, and predict on $0 \leq t \leq 10$.

## Data generation and setting up the problem

As usual, the first step in this example is generating the data.

```julia
using OrdinaryDiffEq, Random, Random 

## Step 1: Generate the data and define the likelihood
α = 0.9
β = 1.1
a₀ = 0.8
b₀ = 0.3
σ = 0.2
t = LinRange(0, 10, 21)
@inline function ode_fnc!(du, u, p, t) where {T}
    α, β = p
    a, b = u
    du[1] = α * a - a * b
    du[2] = β * a * b - b
    return nothing
end
# Initial data is obtained by solving the ODE 
tspan = extrema(t)
p = [α, β]
u₀ = [a₀, b₀]
prob = ODEProblem(ode_fnc!, u₀, tspan, p)
sol = solve(prob, Rosenbrock23(), saveat=t)
Random.seed!(2528)
noise_vec = [σ * randn(2) for _ in eachindex(t)]
uᵒ = sol.u .+ noise_vec
```

We now define the likelihood function. 

```julia
@inline function loglik_fnc2(θ::AbstractVector{T}, data, integrator) where {T}
    α, β, a₀, b₀ = θ
    uᵒ, σ, u₀_cache, αβ_cache, n = data
    u₀ = get_tmp(u₀_cache, θ)
    integrator.p[1] = α
    integrator.p[2] = β
    u₀[1] = a₀
    u₀[2] = b₀
    reinit!(integrator, u₀)
    solve!(integrator)
    ℓ = zero(T)
    for i in 1:n
        âᵒ = integrator.sol.u[i][1]
        b̂ᵒ = integrator.sol.u[i][2]
        aᵒ = uᵒ[i][1]
        bᵒ = uᵒ[i][2]
        ℓ = ℓ - 0.5log(2π * σ^2) - 0.5(âᵒ - aᵒ)^2 / σ^2
        ℓ = ℓ - 0.5log(2π * σ^2) - 0.5(b̂ᵒ - bᵒ)^2 / σ^2
    end
    return ℓ
end
```

Now we define our problem, constraining the parameters so that $0.7 \leq \alpha \leq 1.2$, $0.7 \leq \beta \leq 1.4$, $0.5 \leq a_0 \leq 1.2$, and $0.1 \leq b_0 \leq 0.5$. We want to use forward differentiation for this. Let us start by showing a method that fails:

```julia
using AbbreviatedStackTraces, PreallocationTools, Optimization, OrdinaryDiffEq, OptimizationOptimJL
lb = [0.7, 0.7, 0.5, 0.1]
ub = [1.2, 1.4, 1.2, 0.5]
θ₀ = [0.75, 1.23, 0.76, 0.292]
syms = [:α, :β, :a₀, :b₀]
u₀_cache = DiffCache(zeros(2), 12)
αβ_cache = DiffCache(zeros(2), 12)
n = findlast(t .≤ 7) # Using t ≤ 7 for estimation
prob = LikelihoodProblem(
    loglik_fnc2, θ₀, ode_fnc!, u₀, tspan;
    syms=syms,
    data=(uᵒ, σ, u₀_cache, αβ_cache, n),
    ode_parameters=[1.0, 1.0],
    ode_kwargs=(verbose=false, saveat=t),
    f_kwargs=(adtype=Optimization.AutoForwardDiff(),),
    prob_kwargs=(lb=lb, ub=ub),
    ode_alg=Rosenbrock23()
)
mle(prob, Optim.LBFGS())
```

```julia
julia> sol = mle(prob, Optim.LBFGS())
ERROR: MethodError: no method matching Float64(::ForwardDiff.Dual{ForwardDiff.Tag{Optimization.var"#89#106"{OptimizationFunction{true, Optimization.AutoForwardDiff{nothing}, …}, Tuple{Vector{Vector{Float64}}, Float64, …}}, Float64}, Float64, …})
Closest candidates are:
  (::Type{T})(::Real, ::RoundingMode) where T<:AbstractFloat at rounding.jl:200
  (::Type{T})(::T) where T<:Number at boot.jl:772
  (::Type{T})(::VectorizationBase.Double{T}) where T<:Union{Float16, Float32, Float64, VectorizationBase.Vec{<:Any, <:Union{Float16, Float32, Float64}}, VectorizationBase.VecUnroll{var"#s36", var"#s35", var"#s34", V} where {var"#s36", var"#s35", var"#s34"<:Union{Float16, Float32, Float64}, V<:Union{Bool, Float16, Float32, Float64, Int16, Int32, Int64, Int8, UInt16, UInt32, UInt64, UInt8, SIMDTypes.Bit, VectorizationBase.AbstractSIMD{var"#s35", var"#s34"}}}} at C:\Users\licer\.julia\packages\VectorizationBase\e4FnQ\src\special\double.jl:100
  ...
Stacktrace:
  [1-22] ⋮ internal
       @ Base, Optimization, ForwardDiff, OptimizationOptimJL, NLSolversBase, Optim, Unknown
    [23] #mle#39
       @ c:\Users\licer\.julia\dev\ProfileLikelihood\src\mle.jl:15 [inlined]
    [24] mle(::LikelihoodProblem{4, OptimizationProblem{true, OptimizationFunction{true, Optimization.AutoForwardDiff{nothing}, …}, …}, …}, ::LBFGS{Nothing, LineSearches.InitialStatic{Float64}, 
…})
       @ ProfileLikelihood c:\Users\licer\.julia\dev\ProfileLikelihood\src\mle.jl:13
    [25] ⋮ internal
       @ Unknown
Use `err` to retrieve the full stack trace.
```

The error here comes from trying to use dual numbers when we modify `integrator.p` with the new values for the parameters. We cannot so simply use `DiffCache` to get around this, we would need to somehow assign the appropriate tags when constructing the integrator (see e.g. [here](https://discourse.julialang.org/t/declaring-forwarddiff-tag-directly-with-a-differentialequations-integrator-nested-function/83766) for some more discussion). Note also that this is not just a Optim.jl issue. With NLopt.jl, we do not error, but the optimiser does not go anywhere:

```julia
using OptimizationNLopt 
```

```julia
julia> mle(_prob, NLopt.LD_LBFGS())
LikelihoodSolution. retcode: Failure
Maximum likelihood: -0.0
Maximum likelihood estimates: 4-element Vector{Float64}
     α: 0.75
     β: 1.23
     a₀: 0.76
     b₀: 0.292
```

 To get around this, we can use `GeneralLazyBufferCache` from PreallocationTools.jl. This is a cache that wraps around a function, creating the cache when the function is called (and reused if the same function method is used). This does make things a bit slower (in fact, automatic differentiation is slower than using finite differences or e.g. Nelder-Mead for this problem -- this is just a demonstration) since the dynamic dispatch slows things down. We provide a method for constructing a `LikelihoodProblem` using this cache. The method requires that we first define a function that maps the arguments that would be used for constructing an integrator into a `GeneralLazyBufferCache`. For this problem, this function is as follows:

 ```julia
 lbc = @inline (f, u, p, tspan, ode_alg; kwargs...) -> GeneralLazyBufferCache(
    @inline function ((cache, α, β),) # Needs to be a 1-argument function
        αβ = get_tmp(cache, α)
        αβ[1] = α
        αβ[2] = β
        int = construct_integrator(f, u₀, tspan, αβ, ode_alg; kwargs...)
        return int
    end
)
```

This `cache` argument in the inner function is why we need `αβ_cache` in our likelihood function. The second thing we need is a method that takes `(θ, p)` into the appropriate set of arguments for our `GeneralLazyBufferCache`. For this problem, we want to put `α` and `β` into the cache, and we should also put `αβ_cache` into it. This corresponds to forwarding `θ[1]`, `θ[2]`, and `p[4]` into the function, so we define 

```julia 
lbc_index = @inline (θ, p) -> (p[4], θ[1], θ[2])
```

With these ingredients, we can now define our `LikelihoodProblem`. The constructor is the same as usual for an ODE problem, except with these two functions at the end of the arguments:

```julia 
prob = LikelihoodProblem(
    loglik_fnc2, θ₀, ode_fnc!, u₀, tspan, lbc, lbc_index;
    syms=syms,
    data=(uᵒ, σ, u₀_cache, αβ_cache, n),
    ode_parameters=[1.0, 1.0],
    ode_kwargs=(verbose=false, saveat=t),
    f_kwargs=(adtype=Optimization.AutoForwardDiff(),),
    prob_kwargs=(lb=lb, ub=ub),
    ode_alg=Rosenbrock23()
)
```
```julia
LikelihoodProblem. In-place: true
θ₀: 4-element Vector{Float64}
     α: 0.75
     β: 1.23
     a₀: 0.76
     b₀: 0.292
```

## Parameter estimation

Let us now proceed as usual, computing the MLEs and obtaining the profiles. 

```julia
julia> @time sol = mle(prob, NLopt.LD_LBFGS())
  0.157597 seconds (1.34 M allocations: 57.224 MiB)
LikelihoodSolution. retcode: Failure
Maximum likelihood: 5.0672221211843596
Maximum likelihood estimates: 4-element Vector{Float64}
     α: 0.9732121347334136
     β: 1.0887773087403383
     a₀: 0.7775811746213865
     b₀: 0.34360331260182864
```

```julia
julia> @time prof = profile(prob, sol; parallel=true)
 30.587001 seconds (413.50 M allocations: 17.053 GiB, 13.68% gc time)
ProfileLikelihoodSolution. MLE retcode: Failure
Confidence intervals: 
     95.0% CI for α: (0.8559769794708246, 1.08492137675284)
     95.0% CI for β: (0.9878542591871937, 1.2123035437369885)
     95.0% CI for a₀: (0.6582428835810638, 0.9116011408658178)
     95.0% CI for b₀: (0.24674957441638262, 0.45056076118992644)
```

Now plotting the profiles:

```julia 
fig = plot_profiles(prof;
    latex_names=[L"\alpha", L"\beta", L"a_0", L"b_0"],
    show_mles=true,
    shade_ci=true,
    nrow=2,
    ncol=2,
    true_vals=[α, β, a₀, b₀])
```

![Lotka profiles](https://github.com/DanielVandH/ProfileLikelihood.jl/blob/main/test/figures/lokta_example_profiles.png?raw=true)

## Bivariate profiles 

In all the examples thus far, we have only considered univariate profiles. We also provide a method for computing bivariate profiles through the `bivariate_profile` function. In this function instead of providing a set of integers for the parameters to profile, we provide tuples of integers (or symbols). Let's compute the bivariate profiles for all pairs. In the code below, `resolution=25` means we define 25 layers between the MLE and the bounds for each parameter (see the implementation details section in the sidebar for a definition of a layer). Setting `outer_layers=10` means that we go out 10 layers even after finding the complete confidence region.

```julia 
param_pairs = ((:α, :β), (:α, :a₀), (:α, :b₀),
    (:β, :a₀), (:β, :b₀),
    (:a₀, :b₀)) # Same as param_pairs = ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))
@time prof_2 = bivariate_profile(prob, sol, param_pairs; parallel=true, resolution=25, outer_layers=10) 
# Multithreading highly recommended for bivariate profiles - even a resolution of 25 is an upper bound of 2,601 optimisation problems for each pair (in general, this number is 4N(N+1) + 1 for a resolution of N).
```

```julia
303.152134 seconds (6.19 G allocations: 248.989 GiB, 19.71% gc time)
BivariateProfileLikelihoodSolution. MLE retcode: Failure
Profile info: 
     (β, b₀): 25 layers. Bbox for 95.0% CR: [0.9651436140629203, 1.247290653464279] × [0.22398105279668468, 0.47820878016805074]
     (α, β): 25 layers. Bbox for 95.0% CR: [0.8255352964428667, 1.1123497609332378] × [0.9652918919016166, 1.247309583537848]
     (α, a₀): 25 layers. Bbox for 95.0% CR: [0.8257283233039527, 1.1124082018183483] × [0.6310218313830878, 0.9474704551651254]
     (a₀, b₀): 25 layers. Bbox for 95.0% CR: [0.6309957786415554, 0.9473859257363246] × [0.22405539515039705, 0.47825405211860755]
     (α, b₀): 25 layers. Bbox for 95.0% CR: [0.82594429486772, 1.1123826052893557] × [0.22424659978529984, 0.47806214015711407]
     (β, a₀): 23 layers. Bbox for 95.0% CR: [0.965430158032343, 1.2470957980475892] × [0.6310333834567258, 0.9478656372614495]
```

To plot these profiles, we can use `plot_profiles`. These plots usually take a bit more work than the univariate case. Let's first show a poor plot. We specify `xlims` and `ylims` to match [Simpson and Maclaren (2022)](https://doi.org/10.1101/2022.12.14.520367).

```julia
fig_2 = plot_profiles(prof_2, param_pairs; # param_pairs not needed, but this ensures we get the correct order
    latex_names=[L"\alpha", L"\beta", L"a_0", L"b_0"],
    show_mles=true,
    nrow=3,
    ncol=2,
    true_vals=[α, β, a₀, b₀],
    xlim_tuples=[(0.5, 1.5), (0.5, 1.5), (0.5, 1.5), (0.7, 1.3), (0.7, 1.3), (0.5, 1.1)],
    ylim_tuples=[(0.5, 1.5), (0.5, 1.05), (0.1, 0.5), (0.5, 1.05), (0.1, 0.5), (0.1, 0.5)],
    fig_kwargs=(fontsize=24,))
```

![Poor Lotka bivariate profiles](https://github.com/DanielVandH/ProfileLikelihood.jl/blob/main/test/figures/lokta_example_bivariate_profiles_low_quality.png?raw=true)

In these plots, the red boundaries mark the confidence region's boundary, the red dot shows the MLE, and the black dots are the true values. There are wo issues with these plots:

1. The plots are quite pixelated due to the low resolution.
2. The plots don't fill out the entire axis.

These two issues can be resolved using the interpolant defined from the original data. Setting `interpolant = true` resolves these two problems. (If we also had a poor quality confidence region, you could also set `smooth_confidence_boundary = true`.)

```julia
fig_3 = plot_profiles(prof_2, param_pairs;
    latex_names=[L"\alpha", L"\beta", L"a_0", L"b_0"],
    show_mles=true,
    nrow=3,
    ncol=2,
    true_vals=[α, β, a₀, b₀],
    interpolation=true,
    xlim_tuples=[(0.5, 1.5), (0.5, 1.5), (0.5, 1.5), (0.7, 1.3), (0.7, 1.3), (0.5, 1.1)],
    ylim_tuples=[(0.5, 1.5), (0.5, 1.05), (0.1, 0.5), (0.5, 1.05), (0.1, 0.5), (0.1, 0.5)],
    fig_kwargs=(fontsize=24,))
```

![Smooth Lotka bivariate profiles](https://github.com/DanielVandH/ProfileLikelihood.jl/blob/main/test/figures/lokta_example_bivariate_profiles_smoothed_quality.png?raw=true)

## Prediction intervals

Let's now proceed with finding prediction intervals. We first find the prediction intervals using our univariate results. We use the in-place version of a prediction function:

```julia
function prediction_function!(q, θ::AbstractVector{T}, data) where {T}
    α, β, a₀, b₀ = θ
    t, a_idx, b_idx = data
    prob = ODEProblem(ODEFunction(ode_fnc!, syms=(:a, :b)), [a₀, b₀], extrema(t), (α, β))
    sol = solve(prob, Rosenbrock23(), saveat=t)
    q[a_idx] .= sol[:a]
    q[b_idx] .= sol[:b]
    return nothing
end
t_many_pts = LinRange(extrema(t)..., 1000)
a_idx = 1:1000
b_idx = 1001:2000
pred_data = (t_many_pts, a_idx, b_idx)
q_prototype = zeros(2000)
individual_intervals, union_intervals, q_vals, param_ranges =
    get_prediction_intervals(prediction_function!, prof, pred_data; parallel=true,
        q_prototype)
```

Now we plot these results, plotting the individual intervals as well as the union intervals. As in Example II, we also look at the intervals from the full likelihood.

```julia 
# Evaluate the exact and MLE solutions
exact_soln = zeros(2000)
mle_soln = zeros(2000)
prediction_function!(exact_soln, [α, β, a₀, b₀], pred_data)
prediction_function!(mle_soln, get_mle(sol), pred_data)

# Plot the parameter-wise intervals 
fig = Figure(fontsize=38, resolution=(2935.488f0, 1392.64404f0))
alp = [['a', 'b', 'e', 'f'], ['c', 'd', 'g', 'h']]
latex_names = [L"\alpha", L"\beta", L"a_0", L"b_0"]
for (k, idx) in enumerate((a_idx, b_idx))
    for i in 1:4
        ax = Axis(fig[i < 3 ? 1 : 2, mod1(i, 2)+(k==2)*2], title=L"(%$(alp[k][i])): Profile-wise PI for %$(latex_names[i])",
            titlealign=:left, width=600, height=300, xlabel=L"t", ylabel=k == 1 ? L"a(t)" : L"b(t)")
        vlines!(ax, [7.0], color=:purple, linestyle=:dash, linewidth=2)
        lines!(ax, t_many_pts, exact_soln[idx], color=:red, linewidth=3)
        lines!(ax, t_many_pts, mle_soln[idx], color=:blue, linestyle=:dash, linewidth=3)
        lines!(ax, t_many_pts, getindex.(individual_intervals[i], 1)[idx], color=:black, linewidth=3)
        lines!(ax, t_many_pts, getindex.(individual_intervals[i], 2)[idx], color=:black, linewidth=3)
        band!(ax, t_many_pts, getindex.(individual_intervals[i], 1)[idx], getindex.(individual_intervals[i], 2)[idx], color=(:grey, 0.35))
    end
end

# Plot the union intervals
a_ax = Axis(fig[3, 1:2], title=L"(i):$ $ Union of all intervals",
    titlealign=:left, width=1200, height=300, xlabel=L"t", ylabel=L"a(t)")
b_ax = Axis(fig[3, 3:4], title=L"(j):$ $ Union of all intervals",
    titlealign=:left, width=1200, height=300, xlabel=L"t", ylabel=L"b(t)")
_ax = (a_ax, b_ax)
for (k, idx) in enumerate((a_idx, b_idx))
    band!(_ax[k], t_many_pts, getindex.(union_intervals, 1)[idx], getindex.(union_intervals, 2)[idx], color=(:grey, 0.35))
    lines!(_ax[k], t_many_pts, getindex.(union_intervals, 1)[idx], color=:black, linewidth=3)
    lines!(_ax[k], t_many_pts, getindex.(union_intervals, 2)[idx], color=:black, linewidth=3)
    lines!(_ax[k], t_many_pts, exact_soln[idx], color=:red, linewidth=3)
    lines!(_ax[k], t_many_pts, mle_soln[idx], color=:blue, linestyle=:dash, linewidth=3)
    vlines!(_ax[k], [7.0], color=:purple, linestyle=:dash, linewidth=2)
end

# Compare to the results obtained from the full likelihood
lb = get_lower_bounds(prob)
ub = get_upper_bounds(prob)
N = 1e5
grid = [[lb[i] + (ub[i] - lb[i]) * rand() for _ in 1:N] for i in 1:4]
grid = permutedims(reduce(hcat, grid), (2, 1))
ig = IrregularGrid(lb, ub, grid)
gs, lik_vals = grid_search(prob, ig; parallel=Val(true), save_vals=Val(true))
lik_vals .-= get_maximum(sol) # normalised 
feasible_idx = findall(lik_vals .> ProfileLikelihood.get_chisq_threshold(0.95)) # values in the confidence region 
parameter_evals = grid[:, feasible_idx]
full_q_vals = zeros(2000, size(parameter_evals, 2))
@views [prediction_function!(full_q_vals[:, j], parameter_evals[:, j], pred_data) for j in axes(parameter_evals, 2)]
q_lwr = minimum(full_q_vals; dims=2) |> vec
q_upr = maximum(full_q_vals; dims=2) |> vec
for (k, idx) in enumerate((a_idx, b_idx))
    lines!(_ax[k], t_many_pts, q_lwr[idx], color=:magenta, linewidth=3)
    lines!(_ax[k], t_many_pts, q_upr[idx], color=:magenta, linewidth=3)
end
```

![Lotka univariate predictions](https://github.com/DanielVandH/ProfileLikelihood.jl/blob/main/test/figures/lokta_example_univariate_predictions.png?raw=true)

We see that the uncertainty around our predictions increases significantly for $t > 7$, as expected since we only use data in $0 \leq t \leq 7$ for estmiating the parameters. Moreover, the union intervals are good approximations to the intervals from the full likelihood.

Now let us extend these results, instead computing prediction intervals from our bivariate profiles. The exact same function can be used for this.

```julia
# Bivariate prediction intervals 
individual_intervals, union_intervals, q_vals, param_ranges =
    get_prediction_intervals(prediction_function!, prof_2, pred_data; parallel=true,
        q_prototype)

# Plot the intervals 
fig = Figure(fontsize=38, resolution=(2935.488f0, 1854.64404f0))
integer_param_pairs = ProfileLikelihood.convert_symbol_tuples(param_pairs, prof_2) # converts to the integer representation
alp = [['a', 'b', 'e', 'f', 'i', 'j'], ['c', 'd', 'g', 'h', 'k', 'l']]
for (k, idx) in enumerate((a_idx, b_idx))
    for (i, (u, v)) in enumerate(integer_param_pairs)
        ax = Axis(fig[i < 3 ? 1 : (i < 5 ? 2 : 3), mod1(i, 2)+(k==2)*2], title=L"(%$(alp[k][i])): Profile-wise PI for (%$(latex_names[u]), %$(latex_names[v]))",
            titlealign=:left, width=600, height=300, xlabel=L"t", ylabel=k == 1 ? L"a(t)" : L"b(t)")
        vlines!(ax, [7.0], color=:purple, linestyle=:dash, linewidth=2)
        lines!(ax, t_many_pts, exact_soln[idx], color=:red, linewidth=3)
        lines!(ax, t_many_pts, mle_soln[idx], color=:blue, linestyle=:dash, linewidth=3)
        lines!(ax, t_many_pts, getindex.(individual_intervals[(u, v)], 1)[idx], color=:black, linewidth=3)
        lines!(ax, t_many_pts, getindex.(individual_intervals[(u, v)], 2)[idx], color=:black, linewidth=3)
        band!(ax, t_many_pts, getindex.(individual_intervals[(u, v)], 1)[idx], getindex.(individual_intervals[(u, v)], 2)[idx], color=(:grey, 0.35))
    end
end
a_ax = Axis(fig[4, 1:2], title=L"(i):$ $ Union of all intervals",
    titlealign=:left, width=1200, height=300, xlabel=L"t", ylabel=L"a(t)")
b_ax = Axis(fig[4, 3:4], title=L"(j):$ $ Union of all intervals",
    titlealign=:left, width=1200, height=300, xlabel=L"t", ylabel=L"b(t)")
_ax = (a_ax, b_ax)
for (k, idx) in enumerate((a_idx, b_idx))
    band!(_ax[k], t_many_pts, getindex.(union_intervals, 1)[idx], getindex.(union_intervals, 2)[idx], color=(:grey, 0.35))
    lines!(_ax[k], t_many_pts, getindex.(union_intervals, 1)[idx], color=:black, linewidth=3)
    lines!(_ax[k], t_many_pts, getindex.(union_intervals, 2)[idx], color=:black, linewidth=3)
    lines!(_ax[k], t_many_pts, exact_soln[idx], color=:red, linewidth=3)
    lines!(_ax[k], t_many_pts, mle_soln[idx], color=:blue, linestyle=:dash, linewidth=3)
    vlines!(_ax[k], [7.0], color=:purple, linestyle=:dash, linewidth=2)
end
for (k, idx) in enumerate((a_idx, b_idx))
    lines!(_ax[k], t_many_pts, q_lwr[idx], color=:magenta, linewidth=3)
    lines!(_ax[k], t_many_pts, q_upr[idx], color=:magenta, linewidth=3)
end
```

![Lotka bivariate predictions](https://github.com/DanielVandH/ProfileLikelihood.jl/blob/main/test/figures/lokta_example_bivariate_predictions.png?raw=true)

## Just the code

Here is all the code used for obtaining the results in this example, should you want a version that you can directly copy and paste.

```julia 
## Step 1: Generate the data and define the likelihood
using OrdinaryDiffEq, Random
α = 0.9
β = 1.1
a₀ = 0.8
b₀ = 0.3
σ = 0.2
t = LinRange(0, 10, 21)
@inline function ode_fnc!(du, u, p, t) where {T}
    α, β = p
    a, b = u
    du[1] = α * a - a * b
    du[2] = β * a * b - b
    return nothing
end
# Initial data is obtained by solving the ODE 
tspan = extrema(t)
p = [α, β]
u₀ = [a₀, b₀]
prob = ODEProblem(ode_fnc!, u₀, tspan, p)
sol = solve(prob, Rosenbrock23(), saveat=t)
Random.seed!(252800)
noise_vec = [σ * randn(2) for _ in eachindex(t)]
uᵒ = sol.u .+ noise_vec
@inline function loglik_fnc2(θ::AbstractVector{T}, data, integrator) where {T}
    α, β, a₀, b₀ = θ
    uᵒ, σ, u₀_cache, αβ_cache, n = data
    u₀ = get_tmp(u₀_cache, θ)
    integrator.p[1] = α
    integrator.p[2] = β
    u₀[1] = a₀
    u₀[2] = b₀
    reinit!(integrator, u₀)
    solve!(integrator)
    ℓ = zero(T)
    for i in 1:n
        âᵒ = integrator.sol.u[i][1]
        b̂ᵒ = integrator.sol.u[i][2]
        aᵒ = uᵒ[i][1]
        bᵒ = uᵒ[i][2]
        ℓ = ℓ - 0.5log(2π * σ^2) - 0.5(âᵒ - aᵒ)^2 / σ^2
        ℓ = ℓ - 0.5log(2π * σ^2) - 0.5(b̂ᵒ - bᵒ)^2 / σ^2
    end
    return ℓ
end

## Step 2: Define the problem 
using PreallocationTools, Optimization
lb = [0.7, 0.7, 0.5, 0.1]
ub = [1.2, 1.4, 1.2, 0.5]
θ₀ = [0.75, 1.23, 0.76, 0.292]
syms = [:α, :β, :a₀, :b₀]
u₀_cache = DiffCache(zeros(2), 12)
αβ_cache = DiffCache(zeros(2), 12)
n = findlast(t .≤ 7) # Using t ≤ 7 for estimation
lbc = @inline (f, u, p, tspan, ode_alg; kwargs...) -> GeneralLazyBufferCache(
    @inline function ((cache, α, β),) # Needs to be a 1-argument function
        αβ = get_tmp(cache, α)
        αβ[1] = α
        αβ[2] = β
        int = construct_integrator(f, u₀, tspan, αβ, ode_alg; kwargs...)
        return int
    end
)
lbc_index = @inline (θ, p) -> (p[4], θ[1], θ[2])
prob = LikelihoodProblem(
    loglik_fnc2, θ₀, ode_fnc!, u₀, tspan, lbc, lbc_index;
    syms=syms,
    data=(uᵒ, σ, u₀_cache, αβ_cache, n),
    ode_parameters=[1.0, 1.0],
    ode_kwargs=(verbose=false, saveat=t),
    f_kwargs=(adtype=Optimization.AutoForwardDiff(),),
    prob_kwargs=(lb=lb, ub=ub),
    ode_alg=Rosenbrock23()
)

## Step 3: Compute the MLE 
using OptimizationNLopt 
sol = mle(prob, NLopt.LD_LBFGS())

## Step 4: Profile
prof = profile(prob, sol; parallel=true)

## Step 5: Visualise 
using CairoMakie, LaTeXStrings 
fig = plot_profiles(prof;
    latex_names=[L"\alpha", L"\beta", L"a_0", L"b_0"],
    show_mles=true,
    shade_ci=true,
    nrow=2,
    ncol=2,
    true_vals=[α, β, a₀, b₀])

## Step 6: Obtain the bivariate profiles 
param_pairs = ((:α, :β), (:α, :a₀), (:α, :b₀),
    (:β, :a₀), (:β, :b₀),
    (:a₀, :b₀))
prof_2 = bivariate_profile(prob, sol, param_pairs; parallel=true, resolution=25, outer_layers=10)

## Step 7: Visualise 
using CairoMakie, LaTeXStrings
fig_3 = plot_profiles(prof_2, param_pairs;
    latex_names=[L"\alpha", L"\beta", L"a_0", L"b_0"],
    show_mles=true,
    nrow=3,
    ncol=2,
    true_vals=[α, β, a₀, b₀],
    interpolation=true,
    xlim_tuples=[(0.5, 1.5), (0.5, 1.5), (0.5, 1.5), (0.7, 1.3), (0.7, 1.3), (0.5, 1.1)],
    ylim_tuples=[(0.5, 1.5), (0.5, 1.05), (0.1, 0.5), (0.5, 1.05), (0.1, 0.5), (0.1, 0.5)],
    fig_kwargs=(fontsize=24,))

## Step 8: Get the prediction intervals from the univariate profiles 
function prediction_function!(q, θ::AbstractVector{T}, data) where {T}
    α, β, a₀, b₀ = θ
    t, a_idx, b_idx = data
    prob = ODEProblem(ODEFunction(ode_fnc!, syms=(:a, :b)), [a₀, b₀], extrema(t), (α, β))
    sol = solve(prob, Rosenbrock23(), saveat=t)
    q[a_idx] .= sol[:a]
    q[b_idx] .= sol[:b]
    return nothing
end
t_many_pts = LinRange(extrema(t)..., 1000)
a_idx = 1:1000
b_idx = 1001:2000
pred_data = (t_many_pts, a_idx, b_idx)
q_prototype = zeros(2000)
individual_intervals, union_intervals, q_vals, param_ranges =
    get_prediction_intervals(prediction_function!, prof, pred_data; parallel=true,
        q_prototype)

## Step 9: Visualise
# Evaluate the exact and MLE solutions
exact_soln = zeros(2000)
mle_soln = zeros(2000)
prediction_function!(exact_soln, [α, β, a₀, b₀], pred_data)
prediction_function!(mle_soln, get_mle(sol), pred_data)

# Plot the parameter-wise intervals 
fig = Figure(fontsize=38, resolution=(2935.488f0, 1392.64404f0))
alp = [['a', 'b', 'e', 'f'], ['c', 'd', 'g', 'h']]
latex_names = [L"\alpha", L"\beta", L"a_0", L"b_0"]
for (k, idx) in enumerate((a_idx, b_idx))
    for i in 1:4
        ax = Axis(fig[i < 3 ? 1 : 2, mod1(i, 2)+(k==2)*2], title=L"(%$(alp[k][i])): Profile-wise PI for %$(latex_names[i])",
            titlealign=:left, width=600, height=300, xlabel=L"t", ylabel=k == 1 ? L"a(t)" : L"b(t)")
        vlines!(ax, [7.0], color=:purple, linestyle=:dash, linewidth=2)
        lines!(ax, t_many_pts, exact_soln[idx], color=:red, linewidth=3)
        lines!(ax, t_many_pts, mle_soln[idx], color=:blue, linestyle=:dash, linewidth=3)
        lines!(ax, t_many_pts, getindex.(individual_intervals[i], 1)[idx], color=:black, linewidth=3)
        lines!(ax, t_many_pts, getindex.(individual_intervals[i], 2)[idx], color=:black, linewidth=3)
        band!(ax, t_many_pts, getindex.(individual_intervals[i], 1)[idx], getindex.(individual_intervals[i], 2)[idx], color=(:grey, 0.35))
    end
end

# Plot the union intervals
a_ax = Axis(fig[3, 1:2], title=L"(i):$ $ Union of all intervals",
    titlealign=:left, width=1200, height=300, xlabel=L"t", ylabel=L"a(t)")
b_ax = Axis(fig[3, 3:4], title=L"(j):$ $ Union of all intervals",
    titlealign=:left, width=1200, height=300, xlabel=L"t", ylabel=L"b(t)")
_ax = (a_ax, b_ax)
for (k, idx) in enumerate((a_idx, b_idx))
    band!(_ax[k], t_many_pts, getindex.(union_intervals, 1)[idx], getindex.(union_intervals, 2)[idx], color=(:grey, 0.35))
    lines!(_ax[k], t_many_pts, getindex.(union_intervals, 1)[idx], color=:black, linewidth=3)
    lines!(_ax[k], t_many_pts, getindex.(union_intervals, 2)[idx], color=:black, linewidth=3)
    lines!(_ax[k], t_many_pts, exact_soln[idx], color=:red, linewidth=3)
    lines!(_ax[k], t_many_pts, mle_soln[idx], color=:blue, linestyle=:dash, linewidth=3)
    vlines!(_ax[k], [7.0], color=:purple, linestyle=:dash, linewidth=2)
end

# Compare to the results obtained from the full likelihood
lb = get_lower_bounds(prob)
ub = get_upper_bounds(prob)
N = 1e5
grid = [[lb[i] + (ub[i] - lb[i]) * rand() for _ in 1:N] for i in 1:4]
grid = permutedims(reduce(hcat, grid), (2, 1))
ig = IrregularGrid(lb, ub, grid)
gs, lik_vals = grid_search(prob, ig; parallel=Val(true), save_vals=Val(true))
lik_vals .-= get_maximum(sol) # normalised 
feasible_idx = findall(lik_vals .> ProfileLikelihood.get_chisq_threshold(0.95)) # values in the confidence region 
parameter_evals = grid[:, feasible_idx]
full_q_vals = zeros(2000, size(parameter_evals, 2))
@views [prediction_function!(full_q_vals[:, j], parameter_evals[:, j], pred_data) for j in axes(parameter_evals, 2)]
q_lwr = minimum(full_q_vals; dims=2) |> vec
q_upr = maximum(full_q_vals; dims=2) |> vec
for (k, idx) in enumerate((a_idx, b_idx))
    lines!(_ax[k], t_many_pts, q_lwr[idx], color=:magenta, linewidth=3)
    lines!(_ax[k], t_many_pts, q_upr[idx], color=:magenta, linewidth=3)
end

## Step 10: Get the prediction intervals from the bivariate profiles 
individual_intervals, union_intervals, q_vals, param_ranges =
    get_prediction_intervals(prediction_function!, prof_2, pred_data; parallel=true,
        q_prototype)

## Step 11: Visualise the prediction intervals 
fig = Figure(fontsize=38, resolution=(2935.488f0, 1854.64404f0))
integer_param_pairs = ProfileLikelihood.convert_symbol_tuples(param_pairs, prof_2) # converts to the integer representation
alp = [['a', 'b', 'e', 'f', 'i', 'j'], ['c', 'd', 'g', 'h', 'k', 'l']]
for (k, idx) in enumerate((a_idx, b_idx))
    for (i, (u, v)) in enumerate(integer_param_pairs)
        ax = Axis(fig[i < 3 ? 1 : (i < 5 ? 2 : 3), mod1(i, 2)+(k==2)*2], title=L"(%$(alp[k][i])): Profile-wise PI for (%$(latex_names[u]), %$(latex_names[v]))",
            titlealign=:left, width=600, height=300, xlabel=L"t", ylabel=k == 1 ? L"a(t)" : L"b(t)")
        vlines!(ax, [7.0], color=:purple, linestyle=:dash, linewidth=2)
        lines!(ax, t_many_pts, exact_soln[idx], color=:red, linewidth=3)
        lines!(ax, t_many_pts, mle_soln[idx], color=:blue, linestyle=:dash, linewidth=3)
        lines!(ax, t_many_pts, getindex.(individual_intervals[(u, v)], 1)[idx], color=:black, linewidth=3)
        lines!(ax, t_many_pts, getindex.(individual_intervals[(u, v)], 2)[idx], color=:black, linewidth=3)
        band!(ax, t_many_pts, getindex.(individual_intervals[(u, v)], 1)[idx], getindex.(individual_intervals[(u, v)], 2)[idx], color=(:grey, 0.35))
    end
end
a_ax = Axis(fig[4, 1:2], title=L"(i):$ $ Union of all intervals",
    titlealign=:left, width=1200, height=300, xlabel=L"t", ylabel=L"a(t)")
b_ax = Axis(fig[4, 3:4], title=L"(j):$ $ Union of all intervals",
    titlealign=:left, width=1200, height=300, xlabel=L"t", ylabel=L"b(t)")
_ax = (a_ax, b_ax)
for (k, idx) in enumerate((a_idx, b_idx))
    band!(_ax[k], t_many_pts, getindex.(union_intervals, 1)[idx], getindex.(union_intervals, 2)[idx], color=(:grey, 0.35))
    lines!(_ax[k], t_many_pts, getindex.(union_intervals, 1)[idx], color=:black, linewidth=3)
    lines!(_ax[k], t_many_pts, getindex.(union_intervals, 2)[idx], color=:black, linewidth=3)
    lines!(_ax[k], t_many_pts, exact_soln[idx], color=:red, linewidth=3)
    lines!(_ax[k], t_many_pts, mle_soln[idx], color=:blue, linestyle=:dash, linewidth=3)
    vlines!(_ax[k], [7.0], color=:purple, linestyle=:dash, linewidth=2)
end
for (k, idx) in enumerate((a_idx, b_idx))
    lines!(_ax[k], t_many_pts, q_lwr[idx], color=:magenta, linewidth=3)
    lines!(_ax[k], t_many_pts, q_upr[idx], color=:magenta, linewidth=3)
end
```