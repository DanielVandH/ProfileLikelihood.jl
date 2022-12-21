# Example IV: Diffusion equation on a square plate

*Warning*: Much of the code in this example takes a very long time, e.g. the MLEs take just under an hour. The total runtime is around six hours on my machine (mostly coming from the mesh for the PDE being very dense). 

The packages we use in this example are:

```julia 
using FiniteVolumeMethod 
using ProfileLikelihood 
using DelaunayTriangulation
using Random 
using LinearSolve 
using OrdinaryDiffEq
using CairoMakie 
using LaTeXStrings
using StaticArraysCore
using Optimization 
using OptimizationNLopt
```

Let us now consider the problem of estimating parameters defining a diffusion equation on a square plate. In particular, consider 

```math
\begin{equation*}
\begin{array}{rcll}
\displaystyle
\frac{\partial u(x, y, t)}{\partial t} &=& \dfrac{1}{k}\boldsymbol{\nabla}^2 u(x, y, t) & (x, y) \in \Omega,t>0, \\
u(x, y, t) &= & 0 & (x, y) \in \partial \Omega,t>0, \\
u(x, y, 0) &= & u_0\mathbb{I}(y \leq c) &(x,y)\in\Omega,
\end{array}
\end{equation*}
```

where $\Omega = [0, 2]^2$. This problem extends the corresponding example given in FiniteVolumeMethod.jl, namely [this example](https://github.com/DanielVandH/FiniteVolumeMethod.jl#diffusion-equation-on-a-square-plate), and so not all the code used in defining this PDE will be explained here; refer to the FiniteVolumeMethod.jl documentation. We will take the true values $k = 9$, $c = 1$, $u_0 = 50$, and let the standard deviation of the noise, $\sigma$, in the data be $0.1$. We are interested in recovering $(k, c, u_0)$; we do not consider estimating $\sigma$ here, estimating it leads to identifiability issues that distract from the main point of our example here, i.e. to just show how to setup a problem.

## Building the FVMProblem 

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

## Defining a summary statistic

Now, one complication with a PDE compared to the scalar ODE cases that we considered previously is that we have data at $(x_i, y_j, t_k)$ for many indices $(i, j, k)$. Rather than defining our objective function in terms of these data points, we will instead use a summary statistic. The summary statistic we use in this example is the average density,

```math
\tilde M(t) = \frac{1}{\mathrm{Area}(\Omega)}\iint_\Omega u(x, y, t)\mathrm{dA}. 
```

We need to be able to compute this integral efficiently and accurately. For this, recall that the finite volume method discretises the domain into triangles. If $\mathcal T$ is this set of triangles, then 

```math 
\tilde M(t) = \frac{1}{\mathrm{Area}(\Omega)}\sum_{T_k \in \mathcal T} \iint_{T_k} u(x, y, t)\mathrm{dA}. 
```

Then, recall that $u$ is represented as a linear function $\alpha_k x + \beta_k y + \gamma_k$ inside the triangle $T_k$, thus 

```math
\tilde M(t) \approx \frac{1}{\mathrm{Area}(\Omega)}\sum_{T_k \in \mathcal T} \left[\alpha_k \iint_{T_k} x\mathrm{dA} + \beta_k \iint_{T_k} y\mathrm{dA} + \gamma_k\iint_{T_k}\mathrm{dA}\right] 
``` 

Now factoring out an $\mathrm{Area}(T_k) = \iint_{T_k}\mathrm{dA}$, 

```math
\tilde M(t) \approx \sum_{T_k \in \mathcal T} \frac{\mathrm{Area}(T_k)}{\mathrm{Area}(\Omega)}\left[\alpha_k \dfrac{\iint_{T_k} x\mathrm{dA}}{\iint_{T_k} \mathrm{dA}} + \beta_k \dfrac{\iint_{T_k} y\mathrm{dA}}{\iint_{T_k} \mathrm{dA}} + \gamma_k\right]. 
```

Notice that the two ratios of integrals shown are simply $\hat x_k$ and $\hat y_k$, where $(\hat x_k, \hat y_k)$ is the centroid of $T_k$. Thus, the term in brackets is $\alpha_k \hat x_k + \beta_k \hat y_k + \gamma_k$, which is the approximation to $u$ at the centroid, $\tilde u(\hat x_k, \hat y_k, t)$. Thus, our approximation to the average density is 

```math
\tilde M(t) \approx \sum_{T_k \in \mathcal T} w_k \tilde u(\hat x_k, \hat y_k, t), \qquad w_k = \frac{\mathrm{Area}(T_k)}{\mathrm{Area}(\Omega)}. 
```

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

## Defining the LikelihoodProblem

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

## Parameter estimation

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

```math
\mathcal S(t) = w\tilde M(t) + (1-w)\tilde A(t),
```

for some $0 \leq w \leq 1$, where $\tilde A(t)$ is the area of the region below the leading edge of the solution, i.e. the area of the non-zero part of the solution. We do not consider this here. What we do consider is fixing $c$, keeping the summary statistic $\tilde M(t)$, and seeing what we can do with only two parameters $k$ and $u_0$.

## Reducing to two parameters and grid searching 

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

## Comparing methods for constructing initial estimates when profiling

In the mathematical details section at the end of this README, it is mentioned that initial values for $\boldsymbol\omega_j$ (the parameters to be optimised while an interest parameter is held fixed) can currently be set in two ways:

- Method 1: Simply starting $\boldsymbol\omega_j$ at $\boldsymbol\omega_{j-1}$. This is the `next_initial_estimate_method = :prev` option in `profile`, and is the default.
- Method 2: Using linear interpolation, we can use the previous two values and set $\boldsymbol\omega_j = [\boldsymbol\omega_{j-2}(\psi_{j-1} - \psi_j) + \boldsymbol\omega_{j-1}(\psi_j - \psi_{j-2})] / (\psi_{j-1} - \psi_{j-2})$ (if $\boldsymbol\omega_j$ then starts outside of the parameter bounds, we fall back to the first method). This is the `next_initial_estimate_method = :interp` option in `profile`.

Is there a big difference in these methods? Let's demonstrate if there is any difference by doing some benchmarking. We will also compare multithreading versus no multithreading.

```julia
bnch_prev_serial = @benchmark profile($likprob_2, $mle_sol; ftol_abs=$1e-4, ftol_rel=$1e-4, xtol_abs=$1e-4, xtol_rel=$1e-4, parallel=$false, next_initial_estimate_method=$:prev)
bnch_interp_serial = @benchmark profile($likprob_2, $mle_sol; ftol_abs=$1e-4, ftol_rel=$1e-4, xtol_abs=$1e-4, xtol_rel=$1e-4, parallel=$false, next_initial_estimate_method=$:interp)
bnch_prev_parallel = @benchmark profile($likprob_2, $mle_sol; ftol_abs=$1e-4, ftol_rel=$1e-4, xtol_abs=$1e-4, xtol_rel=$1e-4, parallel=$true, next_initial_estimate_method=$:prev)
bnch_interp_parallel = @benchmark profile($likprob_2, $mle_sol; ftol_abs=$1e-4, ftol_rel=$1e-4, xtol_abs=$1e-4, xtol_rel=$1e-4, parallel=$true, next_initial_estimate_method=$:interp)
```

Here are the results:

```julia
julia> bnch_prev_serial
BenchmarkTools.Trial: 1 sample with 1 evaluation.
 Single result which took 855.578 s (0.23% GC) to evaluate,
 with a memory estimate of 155.70 GiB, over 24670284 allocations.
```

```julia
julia> bnch_interp_serial
BenchmarkTools.Trial: 1 sample with 1 evaluation.
 Single result which took 757.444 s (0.24% GC) to evaluate,
 with a memory estimate of 144.34 GiB, over 22976564 allocations.
```

```julia
julia> bnch_prev_parallel
BenchmarkTools.Trial: 1 sample with 1 evaluation.
 Single result which took 548.814 s (0.34% GC) to evaluate,
 with a memory estimate of 155.87 GiB, over 25443078 allocations.
```

```julia
julia> bnch_interp_parallel
BenchmarkTools.Trial: 1 sample with 1 evaluation.
 Single result which took 498.408 s (0.36% GC) to evaluate,
 with a memory estimate of 144.52 GiB, over 23809418 allocations.
```

We see that linear interpolation is a significant help to the algorithm, saving 100 seconds when we profile without multithreading, and 50 seconds when we profile with multithreading. In summary, profiling with the `:interp` method was about 12% faster than `:prev` without multithreading, and about 10% faster with multithreading --- interpolation is certainly a big help. For problems where the likelihood function is much faster to compute, these results may be opposite -- it is worth thinking about this for your applications.

## Prediction intervals for the mass

Let us now consider propagating the uncertainty in $k$ and $u_0$ into computing prediction intervals for $\tilde M(t)$ at each $t$. This is done using the `get_prediction_intervals` function introduced in the second example. First, we must define our prediction function.

```julia
@inline function compute_mass_function(θ::AbstractVector{T}, data) where {T}
    k, u₀ = θ
    (; c, prob, t, alg, jac) = data
    prob.flux_parameters[1] = k
    pts = FiniteVolumeMethod.get_points(prob)
    for i in axes(pts, 2)
        pt = get_point(pts, i)
        prob.initial_condition[i] = gety(pt) ≤ c ? u₀ : zero(T)
    end
    sol = solve(prob, alg; saveat=t, parallel=true, jac_prototype=jac)
    shape_cache = zeros(T, 3)
    mass_cache = zeros(T, length(sol.u))
    compute_mass!(mass_cache, shape_cache, sol, prob)
    return mass_cache
end
```

Now let's get the intervals.
```julia
t_many_pts = LinRange(prob.initial_time, prob.final_time, 250)
jac = FiniteVolumeMethod.jacobian_sparsity(prob)
prediction_data = (c=c, prob=prob, t=t_many_pts, alg=alg, jac=jac)
parameter_wise, union_intervals, all_curves, param_range =
    get_prediction_intervals(compute_mass_function, prof, prediction_data; q_type=Vector{Float64})
```

Now we can visualise the curves. We will also show the mass curve from the exact parameter values, as well as from the MLE. 

```julia
exact_soln = compute_mass_function([k[1], u₀], prediction_data)
mle_soln = compute_mass_function(get_mle(mle_sol), prediction_data)
fig = Figure(fontsize=38, resolution=(1360.512f0, 848.64404f0))
alp = join('a':'z')
latex_names = [L"k", L"u_0"]
for i in 1:2
    ax = Axis(fig[1, i], title=L"(%$(alp[i])): Profile-wise PI for %$(latex_names[i])",
        titlealign=:left, width=600, height=300)
    [lines!(ax, t_many_pts, all_curves[i][j], color=:grey) for j in eachindex(param_range[1])]
    lines!(ax, t_many_pts, exact_soln, color=:red)
    lines!(ax, t_many_pts, mle_soln, color=:blue, linestyle=:dash)
    lines!(ax, t_many_pts, getindex.(parameter_wise[i], 1), color=:black)
    lines!(ax, t_many_pts, getindex.(parameter_wise[i], 2), color=:black)
end
ax = Axis(fig[2, 1:2], title=L"(c):$ $ Union of all intervals",
    titlealign=:left, width=1200, height=300)
band!(ax, t_many_pts, getindex.(union_intervals, 1), getindex.(union_intervals, 2), color=:grey)
lines!(ax, t_many_pts, getindex.(union_intervals, 1), color=:black)
lines!(ax, t_many_pts, getindex.(union_intervals, 2), color=:black)
lines!(ax, t_many_pts, exact_soln, color=:red)
lines!(ax, t_many_pts, mle_soln, color=:blue, linestyle=:dash)
```

Let us also add onto these plots the intervals coming from the full likelihood. (The reason to just not do this everytime in applications is because the code below takes a *very* long time to compute - a lifetime compared to the profile-wise intervals above.)

```julia
lb = [8.0, 45.0]
ub = [11.0, 50.0]
N = 1e4
grid = [[lb[i] + (ub[i] - lb[i]) * rand() for _ in 1:N] for i in 1:2]
grid = permutedims(reduce(hcat, grid), (2, 1))
ig = IrregularGrid(lb, ub, grid)
gs, lik_vals = grid_search(likprob_2, ig; parallel=Val(true), save_vals=Val(true))
lik_vals .-= get_maximum(mle_sol) # normalised 
feasible_idx = findall(lik_vals .> ProfileLikelihood.get_chisq_threshold(0.95)) # values in the confidence region 
parameter_evals = grid[:, feasible_idx]
q = [compute_mass_function(θ, prediction_data) for θ in eachcol(parameter_evals)]
q_mat = reduce(hcat, q)
q_lwr = minimum(q_mat; dims=2) |> vec
q_upr = maximum(q_mat; dims=2) |> vec
lines!(ax, t_many_pts, q_lwr, color=:magenta)
lines!(ax, t_many_pts, q_upr, color=:magenta)
```

![Prediction intervals for the mass](https://github.com/DanielVandH/ProfileLikelihood.jl/blob/main/test/figures/heat_pde_example_mass.png?raw=true)

The exact curve has been recovered by our profile likelihood results, and the uncertainty is extremely small. Moreover, the intervals are indeed close to the interval obtained the full profile likelihood as we would hope.