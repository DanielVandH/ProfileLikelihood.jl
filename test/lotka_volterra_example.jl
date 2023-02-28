using Random
using ..ProfileLikelihood
using Optimization
using OrdinaryDiffEq
using CairoMakie
using LaTeXStrings
using OptimizationOptimJL
using OptimizationNLopt
using PreallocationTools
using LoopVectorization
using DelaunayTriangulation
const SAVE_FIGURE = true

######################################################
## Example V: Lotka-Volterra ODE
######################################################
## Step 1: Generate the data and define the likelihood
α = 0.9
β = 1.1
a₀ = 0.8
b₀ = 0.3
σ = 0.2
t = LinRange(0, 10, 21)
@inline function ode_fnc!(du, u, p, t)
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
lb = [0.7, 0.7, 0.5, 0.1]
ub = [1.2, 1.4, 1.2, 0.5]
θ₀ = [0.75, 1.23, 0.76, 0.292]
syms = [:α, :β, :a₀, :b₀]
u₀_cache = DiffCache(zeros(2), 12)
αβ_cache = DiffCache(zeros(2), 12)
n = findlast(t .≤ 7) # Using t ≤ 7 for estimation
lbc = @inline (f, u, p, tspan, ode_alg; kwargs...) -> GeneralLazyBufferCache(
    @inline function ((cache, u₀_cache, α, β, a₀, b₀),) # Needs to be a 1-argument function
        αβ = get_tmp(cache, α)
        αβ[1] = α
        αβ[2] = β
        u₀ = get_tmp(u₀_cache, a₀)
        u₀[1] = a₀
        u₀[2] = b₀
        int = construct_integrator(f, u₀, tspan, αβ, ode_alg; kwargs...)
        return int
    end
)
lbc_index = @inline (θ, p) -> (p[4], p[3], θ[1], θ[2], θ[3], θ[4])
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
_prob = LikelihoodProblem(
    loglik_fnc2, θ₀, ode_fnc!, u₀, tspan;
    syms=syms,
    data=(uᵒ, σ, u₀_cache, αβ_cache, n),
    ode_parameters=[1.0, 1.0],
    ode_kwargs=(verbose=false, saveat=t),
    f_kwargs=(adtype=Optimization.AutoFiniteDiff(),),
    prob_kwargs=(lb=lb, ub=ub),
    ode_alg=Rosenbrock23()
)

## Step 3: Compute the MLE 
mle(prob, Optim.LBFGS())
mle(_prob, Optim.LBFGS())
mle(prob, NLopt.LD_LBFGS())
mle(_prob, NLopt.LD_LBFGS())

@time sol = mle(prob, NLopt.LD_LBFGS())
@time _sol = mle(_prob, NLopt.LD_LBFGS())

@test get_mle(sol) ≈ get_mle(_sol) rtol = 1e-2
@test get_maximum(sol) ≈ get_maximum(_sol) rtol = 1e-3

## Step 4: Profile 
profile(prob, sol; parallel=true)
profile(_prob, _sol; parallel=true)

@time prof = profile(prob, sol; parallel=true)
@time _prof = profile(_prob, _sol; parallel=true)

@test α ∈ get_confidence_intervals(prof, :α)
@test β ∈ get_confidence_intervals(prof, :β)
@test a₀ ∈ get_confidence_intervals(prof, :a₀)
@test b₀ ∈ get_confidence_intervals(prof, :b₀)
for s in [:α, :β, :a₀, :b₀]
    @test collect(ProfileLikelihood.get_bounds(get_confidence_intervals(prof, s))) ≈
          collect(ProfileLikelihood.get_bounds(get_confidence_intervals(_prof, s))) rtol = 1e-3
end

## Step 5: Plot 
fig = plot_profiles(prof;
    latex_names=[L"\alpha", L"\beta", L"a_0", L"b_0"],
    show_mles=true,
    shade_ci=true,
    nrow=2,
    ncol=2,
    true_vals=[α, β, a₀, b₀])
SAVE_FIGURE && save("figures/lokta_example_profiles.png", fig)

## Step 6: Obtain the bivariate profiles 
param_pairs = ((:α, :β), (:α, :a₀), (:α, :b₀),
    (:β, :a₀), (:β, :b₀),
    (:a₀, :b₀)) # Same as param_pairs = ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))
integer_n = ntuple(i -> (SciMLBase.sym_to_index(param_pairs[i][1], prob), SciMLBase.sym_to_index(param_pairs[i][2], prob)), 6)
@test integer_n == ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)) == ProfileLikelihood.convert_symbol_tuples(param_pairs, prob)
@test ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)) == ProfileLikelihood.convert_symbol_tuples(((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)), prob)
@test (1, 2) == ProfileLikelihood.convert_symbol_tuples((1, 2), prob)
@test (1, 2) == ProfileLikelihood.convert_symbol_tuples((:α, :β), prob)
@test (2, 4) == ProfileLikelihood.convert_symbol_tuples((:β, :b₀), prob)
@test (2, 4) == ProfileLikelihood.convert_symbol_tuples((:β, :b₀), prof)
@test integer_n == ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)) == ProfileLikelihood.convert_symbol_tuples(param_pairs, prof)
@test ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)) == ProfileLikelihood.convert_symbol_tuples(((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)), prof)
@test (1, 2) == ProfileLikelihood.convert_symbol_tuples((1, 2), prof)
@test (1, 2) == ProfileLikelihood.convert_symbol_tuples((:α, :β), prof)
@test (2, 4) == ProfileLikelihood.convert_symbol_tuples((:β, :b₀), prof)

@time prof_2 = bivariate_profile(prob, sol, param_pairs; parallel=true, resolution=25, outer_layers=10) # Multithreading highly recommended for bivariate profiles - even a resolution of 25 is an upper bound of 2,601 optimisation problems for each pair (in general, this number is 4N(N+1) + 1 for a resolution of N).
@time _prof_2 = bivariate_profile(_prob, _sol, param_pairs; parallel=true, resolution=25, outer_layers=10)

@test all(collect(ProfileLikelihood.get_bounding_box(prof_2, i, j)) == collect(ProfileLikelihood.get_bounding_box(prof_2, i, j)) for (i, j) in param_pairs)
for (i, j) in param_pairs
    CR_1 = get_confidence_regions(prof_2, i, j)
    CR_2 = get_confidence_regions(_prof_2, i, j)
    A_1 = DelaunayTriangulation.area([[x, y] for (x, y) in CR_1])
    A_2 = DelaunayTriangulation.area([[x, y] for (x, y) in CR_2])
    @test A_1 ≈ A_2 rtol = 1e-2
end

fig_2 = plot_profiles(prof_2, param_pairs; # param_pairs not needed, but this ensures we get the same order as above
    latex_names=[L"\alpha", L"\beta", L"a_0", L"b_0"],
    show_mles=true,
    nrow=3,
    ncol=2,
    true_vals=[α, β, a₀, b₀],
    xlim_tuples=[(0.5, 1.5), (0.5, 1.5), (0.5, 1.5), (0.7, 1.3), (0.7, 1.3), (0.5, 1.1)],
    ylim_tuples=[(0.5, 1.5), (0.5, 1.05), (0.1, 0.5), (0.5, 1.05), (0.1, 0.5), (0.1, 0.5)],
    fig_kwargs=(fontsize=24,))
SAVE_FIGURE && save("figures/lokta_example_bivariate_profiles_low_quality.png", fig_2)

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
SAVE_FIGURE && save("figures/lokta_example_bivariate_profiles_smoothed_quality.png", fig_3)

## Step 7: Get prediction intervals
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
SAVE_FIGURE && save("figures/lokta_example_univariate_predictions.png", fig)

## Bivariate prediction intervals 
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
SAVE_FIGURE && save("figures/lokta_example_bivariate_predictions.png", fig)