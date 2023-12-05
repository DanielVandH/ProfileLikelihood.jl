using ..ProfileLikelihood
using MovingBoundaryProblems1D
using OrdinaryDiffEq
using LinearSolve
using Optimization
using CairoMakie
using OptimizationNLopt
using StableRNGs
using ReferenceTests

## Step 1: Define the problem 
n = 100     # number of mesh points 
κ = [20.0]  # Stefan problem 
α = 1 / 2   # initial condition height 
β = 1.0     # initial endpoint 
D = [0.89]  # diffusivity 
λ = [1.1]   # proliferation rate
K = [3.0]   # carrying capacity density
T = 40.0    # final time
mesh_points = LinRange(0, β, n)
diffusion_function = (u, x, t, D) -> oftype(u, D[1])
reaction_function = (u, x, t, (λ, K)) -> λ[1] * u * (1 - u / K[1])
lhs = Neumann(0.0)
rhs = Dirichlet(0.0)
moving_boundary = Robin(
    (u, t, κ) -> (zero(u), oftype(u, -κ[1])),
    κ
)
ic = x -> x < β ? α : zero(x)
initial_condition = ic.(mesh_points)
prob = MBProblem(
    mesh_points, lhs, rhs, moving_boundary;
    diffusion_function,
    diffusion_parameters=D,
    reaction_function,
    reaction_parameters=(λ, K),
    initial_condition,
    initial_endpoint=β,
    final_time=T
)

## Step 2: Generate the data 
function compute_average(sol::AbstractVector, Δx)
    s = @views sol[begin:(end-1)]
    M = (s[1] + s[end]) / 2
    for i in 2:(length(s)-1)
        M += s[i]
    end
    M *= Δx
    return M
end
function compute_average(sol) # (1/L)∫₀ᴸ u(x, t) dx = ∫₀¹ u(ξ, t) dξ
    prob = sol.prob.p
    mesh_points = prob.geometry.mesh_points
    Δx = mesh_points[2] - mesh_points[1]
    M = map(sol) do _sol
        compute_average(_sol, Δx)
    end
    return M
end

sol = solve(prob, TRBDF2(linsolve=KLUFactorization()), saveat=T / 250)
M = compute_average(sol)
_original_M = deepcopy(M)
rng = StableRNG(123)
σM = 0.3
M .= max.(M .+ σM * randn(rng, length(M)), 0.0)
σL = 4.2
L = max.(sol[end, :] .+ σL * randn(rng, length(sol[end, :])), 0.0)
_original_L = deepcopy(sol[end, :])

fig = Figure(fontsize=41)
ax = Axis(fig[1, 1], width=1200, height=400,
    xlabel=L"t", ylabel=L"M(t)", titlealign=:left)
L1 = lines!(ax, sol.t, M, color=:black, linewidth=6)
L2 = lines!(ax, sol.t, _original_M, color=:red, linewidth=4)
axislegend(ax, [L1, L2], [L"$ $Noisy", L"$ $Original"], L"$ $Solution", position=:rb)
ax = Axis(fig[2, 1], width=1200, height=400,
    xlabel=L"t", ylabel=L"L(t)", titlealign=:left)
L1 = lines!(ax, sol.t, L, color=:black, linewidth=6)
L2 = lines!(ax, sol.t, _original_L, color=:red, linewidth=4)
axislegend(ax, [L1, L2], [L"$ $Noisy", L"$ $Original"], L"$ $Solution", position=:rb)
resize_to_layout!(fig)
fig_path = normpath(@__DIR__, "..", "docs", "src", "figures")
@test_reference joinpath(fig_path, "noisy_pde_data.png") fig

## Step 3: Define the LikelihoodProblem 
function mb_loglik(θ, data, integrator)
    λ, K, D, κ = θ
    M, L, n, σM, σL = data
    prob = integrator.p
    Δx = prob.geometry.mesh_points[2] - prob.geometry.mesh_points[1]
    prob.diffusion_parameters[1] = D
    prob.reaction_parameters[1][1] = λ
    prob.reaction_parameters[2][1] = K
    prob.boundary_conditions.moving_boundary.p[1] = κ
    reinit!(integrator)
    solve!(integrator)
    if !SciMLBase.successful_retcode(integrator.sol)
        return typemin(eltype(θ))
    end
    ℓ = zero(eltype(θ))
    for i in 1:n
        Mᵢ = compute_average(integrator.sol.u[i], Δx)
        ℓ = ℓ - 0.5log(2π * σM^2) - 0.5(M[i] - Mᵢ)^2 / σM^2
        Lᵢ = integrator.sol.u[i][end]
        ℓ = ℓ - 0.5log(2π * σL^2) - 0.5(L[i] - Lᵢ)^2 / σL^2
    end
    return ℓ
end
lb = [0.2, 0.5, 0.1, 10.0]
ub = [8.0, 8.0, 8.0, 40.0]
θ₀ = [0.5, 2.2, 0.5, 25.0]
syms = [:λ, :K, :D, :κ]
n = findlast(sol.t .≤ 20.0) # using t ≤ 20.0 for estimation 
p = (M, L, n, σM, σL)
integrator = init(prob, TRBDF2(linsolve=KLUFactorization(; reuse_symbolic=false)); # https://github.com/JuliaSparse/KLU.jl/issues/12
    saveat=T / 250, verbose=false)
likprob = LikelihoodProblem(
    mb_loglik, θ₀, integrator;
    syms=syms,
    data=p,
    f_kwargs=(adtype=AutoFiniteDiff(),), # see https://github.com/SciML/Optimization.jl/issues/548
    prob_kwargs=(lb=lb, ub=ub))

## Step 4: Parameter estimation 
@time mle_sol = mle(likprob, NLopt.LN_NELDERMEAD();
    xtol_abs=1e-11, xtol_rel=1e-11,
    ftol_abs=1e-11, ftol_rel=1e-11)

@time prof = profile(likprob, mle_sol;
    xtol_abs=1e-4, xtol_rel=1e-4,
    ftol_abs=1e-4, ftol_rel=1e-4,
    maxiters=100, maxtime=600,
    parallel=true, resolution=60,
    min_steps=20,
    next_initial_estimate_method=:interp)

fig = plot_profiles(prof;
    latex_names=[L"\lambda", L"K", L"D", L"\kappa"],
    show_mles=true,
    shade_ci=true,
    true_vals=[1.1, 3.0, 0.89, 20.0],
    fig_kwargs=(fontsize=41,),
    axis_kwargs=(width=600, height=300))
resize_to_layout!(fig)
@test_reference joinpath(fig_path, "pde_profiles.png") fig

## Step 5: Prediction intervals 
function prediction_function!(q, θ::AbstractVector{T}, data) where {T}
    λ, K, D, κ = θ
    prob, t = data
    prob.diffusion_parameters[1] = D
    prob.reaction_parameters[1][1] = λ
    prob.reaction_parameters[2][1] = K
    prob.boundary_conditions.moving_boundary.p[1] = κ
    sol = solve(prob, TRBDF2(linsolve=KLUFactorization()), saveat=t)
    q .= compute_average(sol)
    return nothing
end
t_many_pts = LinRange(0, T, 1000)
pred_data = (prob, t_many_pts)
q_prototype = zero(t_many_pts)
individual_intervals, union_intervals, q_vals, param_ranges =
    get_prediction_intervals(prediction_function!, prof, pred_data; parallel=true,
        q_prototype)

exact_soln = zero(t_many_pts)
mle_soln = zero(t_many_pts)
prediction_function!(exact_soln, [1.1, 3.0, 0.89, 20.0], pred_data)
prediction_function!(mle_soln, get_mle(mle_sol), pred_data)

# Plot the parameter-wise intervals 
fig = Figure(fontsize=38)
alp = ('a', 'b', 'c', 'd', 'e')
latex_names = [L"\lambda", L"K", L"D", L"\kappa"]
for i in 1:4
    ax = Axis(fig[i < 3 ? 1 : 2, mod1(i, 2)], title=L"(%$(alp[i])): Profile-wise PI for %$(latex_names[i])",
        titlealign=:left, width=600, height=300, xlabel=L"t", ylabel=L"M(t)")
    vlines!(ax, [20.0], color=:purple, linestyle=:dash, linewidth=2)
    lines!(ax, t_many_pts, exact_soln, color=:red, linewidth=3)
    lines!(ax, t_many_pts, mle_soln, color=:blue, linestyle=:dash, linewidth=3)
    lines!(ax, t_many_pts, getindex.(individual_intervals[i], 1), color=:black, linewidth=3)
    lines!(ax, t_many_pts, getindex.(individual_intervals[i], 2), color=:black, linewidth=3)
    band!(ax, t_many_pts, getindex.(individual_intervals[i], 1), getindex.(individual_intervals[i], 2), color=(:grey, 0.35))
end

# Plot the union intervals
ax = Axis(fig[3, 1:2], title=L"(f):$ $ Union of all intervals", titlealign=:left,
    width=1200, height=300, xlabel=L"t", ylabel=L"M(t)")
band!(ax, t_many_pts, getindex.(union_intervals, 1), getindex.(union_intervals, 2), color=(:grey, 0.35))
lines!(ax, t_many_pts, getindex.(union_intervals, 1), color=:black, linewidth=3)
lines!(ax, t_many_pts, getindex.(union_intervals, 2), color=:black, linewidth=3)
lines!(ax, t_many_pts, exact_soln, color=:red, linewidth=3)
lines!(ax, t_many_pts, mle_soln, color=:blue, linestyle=:dash, linewidth=3)
vlines!(ax, [20.0], color=:purple, linestyle=:dash, linewidth=2)

resize_to_layout!(fig)
fig
@test_reference joinpath(fig_path, "pde_prediction_intervals.png") fig
