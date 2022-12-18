######################################################
## Example IV: Heat equation on a square plate
######################################################
## Define the problem. See FiniteVolumeMethod.jl
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
    flux_parameters=deepcopy(k))

## Generate some data.
alg = TRBDF2(linsolve=KLUFactorization(; reuse_symbolic=false))
sol = solve(prob, alg; specialization=SciMLBase.FullSpecialize, saveat=0.01)

## Let us compute the mass at each time and then add some noise to it
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
M = zeros(length(sol.t))
αβγ = zeros(3)
compute_mass!(M, αβγ, sol, prob)
true_M = deepcopy(M)

Random.seed!(29922881)
σ = 0.1
true_M .+= σ * randn(length(M))

## We need to now construct the integrator. Here's a method for converting an FVMProblem into an integrator. 
function ProfileLikelihood.construct_integrator(prob::FVMProblem, alg; ode_problem_kwargs, kwargs...)
    ode_problem = ODEProblem(prob; no_saveat=false, ode_problem_kwargs...)
    return ProfileLikelihood.construct_integrator(ode_problem, alg; kwargs...)
end
jac = float.(FiniteVolumeMethod.jacobian_sparsity(prob))
fvm_integrator = construct_integrator(prob, alg; ode_problem_kwargs=(jac_prototype=jac, saveat=0.01, parallel=true))

## Now define the likelihood problem 
@inline function loglik_fvm(θ::AbstractVector{T}, param, integrator) where {T}
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
    @show ℓ
    return ℓ
end

## Now define the problem
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

## Find the MLEs 
t0 = time()
mle_sol = mle(likprob, (NLopt.GN_DIRECT_L_RAND(), NLopt.LN_BOBYQA); ftol_abs=1e-8, ftol_rel=1e-8, xtol_abs=1e-8, xtol_rel=1e-8) # global, and then refine with a local algorithm
t1 = time()
@show t1 - t0

### Now profile
@time prof = profile(likprob, mle_sol; alg=NLopt.LN_BOBYQA,
    ftol_abs=1e-4, ftol_rel=1e-4, xtol_abs=1e-4, xtol_rel=1e-4,
    resolution=60)
@time _prof = profile(likprob, mle_sol; alg=NLopt.LN_BOBYQA,
    ftol_abs=1e-4, ftol_rel=1e-4, xtol_abs=1e-4, xtol_rel=1e-4,
    resolution=60, parallel=true)

prof1 = prof
prof2 = _prof

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
SAVE_FIGURE && save("figures/heat_pde_example.png", fig)

### Now estimate only two parameters
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

# small test to check with the exact values, larger test later
grid = RegularGrid(get_lower_bounds(likprob_2), get_upper_bounds(likprob_2), 10)
@time gs, lik_vals = grid_search(likprob_2, grid; save_vals=Val(true), parallel=Val(true));
@time _gs, _lik_vals = grid_search(likprob_2, grid; save_vals=Val(true), parallel=Val(false));
exact_lik_vals = [
    likprob_2.log_likelihood_function([k, c], likprob_2.data) for k in LinRange(3, 15, 10), c in LinRange(0, 250, 10)
]
@test lik_vals ≈ exact_lik_vals
@test _lik_vals ≈ exact_lik_vals

grid = RegularGrid(get_lower_bounds(likprob_2), get_upper_bounds(likprob_2), 50)
@time gs, lik_vals = grid_search(likprob_2, grid; save_vals=Val(true), parallel=Val(true));
@time _gs, _lik_vals = grid_search(likprob_2, grid; save_vals=Val(true), parallel=Val(false));
@test lik_vals ≈ _lik_vals

@test get_mle(gs) ≈ get_mle(_gs)
@test get_maximum(gs) ≈ get_maximum(_gs)
@test gs[:k] ≈ 7.408163265306122
@test gs[:u₀] ≈ 51.0204081632653

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
SAVE_FIGURE && save("figures/heat_pde_contour_example.png", fig)

likprob_2 = update_initial_estimate(likprob_2, gs)
mle_sol = mle(likprob_2, NLopt.LN_BOBYQA; ftol_abs=1e-8, ftol_rel=1e-8, xtol_abs=1e-8, xtol_rel=1e-8)
@time prof = profile(likprob_2, mle_sol; ftol_abs=1e-4, ftol_rel=1e-4, xtol_abs=1e-4, xtol_rel=1e-4, parallel=true)

fig = plot_profiles(prof; nrow=1, ncol=3,
    latex_names=[L"k", L"u_0"],
    true_vals=[k[1], u₀],
    fig_kwargs=(fontsize=38, resolution=(1441.9216f0, 470.17322f0)),
    axis_kwargs=(width=600, height=300))
scatter!(fig.content[1], get_parameter_values(prof, :k), get_profile_values(prof, :k), color=:black, markersize=9)
scatter!(fig.content[2], get_parameter_values(prof, :u₀), get_profile_values(prof, :u₀), color=:black, markersize=9)
SAVE_FIGURE && save("figures/heat_pde_example_2.png", fig)

## Prediction interval for mass 
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
t_many_pts = LinRange(prob.initial_time, prob.final_time, 250)
jac = FiniteVolumeMethod.jacobian_sparsity(prob)
prediction_data = (c=c, prob=prob, t=t_many_pts, alg=alg, jac=jac)
parameter_wise, union_intervals, all_curves, param_range =
    get_prediction_intervals(compute_mass_function, prof, prediction_data; q_type=Vector{Float64})

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

SAVE_FIGURE && save("figures/heat_pde_example_mass.png", fig)
