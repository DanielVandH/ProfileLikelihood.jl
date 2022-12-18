using ..ProfileLikelihood 
using LinearAlgebra
using PreallocationTools
using Random 
using Distributions
using LinearSolve 
using OrdinaryDiffEq
using Optimization
using OptimizationOptimJL
using OptimizationNLopt 
using FiniteVolumeMethod 
using DelaunayTriangulation
include("templates.jl")

######################################################
## ProfileLikelihood 
######################################################
## Test that we can correctly construct the parameter ranges
lb = -2.0
ub = 2.0
mpt = 0.0
res = 50
lr, ur = ProfileLikelihood.construct_profile_ranges(lb, ub, mpt, res)
@test lr == LinRange(0.0, -2.0, 50)
@test ur == LinRange(0.0, 2.0, 50)

## Test that we can construct the parameter ranges from a solution 
prob, loglikk, θ, dat = multiple_linear_regression()
true_ℓ = loglikk(reduce(vcat, θ), dat)
lb = (1e-12, -3.0, 0.0, 0.0, -6.0)
ub = (0.2, 0.0, 3.0, 3.0, 6.0)
res = 27
ug = RegularGrid(lb, ub, res)
sol = grid_search(prob, ug; save_vals=Val(false))

res2 = 50
ranges = ProfileLikelihood.construct_profile_ranges(sol, lb, ub, res2)
@test all(i -> ranges[i] == (LinRange(sol.mle[i], lb[i], res2), LinRange(sol.mle[i], ub[i], res2)), eachindex(lb))

res2 = (50, 102, 50, 671, 123)
ranges = ProfileLikelihood.construct_profile_ranges(sol, lb, ub, res2)
@test all(i -> ranges[i] == (LinRange(sol.mle[i], lb[i], res2[i]), LinRange(sol.mle[i], ub[i], res2[i])), eachindex(lb))

## Test that we can extract a problem and solution 
prob, loglikk, θ, dat = multiple_linear_regression()
true_ℓ = loglikk(reduce(vcat, θ), dat)
lb = (1e-12, -3.0, 0.0, 0.0, -6.0)
ub = (0.2, 0.0, 3.0, 3.0, 6.0)
res = 27
ug = RegularGrid(lb, ub, res)
sol = grid_search(prob, ug; save_vals=Val(false))
_opt_prob, _mles, _ℓmax = ProfileLikelihood.extract_problem_and_solution(prob, sol)
@test _opt_prob === sol.problem.problem
@test _mles == sol.mle
@test !(_mles === sol.mle)
@test _ℓmax == sol.maximum

## Test that we can prepare the profile results correctly 
N = 5
T = Float64
F = Float64
_θ, _prof, _other_mles, _splines, _confidence_intervals = ProfileLikelihood.prepare_profile_results(N, T, F)
@test _θ == Dict{Int64,Vector{T}}([])
@test _prof == Dict{Int64,Vector{T}}([])
@test _other_mles == Dict{Int64,Vector{Vector{T}}}([])
@test _confidence_intervals == Dict{Int64,ProfileLikelihood.ConfidenceInterval{T,F}}([])

## Test that we can correctly normalise the objective function 
shifted_opt_prob = ProfileLikelihood.normalise_objective_function(_opt_prob, _ℓmax, false)
@test shifted_opt_prob === _opt_prob
shifted_opt_prob = ProfileLikelihood.normalise_objective_function(_opt_prob, _ℓmax, true)
@test shifted_opt_prob.f(reduce(vcat, θ), dat) ≈ -(loglikk(reduce(vcat, θ), dat) - _ℓmax)
@inferred shifted_opt_prob.f(reduce(vcat, θ), dat)

## Test that we can prepare the cache 
n = 2
_left_profile_vals, _right_profile_vals,
_left_param_vals, _right_param_vals,
_left_other_mles, _right_other_mles,
_combined_profiles, _combined_param_vals, _combined_other_mles,
_cache, _sub_cache = ProfileLikelihood.prepare_cache_vectors(n, N, ranges[n], _mles)
@test _left_profile_vals == Vector{T}([])
@test _right_profile_vals == Vector{T}([])
@test _left_param_vals == Vector{T}([])
@test _right_param_vals == Vector{T}([])
@test _left_other_mles == Vector{Vector{T}}([])
@test _right_other_mles == Vector{Vector{T}}([])
@test _combined_profiles == Vector{T}([])
@test _combined_param_vals == Vector{T}([])
@test _combined_other_mles == Vector{Vector{T}}([])
@test _cache.dual_du == DiffCache(zeros(T, N)).dual_du
@test _cache.du == DiffCache(zeros(T, N)).du
@test _sub_cache == _mles[[1, 3, 4, 5]]

## Test the construction of the ProfileLikelihoodSolution 
Random.seed!(98871)
n = 300
β = [-1.0, 1.0, 0.5, 3.0]
σ = 0.05
x₁ = rand(Uniform(-1, 1), n)
x₂ = rand(Normal(1.0, 0.5), n)
X = hcat(ones(n), x₁, x₂, x₁ .* x₂)
ε = rand(Normal(0.0, σ), n)
y = X * β + ε
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
θ₀ = ones(5)
prob = LikelihoodProblem(loglik_fnc, θ₀;
    data=dat,
    f_kwargs=(adtype=Optimization.AutoForwardDiff(),),
    prob_kwargs=(lb=[0.0, -5.0, -5.0, -5.0, -5.0],
        ub=[15.0, 15.0, 15.0, 15.0, 15.0]),
    syms=[:σ, :β₀, :β₁, :β₂, :β₃])
sol = mle(prob, Optim.LBFGS())
prof = profile(prob, sol, [1, 3])
@test ProfileLikelihood.get_parameter_values(prof) == prof.parameter_values
@test ProfileLikelihood.get_parameter_values(prof, 1) == prof.parameter_values[1]
@test ProfileLikelihood.get_parameter_values(prof, :σ) == prof.parameter_values[1]
@test ProfileLikelihood.get_parameter_values(prof, 3) == prof.parameter_values[3]
@test ProfileLikelihood.get_parameter_values(prof, :β₁) == prof.parameter_values[3]
@test ProfileLikelihood.get_profile_values(prof) == prof.profile_values
@test ProfileLikelihood.get_profile_values(prof, 3) == prof.profile_values[3]
@test ProfileLikelihood.get_profile_values(prof, :σ) == prof.profile_values[1]
@test ProfileLikelihood.get_likelihood_problem(prof) == prof.likelihood_problem == prob
@test ProfileLikelihood.get_likelihood_solution(prof) == prof.likelihood_solution == sol
@test ProfileLikelihood.get_splines(prof) == prof.splines
@test ProfileLikelihood.get_splines(prof, 3) == prof.splines[3]
@test ProfileLikelihood.get_splines(prof, :σ) == prof.splines[1]
@test ProfileLikelihood.get_confidence_intervals(prof) == prof.confidence_intervals
@test ProfileLikelihood.get_confidence_intervals(prof, 1) == prof.confidence_intervals[1]
@test ProfileLikelihood.get_confidence_intervals(prof, :β₁) == prof.confidence_intervals[3]
@test ProfileLikelihood.get_other_mles(prof) == prof.other_mles
@test ProfileLikelihood.get_other_mles(prof, 3) == prof.other_mles[3]
@test ProfileLikelihood.get_syms(prof) == prob.syms == [:σ, :β₀, :β₁, :β₂, :β₃]
@test ProfileLikelihood.get_syms(prof, 4) == :β₂
@test SciMLBase.sym_to_index(:σ, prof) == 1
@test SciMLBase.sym_to_index(:β₀, prof) == 2
@test SciMLBase.sym_to_index(:β₁, prof) == 3
@test SciMLBase.sym_to_index(:β₂, prof) == 4
@test SciMLBase.sym_to_index(:β₃, prof) == 5
@test ProfileLikelihood.profiled_parameters(prof) == [1, 3]
@test ProfileLikelihood.number_of_profiled_parameters(prof) == 2

## Test that views are working correctly on the ProfileLikelihoodSolution
i = 1
prof_view = prof[i]
@test ProfileLikelihood.get_parent(prof_view) == prof
@test ProfileLikelihood.get_index(prof_view) == i
@test ProfileLikelihood.get_parameter_values(prof_view) == prof.parameter_values[i]
@test ProfileLikelihood.get_parameter_values(prof_view, 1) == prof.parameter_values[i][1]
@test ProfileLikelihood.get_parameter_values(prof_view, 3) == prof.parameter_values[i][3]
@test ProfileLikelihood.get_profile_values(prof_view) == prof.profile_values[i]
@test ProfileLikelihood.get_profile_values(prof_view, 3) == prof.profile_values[i][3]
@test ProfileLikelihood.get_likelihood_problem(prof_view) == prof.likelihood_problem == prob
@test ProfileLikelihood.get_likelihood_solution(prof_view) == prof.likelihood_solution == sol
@test ProfileLikelihood.get_splines(prof_view) == prof.splines[i]
@test ProfileLikelihood.get_confidence_intervals(prof_view) == prof.confidence_intervals[i]
@test ProfileLikelihood.get_confidence_intervals(prof_view, 1) == prof.confidence_intervals[i][1]
@test ProfileLikelihood.get_other_mles(prof_view) == prof.other_mles[i]
@test ProfileLikelihood.get_other_mles(prof_view, 3) == prof.other_mles[i][3]
@test ProfileLikelihood.get_syms(prof_view) == :σ
@test prof[:β₁] == prof[3]

## Test that we can correctly call the profiles 
x = prof.splines[i].itp.knots
@test prof_view(x) == prof(x, i) == prof.splines[i](x)

## Threads 
Random.seed!(98871)
n = 30000
β = [-1.0, 1.0, 0.5, 3.0]
σ = 0.4
x₁ = rand(Uniform(-1, 1), n)
x₂ = rand(Normal(1.0, 0.5), n)
X = hcat(ones(n), x₁, x₂, x₁ .* x₂)
ε = rand(Normal(0.0, σ), n)
y = X * β + ε
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
θ₀ = ones(5)
prob = LikelihoodProblem(loglik_fnc, θ₀;
    data=dat,
    f_kwargs=(adtype=Optimization.AutoForwardDiff(),),
    prob_kwargs=(lb=[0.0, -5.0, -5.0, -5.0, -5.0],
        ub=[15.0, 15.0, 15.0, 15.0, 15.0]),
    syms=[:σ, :β₀, :β₁, :β₂, :β₃])
sol = mle(prob, Optim.LBFGS())
prof_serial = profile(prob, sol)
prof_parallel = profile(prob, sol; parallel=true)
@test all(i -> abs((prof_parallel.confidence_intervals[i].lower - prof_serial.confidence_intervals[i].lower) / prof_serial.confidence_intervals[i].lower) < 1e-2, 1:5)
@test all(i -> abs((prof_parallel.confidence_intervals[i].upper - prof_serial.confidence_intervals[i].upper) / prof_serial.confidence_intervals[i].upper) < 1e-2, 1:5)
@test all(i -> prof_parallel.parameter_values[i] ≈ prof_serial.parameter_values[i], 1:5)
@test all(i -> prof_parallel.profile_values[i] ≈ prof_serial.profile_values[i], 1:5)

## More parallel testing for a problem involving an integrator
Random.seed!(2929911002)
u₀, λ, K, n, T = 0.5, 1.0, 1.0, 100, 10.0
t = LinRange(0, T, n)
u = @. K * u₀ * exp(λ * t) / (K - u₀ + u₀ * exp(λ * t))
σ = 0.1
uᵒ = u .+ [0.0, σ * randn(length(u) - 1)...] # add some noise 
@inline function ode_fnc(u, p, t)
    local λ, K
    λ, K = p
    du = λ * u * (1 - u / K)
    return du
end
@inline function loglik_fnc2(θ, data, integrator)
    local uᵒ, n, λ, K, σ, u0
    uᵒ, n = data
    λ, K, σ, u0 = θ
    integrator.p[1] = λ
    integrator.p[2] = K
    reinit!(integrator, u0)
    solve!(integrator)
    return gaussian_loglikelihood(uᵒ, integrator.sol.u, σ, n)
end
θ₀ = [0.7, 2.0, 0.15, 0.4]
lb = [0.0, 1e-6, 1e-6, 0.0]
ub = [10.0, 10.0, 10.0, 10.0]
syms = [:λ, :K, :σ, :u₀]
prob = LikelihoodProblem(
    loglik_fnc2, θ₀, ode_fnc, u₀, (0.0, T); # u₀ is just a placeholder IC in this case
    syms=syms,
    data=(uᵒ, n),
    ode_parameters=[1.0, 1.0], # temp values for [λ, K],
    ode_kwargs=(verbose=false, saveat=t),
    f_kwargs=(adtype=Optimization.AutoFiniteDiff(),),
    prob_kwargs=(lb=lb, ub=ub),
    ode_alg=Tsit5()
)
sol = mle(prob, NLopt.LN_NELDERMEAD)
prof1 = profile(prob, sol)
prof2 = profile(prob, sol; parallel=true)

@test prof1.confidence_intervals[1].lower ≈ prof2.confidence_intervals[1].lower rtol = 1e-1
@test prof1.confidence_intervals[2].lower ≈ prof2.confidence_intervals[2].lower rtol = 1e-1
@test prof1.confidence_intervals[3].lower ≈ prof2.confidence_intervals[3].lower rtol = 1e-1
@test prof1.confidence_intervals[4].lower ≈ prof2.confidence_intervals[4].lower rtol = 1e-1
@test prof1.confidence_intervals[1].upper ≈ prof2.confidence_intervals[1].upper rtol = 1e-1
@test prof1.confidence_intervals[2].upper ≈ prof2.confidence_intervals[2].upper rtol = 1e-1
@test prof1.confidence_intervals[3].upper ≈ prof2.confidence_intervals[3].upper rtol = 1e-1
@test prof1.confidence_intervals[4].upper ≈ prof2.confidence_intervals[4].upper rtol = 1e-1
@test prof1.other_mles[1] ≈ prof2.other_mles[1] rtol = 1e-0 atol = 1e-1
@test prof1.other_mles[2] ≈ prof2.other_mles[2] rtol = 1e-1 atol = 1e-1
@test prof1.other_mles[3] ≈ prof2.other_mles[3] rtol = 1e-3 atol = 1e-1
@test prof1.other_mles[4] ≈ prof2.other_mles[4] rtol = 1e-0 atol = 1e-1
@test prof1.parameter_values[1] ≈ prof2.parameter_values[1] rtol = 1e-1
@test prof1.parameter_values[2] ≈ prof2.parameter_values[2] rtol = 1e-3
@test prof1.parameter_values[3] ≈ prof2.parameter_values[3] rtol = 1e-3
@test prof1.parameter_values[4] ≈ prof2.parameter_values[4] rtol = 1e-3
@test issorted(prof1.parameter_values[1])
@test issorted(prof1.parameter_values[2])
@test issorted(prof1.parameter_values[3])
@test issorted(prof1.parameter_values[4])
@test issorted(prof2.parameter_values[1])
@test issorted(prof2.parameter_values[2])
@test issorted(prof2.parameter_values[3])
@test issorted(prof2.parameter_values[4])
@test prof1.profile_values[1] ≈ prof2.profile_values[1] rtol = 1e-1
@test prof1.profile_values[2] ≈ prof2.profile_values[2] rtol = 1e-1
@test prof1.profile_values[3] ≈ prof2.profile_values[3] rtol = 1e-1
@test prof1.profile_values[4] ≈ prof2.profile_values[4] rtol = 1e-1
@test prof1.splines[1].itp.knots ≈ prof2.splines[1].itp.knots
@test prof1.splines[2].itp.knots ≈ prof2.splines[2].itp.knots
@test prof1.splines[3].itp.knots ≈ prof2.splines[3].itp.knots
@test prof1.splines[4].itp.knots ≈ prof2.splines[4].itp.knots

## More parallel testing for a problem involving a PDE 
# Need to setup 
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
r = 0.07
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
final_time = 0.2
k = [9.0]
prob = FVMProblem(mesh, BCs; iip_flux,
    flux_function=flux, reaction_function=R,
    initial_condition=initc, final_time,
    flux_parameters=k)
alg = TRBDF2(linsolve=UMFPACKFactorization(; reuse_symbolic=false))
sol = solve(prob, alg; specialization=SciMLBase.FullSpecialize, saveat=0.01)

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
true_M = M ./ first(M)
Random.seed!(29922881)
σ = 0.01
true_M .+= σ * randn(length(M))
function ProfileLikelihood.construct_integrator(prob::FVMProblem, alg; ode_problem_kwargs, kwargs...)
    ode_problem = ODEProblem(prob; no_saveat=false, ode_problem_kwargs...)
    return ProfileLikelihood.construct_integrator(ode_problem, alg; kwargs...)
end
jac = float.(FiniteVolumeMethod.jacobian_sparsity(prob))
fvm_integrator = construct_integrator(prob, alg; ode_problem_kwargs=(jac_prototype=jac, saveat=0.01, parallel=true))
function loglik_fvm(θ::AbstractVector{T}, param, integrator) where {T}
    _k, _c, _u₀, _σ = θ
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
    (; mass_data, mass_cache, shape_cache) = param
    compute_mass!(mass_cache, shape_cache, integrator.sol, prob)
    mass_cache ./= first(mass_cache)
    ℓ = @views gaussian_loglikelihood(mass_data[2:end], mass_cache[2:end], _σ, length(mass_data) - 1) # first value is 1 for both
    if isinf(ℓ)
        ℓ = -ℓ # make -typemin(T)
    end
    return ℓ
end
likprob = LikelihoodProblem(
    loglik_fvm,
    [8.54, 0.98, 29.83, 0.05],
    fvm_integrator;
    syms=[:k, :c, :u₀, :σ],
    data=(prob=prob, mass_data=true_M, mass_cache=zeros(length(true_M)), shape_cache=zeros(3)),
    f_kwargs=(adtype=Optimization.AutoFiniteDiff(),),
    prob_kwargs=(lb=[3.0, 0.0, 25.0, 1e-6],
        ub=[15.0, 2.0, 100.0, 0.2])
)
mle_sol = mle(likprob, NLopt.LN_BOBYQA)

prof1 = profile(likprob, mle_sol; ftol_abs=1e-4, ftol_rel=1e-4, xtol_abs=1e-4, xtol_rel=1e-4,
    resolution=50)
prof2 = profile(likprob, mle_sol; ftol_abs=1e-4, ftol_rel=1e-4, xtol_abs=1e-4, xtol_rel=1e-4,
    resolution=50, parallel=true)

using BenchmarkTools
#=b1 = @benchmark profile($likprob, $mle_sol; ftol_abs=$1e-4, ftol_rel=$1e-4, xtol_abs=$1e-4,
    xtol_rel=$1e-4,
    resolution=$80)
b2 = @benchmark profile($likprob, $mle_sol; ftol_abs=$1e-4, ftol_rel=$1e-4, xtol_abs=$1e-4,
    xtol_rel=$1e-4,
    resolution=$80, parallel=$true)=#

@test prof1.confidence_intervals[1].lower ≈ prof2.confidence_intervals[1].lower rtol = 1e-1
@test prof1.confidence_intervals[2].lower ≈ prof2.confidence_intervals[2].lower rtol = 1e-1
@test prof1.confidence_intervals[3].lower ≈ prof2.confidence_intervals[3].lower rtol = 1e-1
@test prof1.confidence_intervals[4].lower ≈ prof2.confidence_intervals[4].lower rtol = 1e-1
@test prof1.confidence_intervals[1].upper ≈ prof2.confidence_intervals[1].upper rtol = 1e-1
@test prof1.confidence_intervals[2].upper ≈ prof2.confidence_intervals[2].upper rtol = 1e-1
@test prof1.confidence_intervals[3].upper ≈ prof2.confidence_intervals[3].upper rtol = 1e-1
@test prof1.confidence_intervals[4].upper ≈ prof2.confidence_intervals[4].upper rtol = 1e-1
@test prof1.other_mles[1] ≈ prof2.other_mles[1] rtol = 1e-0 atol = 1e-1
@test prof1.other_mles[2] ≈ prof2.other_mles[2] rtol = 1e-1 atol = 1e-1
@test prof1.other_mles[3] ≈ prof2.other_mles[3] rtol = 1e-3 atol = 1e-1
@test prof1.other_mles[4] ≈ prof2.other_mles[4] rtol = 1e-0 atol = 1e-1
@test prof1.parameter_values[1] ≈ prof2.parameter_values[1] rtol = 1e-1
@test prof1.parameter_values[2] ≈ prof2.parameter_values[2] rtol = 1e-3
@test prof1.parameter_values[3] ≈ prof2.parameter_values[3] rtol = 1e-3
@test prof1.parameter_values[4] ≈ prof2.parameter_values[4] rtol = 1e-3
@test issorted(prof1.parameter_values[1])
@test issorted(prof1.parameter_values[2])
@test issorted(prof1.parameter_values[3])
@test issorted(prof1.parameter_values[4])
@test issorted(prof2.parameter_values[1])
@test issorted(prof2.parameter_values[2])
@test issorted(prof2.parameter_values[3])
@test issorted(prof2.parameter_values[4])
@test prof1.profile_values[1] ≈ prof2.profile_values[1] rtol = 1e-1
@test prof1.profile_values[2] ≈ prof2.profile_values[2] rtol = 1e-1
@test prof1.profile_values[3] ≈ prof2.profile_values[3] rtol = 1e-1
@test prof1.profile_values[4] ≈ prof2.profile_values[4] rtol = 1e-1
@test prof1.splines[1].itp.knots ≈ prof2.splines[1].itp.knots
@test prof1.splines[2].itp.knots ≈ prof2.splines[2].itp.knots
@test prof1.splines[3].itp.knots ≈ prof2.splines[3].itp.knots
@test prof1.splines[4].itp.knots ≈ prof2.splines[4].itp.knots