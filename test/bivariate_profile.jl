using Random
using ..ProfileLikelihood
using Optimization
using OrdinaryDiffEq
using CairoMakie
using LaTeXStrings
using OptimizationNLopt
using Distributions
using PreallocationTools
using OptimizationOptimJL
using LinearAlgebra
using InvertedIndices
using Contour
using OffsetArrays
using Interpolations
using DelaunayTriangulation
using PolygonInbounds
const SAVE_FIGURE = false

## Constructing the profile grids 
# Setup
Random.seed!(98871)
n = 600
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
    prob_kwargs=(
        lb=[0.0, -Inf, -Inf, -Inf, -Inf],
        ub=Inf * ones(5)
    ),
    syms=[:σ, :β₀, :β₁, :β₂, :β₃]
)
sol = mle(prob, Optim.LBFGS())

# The grid 
n = ((1, 2), (2, 3), (5, 1))
@test_throws "The provided parameter bounds for 1 and 2 must be finite." construct_profile_grids(n, sol, get_lower_bounds(prob), get_upper_bounds(prob), 23)
grids = construct_profile_grids(n, sol, [1e-12, -12.0, -3.3, -5.7, -11.0], [10.0, 12.0, 8.3, 9.3, 10.0], 23)
@test collect(keys(grids)) == [(1, 2), (5, 1), (2, 3)]
@test get_range(grids[n[1]], 1) == get_range(ProfileLikelihood.FusedRegularGrid([1e-12, -12.0], [10.0, 12.0], [sol[1], sol[2]], 23), 1)
@test get_range(grids[n[1]], 2) == get_range(ProfileLikelihood.FusedRegularGrid([1e-12, -12.0], [10.0, 12.0], [sol[1], sol[2]], 23), 2)
@test get_range(grids[n[2]], 1) == get_range(ProfileLikelihood.FusedRegularGrid([-12.0, -3.3], [12.0, 8.3], [sol[2], sol[3]], 23), 1)
@test get_range(grids[n[2]], 2) == get_range(ProfileLikelihood.FusedRegularGrid([-12.0, -3.3], [12.0, 8.3], [sol[2], sol[3]], 23), 2)
@test get_range(grids[n[3]], 1) == get_range(ProfileLikelihood.FusedRegularGrid([-11.0, 1e-12], [10.0, 10.0], [sol[5], sol[1]], 23), 1)
@test get_range(grids[n[3]], 2) == get_range(ProfileLikelihood.FusedRegularGrid([-11.0, 1e-12], [10.0, 10.0], [sol[5], sol[1]], 23), 2)

n = ((1, 2), (5, 1), (2, 3), (2, 1), (3, 5))
lb = [1e-12, -12.0, -3.3, -5.7, -11.0]
ub = [10.0, 12.0, 8.3, 9.3, 10.0]
res = [23, 57, 101, 83, 47]
grids = construct_profile_grids(n, sol, lb, ub, res)
@test collect(keys(grids)) == [(1, 2), (5, 1), (2, 1), (2, 3), (3, 5)]
for i in eachindex(n)
    u, v = n[i]
    for j in 1:2
        @test get_range(grids[(u, v)], j) == get_range(ProfileLikelihood.FusedRegularGrid(lb[[u, v]], ub[[u, v]], [sol[u], sol[v]], maximum(res[[u, v]])), j)
    end
end
for i in eachindex(n)
    u, v = n[i]
    @test grids[n[i]].resolutions == max(res[u], res[v])
end

n = ((1, 2), (5, 1), (2, 3), (2, 1), (5, 3))
lb = [1e-12, -12.0, -3.3, -5.7, -11.0]
ub = [10.0, 12.0, 8.3, 9.3, 10.0]
res = [(107, 43), (282, 103), (587, 203), (501), (10101, 100)]
grids = construct_profile_grids(n, sol, lb, ub, res)
for i in eachindex(n)
    u, v = n[i]
    for j in 1:2
        @test get_range(grids[(u, v)], j) == get_range(ProfileLikelihood.FusedRegularGrid(lb[[u, v]], ub[[u, v]], [sol[u], sol[v]], maximum([res[u]..., res[v]...])), j)
    end
end
for i in eachindex(n)
    u, v = n[i]
    @test grids[n[i]].resolutions == max(res[u]..., res[v]...)
end

## Constructing the layer iterator 
itr = ProfileLikelihood.LayerIterator(1)
@test eltype(itr) == CartesianIndex{2}
@test collect(itr) == CartesianIndex.([
    (-1, -1),
    (0, -1),
    (1, -1),
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0)
])
@test length(itr) == 8

itr = ProfileLikelihood.LayerIterator(2)
@test eltype(itr) == CartesianIndex{2}
@test length(itr) == 16
@test collect(itr) == CartesianIndex.([
    (-2, -2),
    (-1, -2),
    (0, -2),
    (1, -2),
    (2, -2),
    (2, -1),
    (2, 0),
    (2, 1),
    (2, 2),
    (1, 2),
    (0, 2),
    (-1, 2),
    (-2, 2),
    (-2, 1),
    (-2, 0),
    (-2, -1)
])

itr = ProfileLikelihood.LayerIterator(3)
@inferred first(itr)
@test eltype(itr) == CartesianIndex{2}
@test length(itr) == 24
@test collect(itr) == CartesianIndex.([
    (-3, -3),
    (-2, -3),
    (-1, -3),
    (0, -3),
    (1, -3),
    (2, -3),
    (3, -3),
    (3, -2),
    (3, -1),
    (3, 0),
    (3, 1),
    (3, 2),
    (3, 3),
    (2, 3),
    (1, 3),
    (0, 3),
    (-1, 3),
    (-2, 3),
    (-3, 3),
    (-3, 2),
    (-3, 1),
    (-3, 0),
    (-3, -1),
    (-3, -2)
])

itr = ProfileLikelihood.LayerIterator(4)
@inferred first(itr)
@test eltype(itr) == CartesianIndex{2}
@test length(itr) == 32
@test collect(itr) == CartesianIndex.([
    (-4, -4),
    (-3, -4),
    (-2, -4),
    (-1, -4),
    (0, -4),
    (1, -4),
    (2, -4),
    (3, -4),
    (4, -4),
    (4, -3),
    (4, -2),
    (4, -1),
    (4, 0),
    (4, 1),
    (4, 2),
    (4, 3),
    (4, 4),
    (3, 4),
    (2, 4),
    (1, 4),
    (0, 4),
    (-1, 4),
    (-2, 4),
    (-3, 4),
    (-4, 4),
    (-4, 3),
    (-4, 2),
    (-4, 1),
    (-4, 0),
    (-4, -1),
    (-4, -2),
    (-4, -3),
])

function ___testf(itr)
    s1 = 0.0
    s2 = 0.0
    for I in itr
        i, j = Tuple(I)
        s1 += i + 2j + rand()
        s2 += j - 3i - rand()
    end
    return s1 + s2
end
@inferred ___testf(itr)

## Preparing the results 
N = 3
T = Float64
F = Float64
θ, prof, other_mles, interpolants, confidence_regions = ProfileLikelihood.prepare_bivariate_profile_results(N, T, F)
@test θ == Dict{NTuple{2,Int64},NTuple{2,OffsetVector{T,Vector{T}}}}([])
@test prof == Dict{NTuple{2,Int64},OffsetMatrix{T,Matrix{T}}}([])
@test other_mles == Dict{NTuple{2,Int64},OffsetMatrix{Vector{T},Matrix{Vector{T}}}}([])
@test interpolants == Dict{NTuple{2,Int64},Interpolations.GriddedInterpolation{T,2,OffsetMatrix{T,Matrix{T}},Gridded{Linear{Throw{OnGrid}}},Tuple{OffsetVector{T,Vector{T}},OffsetVector{T,Vector{T}}}}}([])
@test confidence_regions == Dict{NTuple{2,Int64},ProfileLikelihood.ConfidenceRegion{Vector{T},F}}([])

## Preparing the cache vectors 
# Serial
n = (1, 3)
mles = [3.0, 4.0, 5.5, 6.6, 8.322]
res = 40
num_params = 5
normalise = true
ℓmax = 0.5993
prof, other, cache, sub_cache, fixed_vals, any_above = ProfileLikelihood.prepare_cache_vectors(n, mles, res, num_params, normalise, ℓmax)
@test prof == OffsetArray(zeros(2res + 1, 2res + 1), -res:res, -res:res)
_other = OffsetArray([zeros(3) for _ in 1:(2res+1), _ in 1:(2res+1)], -res:res, -res:res)
_other[0, 0] .= [4.0, 6.6, 8.322]
@test other == _other
_cache = DiffCache(zeros(num_params))
@test cache.any_du == _cache.any_du
@test cache.dual_du == _cache.dual_du
@test cache.du == _cache.du
_sub_cache = [4.0, 6.6, 8.322]
@test sub_cache == _sub_cache
_fixed = zeros(2)
@test fixed_vals == _fixed
@test prof[0, 0] == 0.0
@test !any_above

n = (1, 3)
mles = [3.0, 4.0, 5.5, 6.6, 8.322]
res = 40
num_params = 5
normalise = false
ℓmax = 0.5993
prof, other, cache, sub_cache, fixed_vals, any_above = ProfileLikelihood.prepare_cache_vectors(n, mles, res, num_params, normalise, ℓmax)
@test prof[0, 0] == 0.5993
_prof = OffsetArray(zeros(2res + 1, 2res + 1), -res:res, -res:res)
_prof[0, 0] = 0.5993
@test prof == _prof
_other = OffsetArray([zeros(3) for _ in 1:(2res+1), _ in 1:(2res+1)], -res:res, -res:res)
_other[0, 0] .= [4.0, 6.6, 8.322]
@test other == _other
_cache = DiffCache(zeros(num_params))
@test cache.any_du == _cache.any_du
@test cache.dual_du == _cache.dual_du
@test cache.du == _cache.du
_sub_cache = [4.0, 6.6, 8.322]
@test sub_cache == _sub_cache
_fixed = zeros(2)
@test fixed_vals == _fixed
@test !any_above

# Parallel 
n = (1, 3)
mles = [3.0, 4.0, 5.5, 6.6, 8.322]
res = 40
num_params = 5
normalise = true
ℓmax = 0.5993
prof, other, cache, sub_cache, fixed_vals, any_above = ProfileLikelihood.prepare_cache_vectors(n, mles, res, num_params, normalise, ℓmax; parallel=true)
@test prof == OffsetArray(zeros(2res + 1, 2res + 1), -res:res, -res:res)
_other = OffsetArray([zeros(3) for _ in 1:(2res+1), _ in 1:(2res+1)], -res:res, -res:res)
_other[0, 0] .= [4.0, 6.6, 8.322]
@test other == _other
_cache = DiffCache(zeros(num_params))
for i in 1:Base.Threads.nthreads()
    @test cache[i].any_du == _cache.any_du
    @test cache[i].dual_du == _cache.dual_du
    @test cache[i].du == _cache.du
end
_sub_cache = [4.0, 6.6, 8.322]
@test all(sub_cache[i] == _sub_cache for i in 1:Base.Threads.nthreads())
_fixed = zeros(2)
@test all(fixed_vals[i] == _fixed for i in 1:Base.Threads.nthreads())
@test prof[0, 0] == 0.0
@test all(!any_above[i] for i in 1:Base.Threads.nthreads())

n = (1, 3)
mles = [3.0, 4.0, 5.5, 6.6, 8.322]
res = 40
num_params = 5
normalise = false
ℓmax = 0.5993
prof, other, cache, sub_cache, fixed_vals, any_above = ProfileLikelihood.prepare_cache_vectors(n, mles, res, num_params, normalise, ℓmax; parallel=true)
@test prof[0, 0] == 0.5993
_prof = OffsetArray(zeros(2res + 1, 2res + 1), -res:res, -res:res)
_prof[0, 0] = 0.5993
@test prof == _prof
_other = OffsetArray([zeros(3) for _ in 1:(2res+1), _ in 1:(2res+1)], -res:res, -res:res)
_other[0, 0] .= [4.0, 6.6, 8.322]
@test other == _other
_cache = DiffCache(zeros(num_params))
for i in 1:Base.Threads.nthreads()
    @test cache[i].any_du == _cache.any_du
    @test cache[i].dual_du == _cache.dual_du
    @test cache[i].du == _cache.du
end
_sub_cache = [4.0, 6.6, 8.322]
@test all(sub_cache[i] == _sub_cache for i in 1:Base.Threads.nthreads())
_fixed = zeros(2)
@test all(fixed_vals[i] == _fixed for i in 1:Base.Threads.nthreads())
@test prof[0, 0] == 0.5993
@test all(!any_above[i] for i in 1:Base.Threads.nthreads())
_fixed = zeros(2)
@test all(fixed_vals[i] == _fixed for i in 1:Base.Threads.nthreads())
@test all(!any_above[i] for i in 1:Base.Threads.nthreads())

## Testing set_next_initial_estimate! 
n = (1, 3)
mles = [3.0, 4.0, 5.5, 6.6, 8.322]
res = 40
num_params = 5
normalise = false
ℓmax = 0.5993
prof, other, cache, sub_cache, fixed_vals = ProfileLikelihood.prepare_cache_vectors(n, mles, res, num_params, normalise, ℓmax)
for I in ProfileLikelihood.LayerIterator(1)
    prof[I] = rand()
    other[I] = rand(3)
end
_orig_other = deepcopy(other) # for checking aliasing later
lb = [2.0, 3.0, 1.0, 5.0, 4.0][[1, 3]]
ub = [15.0, 13.0, 27.0, 10.0, 13.0][[1, 3]]
centre = mles[[1, 3]]
grid = ProfileLikelihood.FusedRegularGrid(lb, ub, centre, res)
layer = 2
Id = CartesianIndex((2, 2))
fixed_vals = zeros(2)
fixed_vals = [grid[1, Id[1]], grid[2, Id[2]]]

# :mle
next_initial_estimate_method = :mle
ProfileLikelihood.set_next_initial_estimate!(sub_cache, other, Id, fixed_vals, grid, layer, n; next_initial_estimate_method=next_initial_estimate_method)
@test sub_cache == mles[[2, 4, 5]]
@test other == _orig_other

# :nearest 
Id = CartesianIndex.([
    (-2, -2),
    (-1, -2),
    (0, -2),
    (1, -2),
    (2, -2),
    (2, -1),
    (2, 0),
    (2, 1),
    (2, 2),
    (1, 2),
    (0, 2),
    (-1, 2),
    (-2, 2),
    (-2, 1),
    (-2, 0),
    (-2, -1)
])
MappedID = CartesianIndex.([
    (-1, -1),
    (-1, -1),
    (0, -1),
    (1, -1),
    (1, -1),
    (1, -1),
    (1, 0),
    (1, 1),
    (1, 1),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1)
])
J = ProfileLikelihood.nearest_node_to_layer.(Id, layer)
@test J == MappedID

Id = CartesianIndex.([
    (-1, -1),
    (0, -1),
    (1, -1),
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0)
])
J = ProfileLikelihood.nearest_node_to_layer.(Id, 1)
@test all(J .== Ref(CartesianIndex(0, 0)))

Id = CartesianIndex.([
    (-2, -2),
    (-1, -2),
    (0, -2),
    (1, -2),
    (2, -2),
    (2, -1),
    (2, 0),
    (2, 1),
    (2, 2),
    (1, 2),
    (0, 2),
    (-1, 2),
    (-2, 2),
    (-2, 1),
    (-2, 0),
    (-2, -1)
])
J = ProfileLikelihood.nearest_node_to_layer.(Id, layer)
for (I, J) in zip(Id, J)
    ProfileLikelihood.set_next_initial_estimate!(sub_cache, other, I, fixed_vals, grid, layer, ProfileLikelihood.exclude_parameter(prob.problem, n); next_initial_estimate_method=:nearest)
    @test sub_cache == other[J]
end
@test other == _orig_other

## Setup the logistic example
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
tspan = extrema(t)
p = (λ, K)
prob = ODEProblem(ode_fnc, u₀, tspan, p)
sol = solve(prob, Rosenbrock23(), saveat=t)
Random.seed!(2828)
uᵒ = sol.u + σ * randn(length(t))
@inline function loglik_fnc2(θ, data, integrator)
    λ, K, u₀ = θ
    uᵒ, σ = data
    integrator.p[1] = λ
    integrator.p[2] = K
    reinit!(integrator, u₀)
    solve!(integrator)
    return gaussian_loglikelihood(uᵒ, integrator.sol.u, σ, length(uᵒ))
end
lb = [0.0, 50.0, 0.0]
ub = [0.05, 150.0, 50.0]
θ₀ = [λ, K, u₀]
syms = [:λ, :K, :u₀]
prob = LikelihoodProblem(
    loglik_fnc2, θ₀, ode_fnc, u₀, maximum(t);
    syms=syms,
    data=(uᵒ, σ),
    ode_parameters=[1.0, 1.0],
    ode_kwargs=(verbose=false, saveat=t),
    f_kwargs=(adtype=Optimization.AutoFiniteDiff(),),
    prob_kwargs=(lb=lb, ub=ub),
    ode_alg=Rosenbrock23()
)
sol = mle(prob, NLopt.LN_BOBYQA)

# Test the :interp method 
n = (1, 2)
mles = get_mle(sol)
res = 40
num_params = 3
normalise = false
ℓmax = get_maximum(sol)
prof, other, cache, sub_cache, fixed_vals = ProfileLikelihood.prepare_cache_vectors(n, mles, res, num_params, normalise, ℓmax)
for (i, I) in enumerate(ProfileLikelihood.LayerIterator(1))
    prof[I] = [-0.002609698982659836
        -0.0017820560535213303
        -0.0027990459433624437
        -0.003224059204491425
        -0.0073355891595099365
        -0.0018454968517858106
        -0.0014532284522275063
        -0.00022646362966582956][i]
    other[I] = [[[8.163357378354423]
        [8.072852109939054]
        [7.739670520045534]
        [7.705262937003664]
        [7.6702606740652515]
        [8.001348326231195]
        [8.091337062998486]
        [8.127585433480208]][i]]
end
_orig_other = deepcopy(other) # for checking aliasing later
lb = get_lower_bounds(prob)[[1, 2]]
ub = get_upper_bounds(prob)[[1, 2]]
centre = mles[[1, 2]]
grid = ProfileLikelihood.FusedRegularGrid(lb, ub, centre, res)
layer = 2
Id = CartesianIndex((2, 2))
fixed_vals = zeros(2)
fixed_vals = [grid[1, Id[1]], grid[2, Id[2]]]
range_1 = get_range(grid, 1, -1, 1)
range_2 = get_range(grid, 2, -1, 1)
interp = extrapolate(interpolate((range_1, range_2), OffsetArray(other[-1:1, -1:1], -1:1, -1:1), Gridded(Linear())), Line())
val = interp(fixed_vals...)
Id = CartesianIndex((2, 2))
ProfileLikelihood.set_next_initial_estimate!(sub_cache, other, Id, fixed_vals, grid, layer, ProfileLikelihood.exclude_parameter(prob.problem, n); next_initial_estimate_method=:interp)
@test sub_cache ≈ val
fixed_vals = [-10.0, 99.0]
ProfileLikelihood.set_next_initial_estimate!(sub_cache, other, Id, fixed_vals, grid, layer, ProfileLikelihood.exclude_parameter(prob.problem, n); next_initial_estimate_method=:interp)
@test sub_cache ≈ mles[[3]]
fixed_vals = [0.01, 2902.0]
ProfileLikelihood.set_next_initial_estimate!(sub_cache, other, Id, fixed_vals, grid, layer, ProfileLikelihood.exclude_parameter(prob.problem, n); next_initial_estimate_method=:interp)
@test sub_cache ≈ mles[[3]]
fixed_vals = [0.12, 22.0]
ProfileLikelihood.set_next_initial_estimate!(sub_cache, other, Id, fixed_vals, grid, layer, ProfileLikelihood.exclude_parameter(prob.problem, n); next_initial_estimate_method=:interp)
@test sub_cache ≈ mles[[3]]
fixed_vals = [0.12, 89.0]
ProfileLikelihood.set_next_initial_estimate!(sub_cache, other, Id, fixed_vals, grid, layer, ProfileLikelihood.exclude_parameter(prob.problem, n); next_initial_estimate_method=:interp)
@test sub_cache ≈ mles[[3]]
fixed_vals = [0.01, 18880.0]
ProfileLikelihood.set_next_initial_estimate!(sub_cache, other, Id, fixed_vals, grid, layer, ProfileLikelihood.exclude_parameter(prob.problem, n); next_initial_estimate_method=:interp)
@test sub_cache ≈ mles[[3]]
fixed_vals = [0.12, 189.0]
ProfileLikelihood.set_next_initial_estimate!(sub_cache, other, Id, fixed_vals, grid, layer, ProfileLikelihood.exclude_parameter(prob.problem, n); next_initial_estimate_method=:interp)
@test sub_cache ≈ mles[[3]]

# Get the results 
@time results = ProfileLikelihood.bivariate_profile(prob, sol, ((1, 2), (3, 1));
    outer_layers=10);
@time results_near = ProfileLikelihood.bivariate_profile(prob, sol, ((1, 2), (3, 1));
    outer_layers=10, next_initial_estimate_method=:nearest);
@time results_int = ProfileLikelihood.bivariate_profile(prob, sol, ((1, 2), (3, 1));
    outer_layers=10, next_initial_estimate_method=:interp);
@time results_par = ProfileLikelihood.bivariate_profile(prob, sol, ((1, 2), (3, 1));
    outer_layers=10, parallel=true);
@time results_near_par = ProfileLikelihood.bivariate_profile(prob, sol, ((1, 2), (3, 1));
    outer_layers=10, next_initial_estimate_method=:nearest, parallel=true);
@time results_int_par = ProfileLikelihood.bivariate_profile(prob, sol, ((1, 2), (3, 1));
    outer_layers=10, next_initial_estimate_method=:interp, parallel=true);

# Test the results
for results in (results, results_near, results_int, results_par, results_near_par, results_int_par)
    @test ProfileLikelihood.get_parameter_values(results) == results.parameter_values
    @test ProfileLikelihood.get_parameter_values(results, 1, 2) == results.parameter_values[(1, 2)]
    @test ProfileLikelihood.get_parameter_values(results, :λ, :K) == results.parameter_values[(1, 2)]
    @test ProfileLikelihood.get_parameter_values(results, 3, 1) == results.parameter_values[(3, 1)]
    @test ProfileLikelihood.get_parameter_values(results, :u₀, :λ) == results.parameter_values[(3, 1)]
    @test ProfileLikelihood.get_parameter_values(results, 1, 2, 1) == results.parameter_values[(1, 2)][1]
    @test ProfileLikelihood.get_parameter_values(results, 1, 2, 2) == results.parameter_values[(1, 2)][2]
    @test ProfileLikelihood.get_parameter_values(results, 1, 2, :λ) == results.parameter_values[(1, 2)][1]
    @test ProfileLikelihood.get_parameter_values(results, 1, 2, :K) == results.parameter_values[(1, 2)][2]
    @test ProfileLikelihood.get_parameter_values(results, 3, 1, 1) == results.parameter_values[(3, 1)][1]
    @test ProfileLikelihood.get_parameter_values(results, 3, 1, 2) == results.parameter_values[(3, 1)][2]
    @test ProfileLikelihood.get_parameter_values(results, 3, 1, :u₀) == results.parameter_values[(3, 1)][1]
    @test ProfileLikelihood.get_parameter_values(results, 3, 1, :λ) == results.parameter_values[(3, 1)][2]
    @test ProfileLikelihood.get_parameter_values(results, :λ, :K, 1) == results.parameter_values[(1, 2)][1]
    @test ProfileLikelihood.get_parameter_values(results, :λ, :K, 2) == results.parameter_values[(1, 2)][2]
    @test ProfileLikelihood.get_parameter_values(results, :λ, :K, :λ) == results.parameter_values[(1, 2)][1]
    @test ProfileLikelihood.get_parameter_values(results, :λ, :K, :K) == results.parameter_values[(1, 2)][2]
    @test ProfileLikelihood.get_parameter_values(results, :u₀, :λ, 1) == results.parameter_values[(3, 1)][1]
    @test ProfileLikelihood.get_parameter_values(results, :u₀, :λ, 2) == results.parameter_values[(3, 1)][2]
    @test ProfileLikelihood.get_parameter_values(results, :u₀, :λ, :u₀) == results.parameter_values[(3, 1)][1]
    @test ProfileLikelihood.get_parameter_values(results, :u₀, :λ, :λ) == results.parameter_values[(3, 1)][2]
    @test ProfileLikelihood.get_parameter_values(results, 1, 2, 1, 5) == results.parameter_values[(1, 2)][1][5]
    @test ProfileLikelihood.get_parameter_values(results, 1, 2, 2, 7) == results.parameter_values[(1, 2)][2][7]
    @test ProfileLikelihood.get_parameter_values(results, 1, 2, :λ, 4) == results.parameter_values[(1, 2)][1][4]
    @test ProfileLikelihood.get_parameter_values(results, 1, 2, :K, 6) == results.parameter_values[(1, 2)][2][6]
    @test ProfileLikelihood.get_parameter_values(results, 3, 1, 1, 2) == results.parameter_values[(3, 1)][1][2]
    @test ProfileLikelihood.get_parameter_values(results, 3, 1, 2, 3) == results.parameter_values[(3, 1)][2][3]
    @test ProfileLikelihood.get_parameter_values(results, 3, 1, :u₀, -3) == results.parameter_values[(3, 1)][1][-3]
    @test ProfileLikelihood.get_parameter_values(results, 3, 1, :λ, 0) == results.parameter_values[(3, 1)][2][0]
    @test ProfileLikelihood.get_parameter_values(results, :λ, :K, 1, 2) == results.parameter_values[(1, 2)][1][2]
    @test ProfileLikelihood.get_parameter_values(results, :λ, :K, 2, 1) == results.parameter_values[(1, 2)][2][1]
    @test ProfileLikelihood.get_parameter_values(results, :λ, :K, :λ, -5) == results.parameter_values[(1, 2)][1][-5]
    @test ProfileLikelihood.get_parameter_values(results, :λ, :K, :K, 0) == results.parameter_values[(1, 2)][2][0]
    @test ProfileLikelihood.get_parameter_values(results, :u₀, :λ, 1, 7) == results.parameter_values[(3, 1)][1][7]
    @test ProfileLikelihood.get_parameter_values(results, :u₀, :λ, 2, 5) == results.parameter_values[(3, 1)][2][5]
    @test ProfileLikelihood.get_parameter_values(results, :u₀, :λ, :u₀, -2) == results.parameter_values[(3, 1)][1][-2]
    @test ProfileLikelihood.get_parameter_values(results, :u₀, :λ, :λ, -2) == results.parameter_values[(3, 1)][2][-2]
    @test ProfileLikelihood.get_profile_values(results) == results.profile_values
    @test ProfileLikelihood.get_profile_values(results, 1, 2) == results.profile_values[(1, 2)]
    @test ProfileLikelihood.get_profile_values(results, :λ, :K) == results.profile_values[(1, 2)]
    @test ProfileLikelihood.get_profile_values(results, 3, 1) == results.profile_values[(3, 1)]
    @test ProfileLikelihood.get_profile_values(results, :u₀, :λ) == results.profile_values[(3, 1)]
    @test ProfileLikelihood.get_likelihood_problem(results) == prob
    @test ProfileLikelihood.get_likelihood_solution(results) == sol
    @test ProfileLikelihood.get_interpolants(results) == results.interpolants
    @test ProfileLikelihood.get_interpolants(results, 1, 2) == results.interpolants[(1, 2)]
    @test ProfileLikelihood.get_interpolants(results, :λ, :K) == results.interpolants[(1, 2)]
    @test ProfileLikelihood.get_interpolants(results, 3, 1) == results.interpolants[(3, 1)]
    @test ProfileLikelihood.get_interpolants(results, :u₀, :λ) == results.interpolants[(3, 1)]
    @test ProfileLikelihood.get_confidence_regions(results) == results.confidence_regions
    @test ProfileLikelihood.get_confidence_regions(results, 1, 2) == results.confidence_regions[(1, 2)]
    @test ProfileLikelihood.get_confidence_regions(results, :λ, :K) == results.confidence_regions[(1, 2)]
    @test ProfileLikelihood.get_confidence_regions(results, 3, 1) == results.confidence_regions[(3, 1)]
    @test ProfileLikelihood.get_confidence_regions(results, :u₀, :λ) == results.confidence_regions[(3, 1)]
    @test ProfileLikelihood.get_other_mles(results) == results.other_mles
    @test ProfileLikelihood.get_other_mles(results, 1, 2) == results.other_mles[(1, 2)]
    @test ProfileLikelihood.get_other_mles(results, :λ, :K) == results.other_mles[(1, 2)]
    @test ProfileLikelihood.get_other_mles(results, 3, 1) == results.other_mles[(3, 1)]
    @test ProfileLikelihood.get_other_mles(results, :u₀, :λ) == results.other_mles[(3, 1)]
    @test ProfileLikelihood.get_syms(results) == [:λ, :K, :u₀]
    @test ProfileLikelihood.get_syms(results, 1, 2) == (:λ, :K)
    @test ProfileLikelihood.get_syms(results, 3, 1) == (:u₀, :λ)
    @test ProfileLikelihood.profiled_parameters(results) == [(1, 2), (3, 1)]
    @test ProfileLikelihood.number_of_profiled_parameters(results) == 2
    @test ProfileLikelihood.number_of_layers(results, 1, 2) == 102
    @test ProfileLikelihood.number_of_layers(results, 3, 1) == 188
    bbox = ProfileLikelihood.get_bounding_box(results, 1, 2)
    @test all(bbox .≈ (0.005641669055546597, 0.020890639625718074, 88.73495146230951, 112.50617436930195))
    bbox = ProfileLikelihood.get_bounding_box(results, 3, 1)
    @test all(bbox .≈ (0.8585793458749309, 22.24256937316934, 0.005641851160910081, 0.02088915793720584))
    CR = ProfileLikelihood.get_confidence_regions(results, 1, 2)
    conf_x = CR.x
    conf_y = CR.y
    xy = [(x, y) for (x, y) in zip(conf_x, conf_y)]
    A = DelaunayTriangulation.area(xy)
    @test A ≈ 0.23243592745692931
    CR = ProfileLikelihood.get_confidence_regions(results, 3, 1)
    conf_x = CR.x
    conf_y = CR.y
    xy = [(x, y) for (x, y) in zip(conf_x, conf_y)]
    A = DelaunayTriangulation.area(xy)
    @test A ≈ 0.09636388019922583
    @test_throws BoundsError results[1, 3]
    @test length(results.parameter_values[(1, 2)][1]) == 205
    @test length(results.parameter_values[(1, 2)][2]) == 205
    @test length(results.profile_values[(1, 2)]) == 205^2
    @test length(results.other_mles[(1, 2)]) == 205^2
    @test length(results.parameter_values[(1, 2)][1]) == 205
    @test length(results.parameter_values[(1, 2)][2]) == 205
    @test length(results.profile_values[(1, 2)]) == 205^2
    @test length(results.other_mles[(1, 2)]) == 205^2
    @test length(results.parameter_values[(3, 1)][1]) == 377
    @test length(results.parameter_values[(3, 1)][2]) == 377
    @test length(results.profile_values[(3, 1)]) == 377^2
    @test length(results.other_mles[(3, 1)]) == 377^2
    @test length(results.parameter_values[(3, 1)][1]) == 377
    @test length(results.parameter_values[(3, 1)][2]) == 377
    @test length(results.profile_values[(3, 1)]) == 377^2
    @test length(results.other_mles[(3, 1)]) == 377^2
    prof = results[1, 2]
    @test ProfileLikelihood.get_parent(prof) === results
    @test ProfileLikelihood.get_index(prof) == (1, 2)
    @test get_parameter_values(prof) == results.parameter_values[(1, 2)]
    @test get_parameter_values(prof, 1) == results.parameter_values[(1, 2)][1]
    @test get_parameter_values(prof, 2) == results.parameter_values[(1, 2)][2]
    @test get_parameter_values(prof, 1, 0) == results.parameter_values[(1, 2)][1][0]
    @test get_parameter_values(prof, 2, -5) == results.parameter_values[(1, 2)][2][-5]
    @test get_parameter_values(prof, :λ) == results.parameter_values[(1, 2)][1]
    @test get_parameter_values(prof, :K, 4) == results.parameter_values[(1, 2)][2][4]
    @test get_profile_values(prof) == results.profile_values[(1, 2)]
    @test get_profile_values(prof, 7, 4) == results.profile_values[(1, 2)][7, 4]
    @test ProfileLikelihood.get_likelihood_problem(prof) == prob
    @test ProfileLikelihood.get_likelihood_solution(prof) == sol
    @test ProfileLikelihood.get_interpolants(prof) == results.interpolants[(1, 2)]
    @test ProfileLikelihood.get_confidence_regions(prof) == results.confidence_regions[(1, 2)]
    @test ProfileLikelihood.get_other_mles(prof) == results.other_mles[(1, 2)]
    @test ProfileLikelihood.get_other_mles(prof, 0, -4) == results.other_mles[(1, 2)][0, -4]
    @test ProfileLikelihood.number_of_layers(prof) == 102
    bbox = ProfileLikelihood.get_bounding_box(prof)
    @test all(bbox .≈ (0.005641669055546597, 0.020890639625718074, 88.73495146230951, 112.50617436930195))

    # Test the confidence intervals 
    λgrid = results.parameter_values[(1, 2)][1].parent
    Kgrid = results.parameter_values[(1, 2)][2].parent
    ℓvals = results.profile_values[(1, 2)].parent
    conf_x = results.confidence_regions[(1, 2)].x
    conf_y = results.confidence_regions[(1, 2)].y
    λK_pts = [(λgrid[i], Kgrid[j]) for i in eachindex(λgrid), j in eachindex(Kgrid)]
    Idx = CartesianIndices(λK_pts)
    λK_vec = vec(λK_pts)
    nodes = [conf_x conf_y]
    edges = [vec(1:length(conf_x)) [(2:length(conf_x))..., 1]]
    tol = 1e-1
    res = inpoly2(λK_vec, nodes, edges, atol=tol)
    for i in axes(res, 1)
        mat_idx = Idx[i]
        inside_ci = res[i, 1]
        if inside_ci
            @test ℓvals[mat_idx] > ProfileLikelihood.get_chisq_threshold(0.95, 2)
        else
            @test ℓvals[mat_idx] < ProfileLikelihood.get_chisq_threshold(0.95, 2)
        end
    end

    u₀grid = results.parameter_values[(3, 1)][1].parent
    λgrid = results.parameter_values[(3, 1)][2].parent
    ℓvals = results.profile_values[(3, 1)].parent
    conf_x = results.confidence_regions[(3, 1)].x
    conf_y = results.confidence_regions[(3, 1)].y
    u₀λ_pts = [(u₀grid[i], λgrid[j]) for i in eachindex(u₀grid), j in eachindex(λgrid)]
    Idx = CartesianIndices(u₀λ_pts)
    u₀λ_vec = vec(u₀λ_pts)
    nodes = [conf_x conf_y]
    edges = [vec(1:length(conf_x)) [(2:length(conf_x))..., 1]]
    tol = 1e-1
    res = inpoly2(u₀λ_vec, nodes, edges, atol=tol)
    for i in axes(res, 1)
        mat_idx = Idx[i]
        inside_ci = res[i, 1]
        if inside_ci
            @test ℓvals[mat_idx] > ProfileLikelihood.get_chisq_threshold(0.95, 2)
        else
            @test ℓvals[mat_idx] < ProfileLikelihood.get_chisq_threshold(0.95, 2)
        end
    end

    # Test the interpolants 
    interp = ProfileLikelihood.get_interpolants(results, :λ, :K)
    @test interp(sol[1], sol[2]) ≈ 0.0
    for i in axes(results.profile_values[(1, 2)], 1)
        for j in axes(results.profile_values[(1, 2)], 2)
            λval = results.parameter_values[(1, 2)][1][i]
            Kval = results.parameter_values[(1, 2)][2][j]
            @test interp(λval, Kval) ≈ results.profile_values[(1, 2)][i, j]
            @test results[1, 2](λval, Kval) ≈ results.profile_values[(1, 2)][i, j]
            @test results(λval, Kval, 1, 2) ≈ results.profile_values[(1, 2)][i, j]
            @test results(λval, Kval, :λ, :K) ≈ results.profile_values[(1, 2)][i, j]
            @test results[:λ, :K](λval, Kval) ≈ results.profile_values[(1, 2)][i, j]
        end
    end
    conf_x = results.confidence_regions[(1, 2)].x
    conf_y = results.confidence_regions[(1, 2)].y
    for (x, y) in zip(conf_x, conf_y)
        @test interp(x, y) ≈ ProfileLikelihood.get_chisq_threshold(0.95, 2)
        @test results(x, y, 1, 2) ≈ ProfileLikelihood.get_chisq_threshold(0.95, 2)
    end

    interp = ProfileLikelihood.get_interpolants(results, :u₀, :λ)
    @test interp(sol[:u₀], sol[1]) ≈ 0.0
    for i in axes(results.profile_values[(3, 1)], 1)
        for j in axes(results.profile_values[(3, 1)], 2)
            u₀val = results.parameter_values[(3, 1)][1][i]
            λval = results.parameter_values[(3, 1)][2][j]
            @test interp(u₀val, λval) ≈ results.profile_values[(3, 1)][i, j]
            @test results[3, 1](u₀val, λval) ≈ results.profile_values[(3, 1)][i, j]
            @test results(u₀val, λval, 3, 1) ≈ results.profile_values[(3, 1)][i, j]
            @test results(u₀val, λval, :u₀, :λ) ≈ results.profile_values[(3, 1)][i, j]
            @test results[:u₀, :λ](u₀val, λval) ≈ results.profile_values[(3, 1)][i, j]
        end
    end
    conf_x = results.confidence_regions[(3, 1)].x
    conf_y = results.confidence_regions[(3, 1)].y
    for (x, y) in zip(conf_x, conf_y)
        @test interp(x, y) ≈ ProfileLikelihood.get_chisq_threshold(0.95, 2)
        @test results(x, y, 3, 1) ≈ ProfileLikelihood.get_chisq_threshold(0.95, 2)
    end
end

# Test the parallelism 
for (serial_res, parallel_res) in ((results, results_par), (results_near, results_near_par), (results_int, results_int_par))
    CR = serial_res.confidence_regions
    interp = serial_res.interpolants
    prob = serial_res.likelihood_problem
    sol = serial_res.likelihood_solution
    other = serial_res.other_mles
    params = serial_res.parameter_values
    profiles = serial_res.profile_values

    CR_par = parallel_res.confidence_regions
    interp_par = parallel_res.interpolants
    prob_par = parallel_res.likelihood_problem
    sol_par = parallel_res.likelihood_solution
    other_par = parallel_res.other_mles
    params_par = parallel_res.parameter_values
    profiles_par = parallel_res.profile_values

    @test CR[(1, 2)].x ≈ CR_par[(1, 2)].x
    @test CR[(3, 1)].x ≈ CR_par[(3, 1)].x
    @test CR[(1, 2)].y ≈ CR_par[(1, 2)].y
    @test CR[(3, 1)].y ≈ CR_par[(3, 1)].y
    @test CR[(1, 2)].level ≈ CR_par[(1, 2)].level
    @test CR[(3, 1)].level ≈ CR_par[(3, 1)].level
    @test interp[(1, 2)] ≈ interp_par[(1, 2)]
    @test interp_par[(3, 1)] ≈ interp_par[(3, 1)]
    @test prob === prob_par
    @test sol === sol_par
    @test other[(1, 2)] ≈ other_par[(1, 2)]
    @test other[(3, 1)] ≈ other_par[(3, 1)]
    @test params[(1, 2)][1] ≈ params_par[(1, 2)][1]
    @test params[(1, 2)][2] ≈ params_par[(1, 2)][2]
    @test params[(3, 1)][1] ≈ params_par[(3, 1)][1]
    @test params[(3, 1)][2] ≈ params_par[(3, 1)][2]
    @test profiles[(1, 2)] ≈ profiles_par[(1, 2)]
    @test profiles[(3, 1)] ≈ profiles_par[(3, 1)]
end

# Test the next_initial_estimate_method methods with the results
for (n1, n2, n3) in ((1, 2, 3), (3, 1, 2))
    n1, n2, n3 = 1, 2, 3
    layers = ProfileLikelihood.number_of_layers(results, n1, n2)
    _prob = ProfileLikelihood.exclude_parameter(prob.problem, (n1, n2))
    sub_cache1 = deepcopy(results.other_mles[(n1, n2)])
    sub_cache2 = deepcopy(results.other_mles[(n1, n2)])
    sub_cache3 = deepcopy(results.other_mles[(n1, n2)])
    grids = ProfileLikelihood.construct_profile_grids(((1, 2), (3, 1)), sol, get_lower_bounds(prob), get_upper_bounds(prob), 200)
    for layer in 1:layers
        for I in ProfileLikelihood.LayerIterator(layer)
            fixed_vals = [grids[(n1, n2)][1, I[1]], grids[(n1, n2)][2, I[2]]]
            @views ProfileLikelihood.set_next_initial_estimate!(sub_cache1[I], results.other_mles[(n1, n2)], I, fixed_vals, grids[(n1, n2)], layer, _prob; next_initial_estimate_method=:mle)
            @views ProfileLikelihood.set_next_initial_estimate!(sub_cache2[I], results.other_mles[(n1, n2)], I, fixed_vals, grids[(n1, n2)], layer, _prob; next_initial_estimate_method=:nearest)
            @views ProfileLikelihood.set_next_initial_estimate!(sub_cache3[I], results.other_mles[(n1, n2)], I, fixed_vals, grids[(n1, n2)], layer, _prob; next_initial_estimate_method=:interp)
            @test sub_cache1[I] == [mles[n3]]
            J = ProfileLikelihood.nearest_node_to_layer(I, layer)
            @test sub_cache2[I] == results.other_mles[(n1, n2)][J] == ProfileLikelihood.get_other_mles(results[n1, n2], J[1], J[2])
            if layer > 1
                range_1 = get_range(grids[(n1, n2)], 1, -layer + 1, layer - 1).parent
                range_2 = get_range(grids[(n1, n2)], 2, -layer + 1, layer - 1).parent
                interp = extrapolate(interpolate((range_1, range_2), results.other_mles[(n1, n2)][(-layer+1):(layer-1), (-layer+1):(layer-1)], Gridded(Linear())), Line())
                val = interp(fixed_vals...)
                if (n1, n2) == (1, 2) && (val[1] < prob.problem.lb[3] || val[1] > prob.problem.ub[3])
                    @test sub_cache3[I] == [mles[3]]
                elseif (n1, n2) == (3, 1) && (val[1] < prob.problem.lb[2] || val[1] > prob.problem.ub[2])
                    @test sub_cache3[I] == [mles[2]]
                else
                    @test sub_cache3[I] == val
                end
            else
                @test sub_cache3[I] == [mles[n3]]
            end
        end
    end
    err1 = (sub_cache1 .- results.other_mles[(n1, n2)]) ./ results.other_mles[(n1, n2)]
    err1 = getindex.(err1, 1).parent
    _err1 = [(mles[n3] .- results.other_mles[(n1, n2)][i, j]) ./ results.other_mles[(n1, n2)][i, j] for i in -layers:layers, j in -layers:layers]
    _err1 = getindex.(_err1, 1)
    @test err1 ≈ _err1
    err2 = (sub_cache2 .- results.other_mles[(n1, n2)]) ./ results.other_mles[(n1, n2)]
    err2 = getindex.(err2, 1).parent
    @test all(abs.(err2) .< 5)
    @test median(abs.(err2)) < 0.01
    err3 = (sub_cache3 .- results.other_mles[(n1, n2)]) ./ results.other_mles[(n1, n2)]
    err3 = getindex.(err3, 1).parent
    @test median(abs.(err3)) < 0.001
end











fig = Figure()
ax = Axis(fig[1, 1])
contourf!(ax, results.parameter_values[(1, 2)][1].parent, results.parameter_values[(1, 2)][2].parent, results.profile_values[(1, 2)].parent, levels=50)
#heatmap!(ax, results.parameter_values[(1, 2)][1].parent, results.parameter_values[(1, 2)][2].parent, results.profile_values[(1, 2)].parent)
lines!(ax, results.confidence_regions[(1, 2)].x, results.confidence_regions[(1, 2)].y, linestyle=:solid)
ylims!(ax, 60, 140)
xlims!(ax, 0, 0.04)
fig
#lines!(ax, [bbox[1], bbox[2], bbox[2], bbox[1], bbox[1]], [bbox[3], bbox[3], bbox[4], bbox[4], bbox[3]])

interp = results.interpolants[(1, 2)]
gridx = LinRange(0, 0.04, 500)
gridy = LinRange(60, 140, 500)
z = [interp(x, y) for x in gridx, y in gridy]
ax = Axis(fig[1, 2])
contourf!(ax, gridx, gridy, z, levels=50)
ylims!(ax, 60, 140)
xlims!(ax, 0, 0.04)
lines!(ax, results.confidence_regions[(1, 2)].x, results.confidence_regions[(1, 2)].y, linestyle=:solid)
fig


gridx = results.parameter_values[(1, 2)][1]
gridy = results.parameter_values[(1, 2)][2]
z = results.other_mles[(1, 2)]
interp = interpolate((gridx, gridy), z, Gridded(Linear()))
interp(gridx[0], gridy[0])


