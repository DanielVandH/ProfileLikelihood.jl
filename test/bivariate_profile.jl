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
using Surrogates
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
n = (1, 3)
mles = [3.0, 4.0, 5.5, 6.6, 8.322]
res = 40
num_params = 5
normalise = true
ℓmax = 0.5993
prof, other, cache, sub_cache, fixed_vals = ProfileLikelihood.prepare_cache_vectors(n, mles, res, num_params, normalise, ℓmax)
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

n = (1, 3)
mles = [3.0, 4.0, 5.5, 6.6, 8.322]
res = 40
num_params = 5
normalise = false
ℓmax = 0.5993
prof, other, cache, sub_cache, fixed_vals = ProfileLikelihood.prepare_cache_vectors(n, mles, res, num_params, normalise, ℓmax)
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

# :mle
next_initial_estimate_method = :mle
ProfileLikelihood.set_next_initial_estimate!(sub_cache, other, Id, fixed_vals, grid, layer; next_initial_estimate_method=next_initial_estimate_method)
@test sub_cache == mles[[2, 4, 5]]
@test other == _orig_other

# nearest 
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
J = ProfileLikelihood.mapped_layer_node.(Id, layer)
@test J == MappedID

Id = CartesianIndex.([
    (-1,-1),
    (0,-1),
    (1,-1),
    (1,0),
    (1,1),
    (0,1),
    (-1,1),
    (-1,0)
])
J = ProfileLikelihood.mapped_layer_node.(Id, 1)
@test all(J .== Ref(CartesianIndex(0,0)))

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
J = ProfileLikelihood.mapped_layer_node.(Id, layer)
for (I, J) in zip(Id, J)
    ProfileLikelihood.set_next_initial_estimate!(sub_cache,other,I,fixed_vals,grid,layer;next_initial_estimate_method=:nearest)
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

# Get the results 
@time results = ProfileLikelihood.bivariate_profile(prob, sol, ((1, 2), (3, 1));
    outer_layers=10);

# Test the getters 
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

fig = Figure()
ax = Axis(fig[1, 1])
contourf!(ax, results.parameter_values[(3, 1)][2].parent, results.parameter_values[(3, 1)][1].parent, results.profile_values[(3, 1)].parent', levels=50)
#heatmap!(ax, results.parameter_values[(1, 2)][1].parent, results.parameter_values[(1, 2)][2].parent, results.profile_values[(1, 2)].parent)
lines!(ax, results.confidence_regions[(3, 1)].y, results.confidence_regions[(3, 1)].x, linestyle=:solid)
fig
#lines!(ax, [bbox[1], bbox[2], bbox[2], bbox[1], bbox[1]], [bbox[3], bbox[3], bbox[4], bbox[4], bbox[3]])


