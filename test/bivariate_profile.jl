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


#=
Base.@kwdef struct ___FusedRegularGrid{PG,GR,C}
    positive_grid::PG
    negative_grid::GR
    centre::C
end
Base.@kwdef struct ___ConfidenceRegion{T,F}
    lower::T
    upper::T
    level::F
end
Base.@kwdef struct ___BivariateProfileLikelihoodSolution{I,V,LP,LS,Spl,CT,CF,OM}
    parameter_values::Dict{I,V}
    profile_values::Dict{I,V}
    likelihood_problem::LP
    likelihood_solution::LS
    interpolants::Dict{I,Spl}
    confidence_regions::Dict{I,___ConfidenceRegion{CT,CF}}
    other_mles::OM
end
=#

normalise = true
n = ((1, 2),)
res = [50, 50, 50]
grids = construct_profile_grids(n, sol, lb, ub, res)
alg = NLopt.LN_NELDERMEAD
conf_level = 0.95
threshold = ProfileLikelihood.get_chisq_threshold(conf_level, 2)

## Extract the problem and solution 
opt_prob, mles, ℓmax = ProfileLikelihood.extract_problem_and_solution(prob, sol)

## Normalise the objective function 
shifted_opt_prob = ProfileLikelihood.normalise_objective_function(opt_prob, ℓmax, normalise)

num_params = ProfileLikelihood.number_of_parameters(shifted_opt_prob)

## Pair to profile 
_n = n[1]
grid = grids[_n]
res = grid.resolutions

## Prepare the cache vectors
profile_vals = OffsetArray(Matrix{Float64}(undef, 2res + 1, 2res + 1), -res:res, -res:res)
other_mles = OffsetArray(Matrix{Vector{Float64}}(undef, 2res + 1, 2res + 1), -res:res, -res:res)
cache = DiffCache(zeros(Float64, num_params))
sub_cache = zeros(Float64, num_params - 2)
sub_cache .= mles[Not(_n[1], _n[2])]
fixed_vals = zeros(Float64, 2)
profile_vals[0, 0] = normalise ? 0.0 : ℓmax
other_mles[0, 0] = mles[Not(_n[1], _n[2])]

## Now restrict the problem 
restricted_prob = ProfileLikelihood.exclude_parameter(shifted_opt_prob, _n)

## Solve outwards
layer = 1
final_layer = res
for i in 1:res
    layer_iterator = ProfileLikelihood.LayerIterator(layer)
    any_above_threshold = false
    for I in layer_iterator
        ProfileLikelihood.get_parameters!(fixed_vals, grid, I)
        fixed_prob = ProfileLikelihood.construct_fixed_optimisation_function(restricted_prob, _n, fixed_vals, cache)
        fixed_prob.u0 .= mles[Not(_n[1], _n[2])]
        soln = solve(fixed_prob, alg)
        profile_vals[I] = -soln.objective - ℓmax * !normalise
        other_mles[I] = soln.u
        if !any_above_threshold && profile_vals[I] > threshold
            any_above_threshold = true
        end
    end
    if !any_above_threshold
        global final_layer
        final_layer = layer
        break
    end
    layer += 1
end

## Resize the arrays 
profile_vals = OffsetArray(profile_vals[-final_layer:final_layer, -final_layer:final_layer], -final_layer:final_layer, -final_layer:final_layer)
other_mles = OffsetArray(other_mles[-final_layer:final_layer, -final_layer:final_layer], -final_layer:final_layer, -final_layer:final_layer)
param_1_range = get_range(grid, 1, -final_layer, final_layer)
param_2_range = get_range(grid, 2, -final_layer, final_layer)

## Make the contour 
c = Contour.contour(param_1_range, param_2_range, profile_vals, threshold)
all_coords = reduce(vcat, [reduce(hcat, coordinates(xy)) for xy in Contour.lines(c)])
region_x = all_coords[:, 1]
region_y = all_coords[:, 2]
region = ProfileLikelihood.ConfidenceRegion(region_x, region_y, conf_level)

## Make an interpolant
interpolant = Interpolations.interpolate((param_1_range, param_2_range), profile_vals, Gridded(Linear()))
