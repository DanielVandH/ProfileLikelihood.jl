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
    @test grids[n[i]].negative_grid.resolution == max(res[u], res[v])
end

n = ((1, 2), (5, 1), (2, 3), (2, 1), (5, 3))
lb = [1e-12, -12.0, -3.3, -5.7, -11.0]
ub = [10.0, 12.0, 8.3, 9.3, 10.0]
res = [(107, 43), (282, 103), (587, 203), (501), (10101, 100)]
grids = construct_profile_grids(n, sol, lb, ub, res)
for i in eachindex(n)
    u, v = n[i]
    for j in 1:2
        @test get_range(grids[(u, v)], j) == get_range(ProfileLikelihood.FusedRegularGrid(lb[[u, v]], ub[[u, v]], [sol[u], sol[v]], maximum(res[[u, v]])), j)
    end
end





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

##
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


## Extract the problem and solution 
opt_prob, mles, ℓmax = ProfileLikelihood.extract_problem_and_solution(prob, sol)

## Prepare the profile results 
n = ((1, 3), (2, 3))
N = length(n)
T = ProfileLikelihood.number_type(mles)
resolution = 200
param_ranges = construct_profile_ranges(n, sol, get_lower_bounds(prob), get_upper_bounds(prob), resolution)

