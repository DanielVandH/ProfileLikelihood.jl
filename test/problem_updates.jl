using ..ProfileLikelihood
using Optimization 
using OptimizationNLopt
using PreallocationTools
using InvertedIndices
######################################################
## Problem updates
######################################################
## Test that we can correctly update the initial estimate 
loglik = (θ, p) -> θ[1] * p[1] + θ[2]
negloglik = ProfileLikelihood.negate_loglik(loglik)
θ₀ = rand(3)
data = (rand(100), [:a, :b])
prob = ProfileLikelihood.construct_optimisation_problem(negloglik, θ₀, data)
θ₁ = [0.0, 1.0, 0.0]
new_prob = ProfileLikelihood.update_initial_estimate(prob, θ₁)
@test new_prob.u0 == θ₁
sol = solve(prob, Opt(:LN_NELDERMEAD, 3))
new_prob_2 = ProfileLikelihood.update_initial_estimate(prob, sol)
@test new_prob_2.u0 == sol.u
@test prob.u0 == θ₀ # check aliasing

## Test that we are correctly reProfileLikelihoodacing the objective function 
loglik = (θ, p) -> θ[1] * p[1] + θ[2]
negloglik = ProfileLikelihood.negate_loglik(loglik)
θ₀ = rand(3)
data = (rand(100), [:a, :b])
prob = ProfileLikelihood.construct_optimisation_problem(negloglik, θ₀, data)
new_obj = (θ, p) -> θ[1] * p[1] + θ[2] + p[2]
new_prob = ProfileLikelihood.replace_objective_function(prob, new_obj)
@test collect(typeof(new_prob).parameters)[Not(2)] == collect(typeof(prob).parameters)[Not(2)]
@test collect(typeof(new_prob.f).parameters)[Not(3)] == collect(typeof(prob.f).parameters)[Not(3)]
@test new_prob.f.f == new_obj
@inferred ProfileLikelihood.replace_objective_function(prob, new_obj)

## Test that we are correctly constructing a fixed function 
loglik = (θ, p) -> θ[1] * p[1] + θ[2]
negloglik = ProfileLikelihood.negate_loglik(loglik)
θ₀ = rand(2)
data = 1.301
prob = ProfileLikelihood.construct_optimisation_problem(negloglik, θ₀, data)
val = 1.0
n = 1
cache = DiffCache(θ₀)
new_f = ProfileLikelihood.construct_fixed_optimisation_function(prob, n, val, cache)
@inferred ProfileLikelihood.construct_fixed_optimisation_function(prob, n, val, cache)
@test new_f.f([2.31], data) ≈ negloglik([val, 2.31], data)
@inferred new_f.f([2.31], data)
cache = [1.0, 2.0]
new_f = ProfileLikelihood.construct_fixed_optimisation_function(prob, n, val, cache)
@inferred ProfileLikelihood.construct_fixed_optimisation_function(prob, n, val, cache)

val = 2.39291
n = 2
cache = DiffCache(θ₀)
new_f = ProfileLikelihood.construct_fixed_optimisation_function(prob, n, val, cache)
@inferred ProfileLikelihood.construct_fixed_optimisation_function(prob, n, val, cache)
@test new_f.f([2.31], data) ≈ negloglik([2.31, val], data)
@inferred new_f.f([2.31], data)
cache = [1.0, 2.0]
new_f = ProfileLikelihood.construct_fixed_optimisation_function(prob, n, val, cache)
@inferred ProfileLikelihood.construct_fixed_optimisation_function(prob, n, val, cache)
@test new_f.f([2.31], data) ≈ negloglik([2.31, val], data)
@inferred new_f.f([2.31], data)

loglik = (θ, p) -> θ[1] * p[1] + θ[2] + θ[3] + θ[4]
negloglik = ProfileLikelihood.negate_loglik(loglik)
θ₀ = rand(4)
data = 1.301
prob = ProfileLikelihood.construct_optimisation_problem(negloglik, θ₀, data)
val = 1.0
n = 3
cache = DiffCache(θ₀)
new_f = ProfileLikelihood.construct_fixed_optimisation_function(prob, n, val, cache)
@inferred ProfileLikelihood.construct_fixed_optimisation_function(prob, n, val, cache)
@test new_f.f([2.31, 4.7, 2.3], data) ≈ negloglik([2.31, 4.7, val, 2.3], data)
@inferred new_f.f([2.31, 4.7, 2.3], data)
cache = [1.0, 2.0, 3.0, 4.0]
new_f = ProfileLikelihood.construct_fixed_optimisation_function(prob, n, val, cache)
@inferred ProfileLikelihood.construct_fixed_optimisation_function(prob, n, val, cache)

#@benchmark $new_f.f($[2.31, 4.7, 2.3], $data)


## Test that we can correctly exclude a parameter
loglik = (θ, p) -> θ[1] * p[1] + θ[2]
negloglik = ProfileLikelihood.negate_loglik(loglik)
θ₀ = rand(2)
data = [1.0]
prob = ProfileLikelihood.construct_optimisation_problem(negloglik, θ₀, data)
n = 1
new_prob = ProfileLikelihood.exclude_parameter(prob, n)
@inferred ProfileLikelihood.exclude_parameter(prob, n)
@test !ProfileLikelihood.has_bounds(new_prob)
@test new_prob.u0 == [θ₀[2]]

n = 2
new_prob = ProfileLikelihood.exclude_parameter(prob, n)
@inferred ProfileLikelihood.exclude_parameter(prob, n)
@test !ProfileLikelihood.has_bounds(new_prob)
@test new_prob.u0 == [θ₀[1]]

prob = ProfileLikelihood.construct_optimisation_problem(negloglik, θ₀, data; lb=[1.0, 2.0], ub=[10.0, 20.0])
n = 1
new_prob = ProfileLikelihood.exclude_parameter(prob, n)
@inferred ProfileLikelihood.exclude_parameter(prob, n)
@test ProfileLikelihood.has_bounds(new_prob)
@test ProfileLikelihood.get_lower_bounds(new_prob) == [2.0]
@test ProfileLikelihood.get_upper_bounds(new_prob) == [20.0]
@test new_prob.u0 == [θ₀[2]]

n = 2
new_prob = ProfileLikelihood.exclude_parameter(prob, n)
@inferred ProfileLikelihood.exclude_parameter(prob, n)
@test ProfileLikelihood.has_bounds(new_prob)
@test ProfileLikelihood.get_lower_bounds(new_prob) == [1.0]
@test ProfileLikelihood.get_upper_bounds(new_prob) == [10.0]
@test new_prob.u0 == [θ₀[1]]

## Test that we can shift the objective function 
loglik = (θ, p) -> θ[1] * p[1] + θ[2]
negloglik = ProfileLikelihood.negate_loglik(loglik)
θ₀ = rand(2)
data = [1.0]
prob = ProfileLikelihood.construct_optimisation_problem(negloglik, θ₀, data)
new_prob = ProfileLikelihood.shift_objective_function(prob, 0.2291)
@inferred ProfileLikelihood.shift_objective_function(prob, 0.2291)
@test new_prob.f(θ₀, data) ≈ negloglik(θ₀, data) - 0.2291
@inferred new_prob.f(θ₀, data)