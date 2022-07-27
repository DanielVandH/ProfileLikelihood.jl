using LaTeXStrings
using Random
using Distributions
using OptimizationNLopt
using DifferentialEquations
using Test
using Optimization
using Interpolations
using PreallocationTools

prob, loglikk, θ, uᵒ, n = LogisticODE();
λ, K, σ, u₀ = θ
sol = mle(prob, NLopt.LN_NELDERMEAD())

for cache in [zeros(num_params(prob)), dualcache(zeros(num_params(prob)))]
    # Scaled OptimizationFunction
    optprob = prob.prob
    mles = mle(sol)
    cache = zeros(num_params(prob))
    new_f = ProfileLikelihood.scaled_f(optprob, mles, cache)
    @test new_f(ones(num_params(optprob)), data(prob)) ≈ optprob.f(mles, data(prob))
    @inferred new_f(ones(num_params(optprob)), data(prob))

    # Scaled OptimizationProblem
    newprob = ProfileLikelihood.scale_prob(optprob, mles, cache)
    @test newprob.u0 == optprob.u0 ./ mles
    @test newprob.f(ones(num_params(optprob)), data(prob)) ≈ optprob.f(mles, data(prob))
    @inferred newprob.f(ones(num_params(optprob)), data(prob))
    @test newprob.lb == optprob.lb ./ mles
    @test newprob.ub == optprob.ub ./ mles
    newprob = ProfileLikelihood.scale_prob(optprob, -mles, cache)
    @test newprob.lb == -optprob.ub ./ mles
    @test newprob.ub == -optprob.lb ./ mles

    # Scaled LikelihoodProblem
    newprob = ProfileLikelihood.scale_prob(prob, mles, cache)
    @test newprob.prob.f(ones(num_params(prob)), data(prob)) ≈ prob.prob.f(mles, data(prob))
    @test newprob.θ₀ ≈ prob.θ₀ ./ mles
    @inferred newprob.prob.f(ones(num_params(prob)), data(prob))
    @test newprob.θ₀ == newprob.prob.u0
    @test newprob.prob.lb == optprob.lb ./ mles
    @test newprob.prob.ub == optprob.ub ./ mles

    # Unscaling OptimizationProblem
    newprob = ProfileLikelihood.scale_prob(optprob, mles, cache)
    newprob2 = ProfileLikelihood.scale_prob(newprob, 1.0 ./ mles, cache)
    @test newprob.f(ones(num_params(optprob)), data(prob)) ≈ newprob2.f(mles, data(prob)) ≈ optprob.f(mles, data(prob))
    @test newprob2.u0 ≈ optprob.u0
    @inferred newprob2.f(mles, data(prob))
    @test newprob2.lb ≈ optprob.lb
    @test newprob2.ub ≈ optprob.ub

    # Unscaling LikelihoodProblem
    newprob = ProfileLikelihood.scale_prob(prob, mles, cache)
    newprob2 = ProfileLikelihood.scale_prob(newprob, 1.0 ./ mles, cache)
    @test newprob.prob.f(ones(num_params(prob)), data(prob)) ≈ newprob2.prob.f(mles, data(prob)) ≈ prob.prob.f(mles, data(prob))
    @test newprob2.θ₀ ≈ prob.θ₀
    @inferred newprob2.prob.f(mles, data(prob))
    @test newprob2.prob.lb ≈ optprob.lb
    @test newprob2.prob.ub ≈ optprob.ub
end

mles = mle(sol)
param_ranges = construct_profile_ranges(prob, sol, 200)
param_ranges = ProfileLikelihood.scale_param_ranges(prob, mles, param_ranges)
@test param_ranges[1][1] ≈ LinRange(1.0, 0.0, 200)
@test param_ranges[1][2] ≈ LinRange(1.0, 10.0 / mles[1], 200)
@test param_ranges[2][1] ≈ LinRange(1.0, 1.0e-6 / mles[2], 200)
@test param_ranges[2][2] ≈ LinRange(1.0, 10.0 / mles[2], 200)
@test param_ranges[3][1] ≈ LinRange(1.0, 1.0e-6 / mles[3], 200)
@test param_ranges[3][2] ≈ LinRange(1.0, 10.0 / mles[3], 200)
@test param_ranges[4][1] ≈ LinRange(1.0, 0.0, 200)
@test param_ranges[4][2] ≈ LinRange(1.0, 10.0 / mles[4], 200)

prof = profile(prob, sol; mle_scale=(true, false))
for i in 1:num_params(prof)
    @test 1.0 ∈ confidence_intervals(prof, i)
end
@test prof.prob.θ₀ == prob.θ₀ ./ mle(sol)
@test prof.prob.prob.u0 == prob.prob.u0 ./ mle(sol)
@inferred prof.prob.prob.f(ones(4), data(prob))
@test prof.prob.prob.f(ones(4), data(prob)) ≈ prob.prob.f(mle(sol), data(prob))
@test prof.mle.θ ≈ ones(4)

prof = profile(prob, sol; mle_scale=(true, true))
@test λ ∈ confidence_intervals(prof, 1)
@test K ∈ confidence_intervals(prof, 2)
@test σ ∈ confidence_intervals(prof, 3)
@test u₀ ∈ confidence_intervals(prof, 4)
@test prof.prob.θ₀ == prob.θ₀
@test prof.prob.prob.u0 == prob.prob.u0
@inferred prof.prob.prob.f(mle(sol), data(prob))
@test prof.prob.prob.f(mle(sol), data(prob)) ≈ prob.prob.f(mle(sol), data(prob))
@test prof.mle.θ ≈ mle(sol)

prof2 = profile(prob, sol)
for i in 1:num_params(prof)
    @test confidence_intervals(prof, i).lower ≈ confidence_intervals(prof2, i).lower
    @test confidence_intervals(prof, i).upper ≈ confidence_intervals(prof2, i).upper
end

