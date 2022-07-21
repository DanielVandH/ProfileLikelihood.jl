using LaTeXStrings
using Random
using Distributions
using OptimizationNLopt
using DifferentialEquations
using Test
using Optimization
using Dierckx
using PreallocationTools

prob, loglikk, θ, uᵒ, n = LogisticODE();
λ, K, σ, u₀ = θ
p = data(prob)
sol = mle(prob, NLopt.LN_NELDERMEAD())

@testset "Scaling problems" begin
    scale = true

    # Scaled OptimizationFunction 
    optprob = prob.prob
    new_f = ProfileLikelihood.scaled_f(optprob, scale)
    @test new_f(θ, p) == optprob.f(θ, p)
    @inferred new_f(θ, p)

    # Scaled OptimizationProblem 
    newoptprob = ProfileLikelihood.scale_prob(optprob, scale)
    @test newoptprob.f(θ, p) == optprob.f(θ, p)
    @inferred newoptprob.f(θ, p)

    # Scaled LikelihoodProblem
    newprob = ProfileLikelihood.scale_prob(prob, scale)
    @test newprob.prob.f(θ, p) == prob.prob.f(θ, p)
    @inferred newprob.prob.f(θ, p)

    for _ in 1:100
        scale = rand()
        # Scaled OptimizationFunction 
        optprob = prob.prob
        new_f = ProfileLikelihood.scaled_f(optprob, scale)
        @test new_f(θ, p) ≈ optprob.f(θ, p) / scale
        @inferred new_f(θ, p)

        # Scaled OptimizationProblem 
        newoptprob = ProfileLikelihood.scale_prob(optprob, scale)
        @test newoptprob.f(θ, p) ≈ optprob.f(θ, p) / scale
        @inferred newoptprob.f(θ, p)

        # Scaled LikelihoodProblem
        newprob = ProfileLikelihood.scale_prob(prob, scale)
        @test newprob.prob.f(θ, p) ≈ prob.prob.f(θ, p) / scale
        @inferred newprob.prob.f(θ, p)
    end
end

@testset "Scaling LikelihoodSolutions" begin
    scale = true
    sol = mle(prob, NLopt.LN_NELDERMEAD; scale)
    @test mle(sol) ≈ [0.7751404076923547, 1.021425899810363, 0.10183166192237934, 0.5354131613054318]
    @test maximum(sol) ≈ 86.54963187499544

    scale = 1.891298
    sol = mle(prob, NLopt.LN_NELDERMEAD; scale)
    @test mle(sol) ≈ [0.7751404076923547, 1.021425899810363, 0.10183166192237934, 0.5354131613054318]
    @test maximum(sol) ≈ 86.54963187499544

    scale = 0.0
    sol = mle(prob, NLopt.LN_NELDERMEAD; scale)
    @test isnan(maximum(sol))

    scale = 1e36
    sol = mle(prob, NLopt.LN_NELDERMEAD; scale)
    @test mle(sol) ≈ [0.7751404076923547, 1.021425899810363, 0.10183166192237934, 0.5354131613054318]
    @test maximum(sol) ≈ 86.54963187499544

    scale = 2.9837171
    sol = mle(prob, (NLopt.LN_BOBYQA, NLopt.LN_NELDERMEAD); scale)
    @test mle(sol) ≈ [0.7751404076923547, 1.021425899810363, 0.10183166192237934, 0.5354131613054318]
    @test maximum(sol) ≈ 86.54963187499544
end

@testset "Normalisation" begin
    _scale = maximum(sol)

    optprob = prob.prob
    new_f = ProfileLikelihood.scaled_f(optprob, _scale; op = +)
    @test new_f(mle(sol), p) ≈ 0.0
    @inferred new_f(mle(sol), p)

    # Scaled OptimizationProblem 
    newoptprob = ProfileLikelihood.scale_prob(optprob, _scale; op = +)
    @test newoptprob.f(mle(sol), p) ≈ 0.0
    @inferred newoptprob.f(mle(sol), p)

    # Scaled LikelihoodProblem
    newprob = ProfileLikelihood.scale_prob(prob, _scale; op = +)
    @test newprob.prob.f(mle(sol), p) ≈ 0.0
    @inferred newprob.prob.f(mle(sol), p)

end