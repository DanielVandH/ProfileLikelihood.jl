using ..ProfileLikelihood
using Optimization
using OptimizationNLopt
using OptimizationBBO

@testset "Basic solution" begin
    loglik = (θ, p) -> -(p[1] - θ[1])^2 - p[2] * (θ[2] - θ[1])^2 + 3.0
    θ₀ = zeros(2)
    dat = [1.0, 100.0]
    prob = LikelihoodProblem(loglik, θ₀; data=dat, syms=[:α, :β],
        f_kwargs=(adtype=Optimization.AutoFiniteDiff(),))
    sol = mle(prob, NLopt.LN_NELDERMEAD())
    @test ProfileLikelihood.get_mle(sol) == sol.mle ≈ [1.0, 1.0]
    @test ProfileLikelihood.get_problem(sol) == sol.problem == prob
    @test ProfileLikelihood.get_optimiser(sol) == sol.optimiser == NLopt.LN_NELDERMEAD
    @test ProfileLikelihood.get_maximum(sol) == sol.maximum ≈ 3.0
    @test ProfileLikelihood.get_retcode(sol) == sol.retcode
    @test ProfileLikelihood.number_of_parameters(prob) == 2
end

@testset "Check we can use multiple algorithms" begin
    loglik = (θ, p) -> -(p[1] - θ[1])^2 - p[2] * (θ[2] - θ[1])^2 + 3.0
    θ₀ = zeros(2)
    dat = [1.6, 100.0]
    prob = LikelihoodProblem(loglik, θ₀; data=dat, syms=[:α, :β],
        f_kwargs=(adtype=Optimization.AutoFiniteDiff(),))
    alg1 = NLopt.LN_NELDERMEAD
    alg2 = NLopt.LN_BOBYQA
    alg3 = NLopt.LN_NELDERMEAD
    alg4 = BBO_adaptive_de_rand_1_bin()
    alg = (alg1, alg2, alg3, alg4, alg1)
    @test_throws "The algorithm" mle(prob, alg)
    @test ProfileLikelihood.number_of_parameters(prob) == 2

    dat = [1.0, 100.0]
    prob = LikelihoodProblem(loglik, θ₀; data=dat, syms=[:α, :β],
        f_kwargs=(adtype=Optimization.AutoFiniteDiff(),))
    sol = mle(prob, (alg1,))
    @test ProfileLikelihood.get_mle(sol) == sol.mle ≈ [1.0, 1.0]
    @test ProfileLikelihood.get_problem(sol) == sol.problem == prob
    @test ProfileLikelihood.get_optimiser(sol) == sol.optimiser == (NLopt.LN_NELDERMEAD,)
    @test ProfileLikelihood.get_maximum(sol) == sol.maximum ≈ 3.0
    @test ProfileLikelihood.get_retcode(sol) == sol.retcode
    @test ProfileLikelihood.number_of_parameters(prob) == 2
end

@testset "Check that we can index correctly" begin
    syms = [:α, :β]
    dat = [1.0, 100.0]
    θ₀ = zeros(2)
    alg1 = NLopt.LN_NELDERMEAD
    alg2 = NLopt.LN_BOBYQA
    alg3 = NLopt.LN_NELDERMEAD
    alg4 = BBO_adaptive_de_rand_1_bin()
    alg = (alg1, alg2, alg3, alg4, alg1)
    loglik = (θ, p) -> -(p[1] - θ[1])^2 - p[2] * (θ[2] - θ[1])^2 + 3.0
    prob = LikelihoodProblem(loglik, θ₀; data=dat, syms=syms,
        f_kwargs=(adtype=Optimization.AutoFiniteDiff(),),
        prob_kwargs=(lb=[-5.0, -5.0], ub=[10.0, 0.5]))
    sol = mle(prob, (alg1, alg2, alg3, alg1, alg4))
    @test ProfileLikelihood.get_mle(sol) == sol.mle ≈ [0.5049504933896497, 0.5]
    @test ProfileLikelihood.get_problem(sol) == sol.problem == prob
    @test ProfileLikelihood.get_optimiser(sol) == sol.optimiser == (alg1, alg2, alg3, alg1, alg4)
    @test ProfileLikelihood.get_maximum(sol) == sol.maximum ≈ 2.7524752475247523
    @test ProfileLikelihood.get_retcode(sol) == sol.retcode
    @test ProfileLikelihood.get_syms(sol) == syms
    @test ProfileLikelihood.number_of_parameters(prob) == 2
    @test ProfileLikelihood.get_mle(sol, 1) == sol.mle[1] ≈ 0.5049504933896497
    @test ProfileLikelihood.get_mle(sol, 2) == sol.mle[2] ≈ 0.5
    @test sol[1] == sol.mle[1]
    @test sol[2] == sol.mle[2]
    @test SciMLBase.sym_to_index(:α, sol) == 1
    @test SciMLBase.sym_to_index(:β, sol) == 2
    @test sol[:α] == sol.mle[1]
    @test sol[:β] == sol.mle[2]
    @test sol[[:α, :β]] == sol.mle
    @test sol[[:β, :α]] == [sol.mle[2], sol.mle[1]]
    @test sol[1:2] == sol.mle[1:2]
    @test sol[[1, 2]] == sol.mle[[1, 2]]
end

@testset "Check that we can use a sol to update an initial estimate" begin
    syms = [:α, :β]
    dat = [1.0, 100.0]
    θ₀ = zeros(2)
    alg1 = NLopt.LN_NELDERMEAD
    alg2 = NLopt.LN_BOBYQA
    alg3 = NLopt.LN_NELDERMEAD
    alg4 = BBO_adaptive_de_rand_1_bin()
    alg = (alg1, alg2, alg3, alg4, alg1)
    loglik = (θ, p) -> -(p[1] - θ[1])^2 - p[2] * (θ[2] - θ[1])^2 + 3.0
    prob = LikelihoodProblem(loglik, θ₀; data=dat, syms=syms,
        f_kwargs=(adtype=Optimization.AutoFiniteDiff(),),
        prob_kwargs=(lb=[-5.0, -5.0], ub=[10.0, 0.5]))
    sol = mle(prob, (alg1, alg2, alg3, alg1, alg4))
    new_prob = ProfileLikelihood.update_initial_estimate(prob, sol)
    @test new_prob.θ₀ == new_prob.problem.u0 == sol.mle
    @test ProfileLikelihood.update_initial_estimate(new_prob, sol) == new_prob
end