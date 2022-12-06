using ProfileLikelihood
using Test
using FiniteDiff
using Optimization
using OrdinaryDiffEq
using OptimizationNLopt

const PL = ProfileLikelihood

######################################################
## LikelihoodProblem 
######################################################
## Test that we are correctly negating the likelihood 
loglik = (θ, p) -> 2.0
negloglik = ProfileLikelihood.negate_loglik(loglik)
@test negloglik(rand(), rand()) == -2.0

loglik = (θ, p) -> θ[1] * p[1] + θ[2]
negloglik = ProfileLikelihood.negate_loglik(loglik)
θ, p = rand(2), rand()
@test negloglik(θ, p) ≈ -loglik(θ, p)

## Test the construction of the OptimizationFunction 
loglik = (θ, p) -> θ[1] * p[1] + θ[2]
negloglik = ProfileLikelihood.negate_loglik(loglik)
optf_1 = ProfileLikelihood.construct_optimisation_function(negloglik, 1:5)
@test optf_1 == OptimizationFunction(negloglik, SciMLBase.NoAD(); syms=1:5)

paramsym_vec = [:a, :sys]
optf_2 = ProfileLikelihood.construct_optimisation_function(negloglik, 1:5; paramsyms=paramsym_vec)
@test optf_2 === OptimizationFunction(negloglik, SciMLBase.NoAD(); syms=1:5, paramsyms=paramsym_vec)

adtype = Optimization.AutoFiniteDiff()
optf_3 = ProfileLikelihood.construct_optimisation_function(negloglik, 1:5; adtype=adtype, paramsyms=paramsym_vec)
@test optf_3 === OptimizationFunction(negloglik, adtype; syms=1:5, paramsyms=paramsym_vec)

## Test the construction of the OptimizationProblem 
loglik = (θ, p) -> θ[1] * p[1] + θ[2]
negloglik = ProfileLikelihood.negate_loglik(loglik)
θ₀ = rand(3)
data = (rand(100), [:a, :b])
prob = ProfileLikelihood.construct_optimisation_problem(negloglik, θ₀, data)
@test prob == OptimizationProblem(negloglik, θ₀, data)
@test !PL.has_upper_bounds(prob)
@test !PL.has_lower_bounds(prob)

lb = [1.0, 2.0]
ub = [Inf, Inf]
prob = ProfileLikelihood.construct_optimisation_problem(negloglik, θ₀, data; lb=lb, ub=ub)
@test prob == OptimizationProblem(negloglik, θ₀, data; lb=lb, ub=ub)
@test prob.lb === lb == PL.get_lower_bounds(prob)
@test prob.ub === ub == PL.get_upper_bounds(prob)
@test PL.finite_lower_bounds(prob)
@test !PL.finite_upper_bounds(prob)
@test PL.has_upper_bounds(prob)
@test PL.has_lower_bounds(prob)

## Test the construction of the integrator 
f = (u, p, t) -> 1.01u
u₀ = 0.5
tspan = (0.0, 1.0)
p = nothing
ode_alg = Tsit5()
integ = ProfileLikelihood.construct_integrator(f, u₀, tspan, p, ode_alg)
solve!(integ)
@test all([abs(integ.sol.u[i] - 0.5exp(1.01integ.sol.t[i])) < 0.01 for i in eachindex(integ.sol)])
@test integ.alg == ode_alg

ode_alg = Rosenbrock23()
integ = ProfileLikelihood.construct_integrator(f, u₀, tspan, p, ode_alg; saveat=0.25)
solve!(integ)
@test all([abs(integ.sol.u[i] - 0.5exp(1.01integ.sol.t[i])) < 0.01 for i in eachindex(integ.sol)])
@test integ.sol.t == 0:0.25:1.0

## Test the construction of the LikelihoodProblem with normal inputs
loglik = (θ, p) -> θ[1] * p[1] + θ[2]
θ₀ = [5.0, 2.0]
syms = [:a, :b]
prob = ProfileLikelihood.LikelihoodProblem(loglik, θ₀; syms)
@test PL.get_problem(prob) == prob.problem
@test PL.get_data(prob) == prob.data
@test PL.get_log_likelihood_function(prob) == loglik
@test PL.get_θ₀(prob) == θ₀ == prob.θ₀ == prob.problem.u0
@test PL.get_syms(prob) == syms == prob.syms
@test !PL.has_upper_bounds(prob)
@test !PL.has_lower_bounds(prob)
@test !PL.finite_lower_bounds(prob)
@test !PL.finite_upper_bounds(prob)

adtype = Optimization.AutoFiniteDiff()
prob = PL.LikelihoodProblem(loglik, θ₀; syms, f_kwargs=(adtype=adtype,))
@test PL.get_problem(prob) == prob.problem
@test PL.get_data(prob) == prob.data
@test PL.get_log_likelihood_function(prob) == loglik
@test PL.get_θ₀(prob) == θ₀ == prob.θ₀ == prob.problem.u0
@test PL.get_syms(prob) == syms == prob.syms
@test prob.problem.f.adtype isa Optimization.AutoFiniteDiff
@test prob.problem.lb === nothing == PL.get_lower_bounds(prob)
@test prob.problem.ub === nothing == PL.get_upper_bounds(prob)
@test !PL.has_upper_bounds(prob)
@test !PL.has_lower_bounds(prob)
@test !PL.finite_lower_bounds(prob)
@test !PL.finite_upper_bounds(prob)
@test !PL.finite_bounds(prob)

adtype = Optimization.AutoFiniteDiff()
data = [1.0, 3.0]
lb = [3.0, -3.0]
ub = [Inf, Inf]
prob = PL.LikelihoodProblem(loglik, θ₀; f_kwargs=(adtype=adtype,), prob_kwargs=(lb=lb, ub=ub), data)
@test PL.get_problem(prob) == prob.problem
@test PL.get_data(prob) == data == prob.data
@test PL.get_log_likelihood_function(prob) == loglik
@test PL.get_θ₀(prob) == θ₀ == prob.θ₀ == prob.problem.u0
@test PL.get_syms(prob) == 1:2
@test prob.problem.f.adtype isa Optimization.AutoFiniteDiff
@test prob.problem.lb == lb == PL.get_lower_bounds(prob)
@test prob.problem.ub == ub == PL.get_upper_bounds(prob)
@test PL.has_upper_bounds(prob)
@test PL.has_lower_bounds(prob)
@test PL.finite_lower_bounds(prob)
@test !PL.finite_upper_bounds(prob)
@test !PL.finite_bounds(prob)

adtype = Optimization.AutoFiniteDiff()
data = [1.0, 3.0]
lb = [Inf, Inf]
ub = [3.0, -3.0]
prob = PL.LikelihoodProblem(loglik, θ₀; f_kwargs=(adtype=adtype,), prob_kwargs=(lb=lb, ub=ub), data)
@test PL.get_problem(prob) == prob.problem
@test PL.get_data(prob) == data == prob.data
@test PL.get_log_likelihood_function(prob) == loglik
@test PL.get_θ₀(prob) == θ₀ == prob.θ₀ == prob.problem.u0
@test PL.get_syms(prob) == 1:2
@test prob.problem.f.adtype isa Optimization.AutoFiniteDiff
@test prob.problem.lb == lb == PL.get_lower_bounds(prob)
@test prob.problem.ub == ub == PL.get_upper_bounds(prob)
@test PL.has_upper_bounds(prob)
@test PL.has_lower_bounds(prob)
@test !PL.finite_lower_bounds(prob)
@test PL.finite_upper_bounds(prob)
@test !PL.finite_bounds(prob)

## Test the construction of the LikelihoodProblem with an integrator
f = (u, p, t) -> 1.01u
u₀ = [0.5]
tspan = (0.0, 1.0)
p = nothing
ode_alg = Tsit5()
prob = PL.LikelihoodProblem(loglik, u₀, f, u₀, tspan;
    ode_alg, ode_kwargs=(saveat=0.25,), ode_parameters=p)
@test PL.get_problem(prob) == prob.problem
@test PL.get_data(prob) == SciMLBase.NullParameters()
@test PL.get_log_likelihood_function(prob).loglik == loglik
@test PL.get_θ₀(prob) == u₀ == prob.θ₀ == prob.problem.u0
@test PL.get_syms(prob) == [1]
@test PL.get_log_likelihood_function(prob).integrator.alg == ode_alg
@test PL.get_log_likelihood_function(prob).integrator.p == p
@test PL.get_log_likelihood_function(prob).integrator.opts.saveat.valtree == 0.25:0.25:1.0
@test PL.get_log_likelihood_function(prob).integrator.f.f == f
@test prob.problem.lb === nothing == PL.get_lower_bounds(prob)
@test prob.problem.ub === nothing == PL.get_upper_bounds(prob)
@test !PL.has_upper_bounds(prob)
@test !PL.has_lower_bounds(prob)
@test !PL.finite_lower_bounds(prob)
@test !PL.finite_upper_bounds(prob)
@test !PL.finite_bounds(prob)

p = [1.0, 3.0]
ode_alg = Rosenbrock23(autodiff=false)
dat = [2.0, 3.0]
syms = [:u]
prob = PL.LikelihoodProblem(loglik, u₀, f, u₀, tspan;
    data=dat, syms=syms,
    ode_alg, ode_kwargs=(saveat=0.25,), ode_parameters=p,
    prob_kwargs=(lb=lb, ub=ub), f_kwargs=(adtype=adtype,))
@test PL.get_problem(prob) == prob.problem
@test PL.get_data(prob) == dat
@test PL.get_log_likelihood_function(prob).loglik == loglik
@test PL.get_θ₀(prob) == u₀ == prob.θ₀ == prob.problem.u0
@test PL.get_syms(prob) == syms
@test PL.get_log_likelihood_function(prob).integrator.alg == Rosenbrock23{1,false,OrdinaryDiffEq.LinearSolve.GenericLUFactorization{OrdinaryDiffEq.LinearAlgebra.RowMaximum},typeof(OrdinaryDiffEq.DEFAULT_PRECS),Val{:forward},true,nothing}(OrdinaryDiffEq.LinearSolve.GenericLUFactorization{OrdinaryDiffEq.LinearAlgebra.RowMaximum}(OrdinaryDiffEq.LinearAlgebra.RowMaximum()), OrdinaryDiffEq.DEFAULT_PRECS)
@test PL.get_log_likelihood_function(prob).integrator.p == p
@test PL.get_log_likelihood_function(prob).integrator.opts.saveat.valtree == 0.25:0.25:1.0
@test PL.get_log_likelihood_function(prob).integrator.f.f == f
@test prob.problem.f.adtype isa Optimization.AutoFiniteDiff
@test prob.problem.lb == lb == PL.get_lower_bounds(prob)
@test prob.problem.ub == ub == PL.get_upper_bounds(prob)
@test PL.has_upper_bounds(prob)
@test PL.has_lower_bounds(prob)
@test !PL.finite_lower_bounds(prob)
@test PL.finite_upper_bounds(prob)
@test !PL.finite_bounds(prob)

######################################################
## MLE
######################################################
loglik = (θ, p) -> -(p[1] - θ[1])^2 - p[2] * (θ[2] - θ[1])^2 + 3.0
θ₀ = zeros(2)
dat = [1.0, 100.0]
prob = LikelihoodProblem(loglik, θ₀; data=dat, syms=[:α, :β],
    f_kwargs=(adtype = Optimization.AutoFiniteDiff(),))
sol = mle(prob, NLopt.LN_NELDERMEAD())
@test PL.get_mle(sol) == sol.mle ≈ [1.0,1.0]
@test PL.get_problem(sol) == sol.problem == prob 
@test PL.get_optimiser(sol) == sol.optimiser == NLopt.LN_NELDERMEAD
@test PL.get_maximum(sol) == sol.maximum ≈ 3.0
@test PL.get_retcode(sol) == sol.retcode 
