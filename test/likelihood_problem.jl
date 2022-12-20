using ..ProfileLikelihood
using Optimization
using OrdinaryDiffEq
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
@test !ProfileLikelihood.has_upper_bounds(prob)
@test !ProfileLikelihood.has_lower_bounds(prob)
@test !ProfileLikelihood.has_bounds(prob)

lb = [1.0, 2.0]
ub = [Inf, Inf]
prob = ProfileLikelihood.construct_optimisation_problem(negloglik, θ₀, data; lb=lb, ub=ub)
@test prob == OptimizationProblem(negloglik, θ₀, data; lb=lb, ub=ub)
@test prob.lb === lb == ProfileLikelihood.get_lower_bounds(prob)
@test prob.ub === ub == ProfileLikelihood.get_upper_bounds(prob)
@test ProfileLikelihood.finite_lower_bounds(prob)
@test !ProfileLikelihood.finite_upper_bounds(prob)
@test ProfileLikelihood.has_upper_bounds(prob)
@test ProfileLikelihood.has_lower_bounds(prob)
@test ProfileLikelihood.has_bounds(prob)
@test ProfileLikelihood.get_lower_bounds(prob, 1) == 1.0
@test ProfileLikelihood.get_lower_bounds(prob, 2) == 2.0
@test ProfileLikelihood.get_upper_bounds(prob, 1) == Inf
@test ProfileLikelihood.get_upper_bounds(prob, 2) == Inf

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
@test ProfileLikelihood.get_problem(prob) == prob.problem
@test ProfileLikelihood.get_data(prob) == prob.data
@test ProfileLikelihood.get_log_likelihood_function(prob) == loglik
@test ProfileLikelihood.get_θ₀(prob) == θ₀ == prob.θ₀ == prob.problem.u0
@test ProfileLikelihood.get_θ₀(prob, 1) == 5.0
@test ProfileLikelihood.get_θ₀(prob, 2) == 2.0
@test ProfileLikelihood.get_syms(prob) == syms == prob.syms
@test !ProfileLikelihood.has_upper_bounds(prob)
@test !ProfileLikelihood.has_lower_bounds(prob)
@test !ProfileLikelihood.finite_lower_bounds(prob)
@test !ProfileLikelihood.finite_upper_bounds(prob)
@test !ProfileLikelihood.has_bounds(prob)
@test ProfileLikelihood.number_of_parameters(prob) == 2

adtype = Optimization.AutoFiniteDiff()
prob = ProfileLikelihood.LikelihoodProblem(loglik, θ₀; syms, f_kwargs=(adtype=adtype,))
@test ProfileLikelihood.get_problem(prob) == prob.problem
@test ProfileLikelihood.get_data(prob) == prob.data
@test ProfileLikelihood.get_log_likelihood_function(prob) == loglik
@test ProfileLikelihood.get_θ₀(prob) == θ₀ == prob.θ₀ == prob.problem.u0
@test ProfileLikelihood.get_syms(prob) == syms == prob.syms
@test prob.problem.f.adtype isa Optimization.AutoFiniteDiff
@test prob.problem.lb === nothing == ProfileLikelihood.get_lower_bounds(prob)
@test prob.problem.ub === nothing == ProfileLikelihood.get_upper_bounds(prob)
@test !ProfileLikelihood.has_upper_bounds(prob)
@test !ProfileLikelihood.has_lower_bounds(prob)
@test !ProfileLikelihood.finite_lower_bounds(prob)
@test !ProfileLikelihood.finite_upper_bounds(prob)
@test !ProfileLikelihood.finite_bounds(prob)
@test !ProfileLikelihood.has_bounds(prob)
@test ProfileLikelihood.number_of_parameters(prob) == 2

adtype = Optimization.AutoFiniteDiff()
data = [1.0, 3.0]
lb = [3.0, -3.0]
ub = [Inf, Inf]
prob = ProfileLikelihood.LikelihoodProblem(loglik, θ₀; f_kwargs=(adtype=adtype,), prob_kwargs=(lb=lb, ub=ub), data)
@test ProfileLikelihood.get_problem(prob) == prob.problem
@test ProfileLikelihood.get_data(prob) == data == prob.data
@test ProfileLikelihood.get_log_likelihood_function(prob) == loglik
@test ProfileLikelihood.get_θ₀(prob) == θ₀ == prob.θ₀ == prob.problem.u0
@test ProfileLikelihood.get_syms(prob) == 1:2
@test prob.problem.f.adtype isa Optimization.AutoFiniteDiff
@test prob.problem.lb == lb == ProfileLikelihood.get_lower_bounds(prob)
@test prob.problem.ub == ub == ProfileLikelihood.get_upper_bounds(prob)
@test ProfileLikelihood.has_upper_bounds(prob)
@test ProfileLikelihood.has_lower_bounds(prob)
@test ProfileLikelihood.finite_lower_bounds(prob)
@test !ProfileLikelihood.finite_upper_bounds(prob)
@test !ProfileLikelihood.finite_bounds(prob)
@test ProfileLikelihood.has_bounds(prob)
@test ProfileLikelihood.number_of_parameters(prob) == 2

adtype = Optimization.AutoFiniteDiff()
data = [1.0, 3.0]
lb = [Inf, Inf]
ub = [3.0, -3.0]
prob = ProfileLikelihood.LikelihoodProblem(loglik, θ₀; f_kwargs=(adtype=adtype,), prob_kwargs=(lb=lb, ub=ub), data)
@test ProfileLikelihood.get_problem(prob) == prob.problem
@test ProfileLikelihood.get_data(prob) == data == prob.data
@test ProfileLikelihood.get_log_likelihood_function(prob) == loglik
@test ProfileLikelihood.get_θ₀(prob) == θ₀ == prob.θ₀ == prob.problem.u0
@test ProfileLikelihood.get_syms(prob) == 1:2
@test prob.problem.f.adtype isa Optimization.AutoFiniteDiff
@test prob.problem.lb == lb == ProfileLikelihood.get_lower_bounds(prob)
@test prob.problem.ub == ub == ProfileLikelihood.get_upper_bounds(prob)
@test ProfileLikelihood.has_upper_bounds(prob)
@test ProfileLikelihood.has_lower_bounds(prob)
@test !ProfileLikelihood.finite_lower_bounds(prob)
@test ProfileLikelihood.finite_upper_bounds(prob)
@test !ProfileLikelihood.finite_bounds(prob)
@test ProfileLikelihood.has_bounds(prob)
@test ProfileLikelihood.number_of_parameters(prob) == 2

## Test the construction of the LikelihoodProblem with an integrator
f = (u, p, t) -> 1.01u
u₀ = [0.5]
tspan = (0.0, 1.0)
p = nothing
ode_alg = Tsit5()
prob = ProfileLikelihood.LikelihoodProblem(loglik, u₀, f, u₀, tspan;
    ode_alg, ode_kwargs=(saveat=0.25,), ode_parameters=p)
@test ProfileLikelihood.get_problem(prob) == prob.problem
@test ProfileLikelihood.get_data(prob) == SciMLBase.NullParameters()
@test ProfileLikelihood.get_log_likelihood_function(prob).loglik == loglik
@test ProfileLikelihood.get_θ₀(prob) == u₀ == prob.θ₀ == prob.problem.u0
@test ProfileLikelihood.get_syms(prob) == [1]
@test ProfileLikelihood.get_log_likelihood_function(prob).integrator.alg == ode_alg
@test ProfileLikelihood.get_log_likelihood_function(prob).integrator.p == p
@test ProfileLikelihood.get_log_likelihood_function(prob).integrator.opts.saveat.valtree == 0.25:0.25:1.0
@test ProfileLikelihood.get_log_likelihood_function(prob).integrator.f.f == f
@test prob.problem.lb === nothing == ProfileLikelihood.get_lower_bounds(prob)
@test prob.problem.ub === nothing == ProfileLikelihood.get_upper_bounds(prob)
@test !ProfileLikelihood.has_upper_bounds(prob)
@test !ProfileLikelihood.has_lower_bounds(prob)
@test !ProfileLikelihood.finite_lower_bounds(prob)
@test !ProfileLikelihood.finite_upper_bounds(prob)
@test !ProfileLikelihood.finite_bounds(prob)
@test !ProfileLikelihood.has_bounds(prob)
@test ProfileLikelihood.number_of_parameters(prob) == 1

p = [1.0, 3.0]
ode_alg = Rosenbrock23(autodiff=false)
dat = [2.0, 3.0]
syms = [:u]
prob = ProfileLikelihood.LikelihoodProblem(loglik, u₀, f, u₀, tspan;
    data=dat, syms=syms,
    ode_alg, ode_kwargs=(saveat=0.25,), ode_parameters=p,
    prob_kwargs=(lb=lb, ub=ub), f_kwargs=(adtype=adtype,))
@test ProfileLikelihood.get_problem(prob) == prob.problem
@test ProfileLikelihood.get_data(prob) == dat
@test ProfileLikelihood.get_log_likelihood_function(prob).loglik == loglik
@test ProfileLikelihood.get_θ₀(prob) == u₀ == prob.θ₀ == prob.problem.u0
@test ProfileLikelihood.get_syms(prob) == syms
@test ProfileLikelihood.get_log_likelihood_function(prob).integrator.alg == Rosenbrock23{1,false,OrdinaryDiffEq.LinearSolve.GenericLUFactorization{OrdinaryDiffEq.LinearAlgebra.RowMaximum},typeof(OrdinaryDiffEq.DEFAULT_PRECS),Val{:forward},true,nothing}(OrdinaryDiffEq.LinearSolve.GenericLUFactorization{OrdinaryDiffEq.LinearAlgebra.RowMaximum}(OrdinaryDiffEq.LinearAlgebra.RowMaximum()), OrdinaryDiffEq.DEFAULT_PRECS)
@test ProfileLikelihood.get_log_likelihood_function(prob).integrator.p == p
@test ProfileLikelihood.get_log_likelihood_function(prob).integrator.opts.saveat.valtree == 0.25:0.25:1.0
@test ProfileLikelihood.get_log_likelihood_function(prob).integrator.f.f == f
@test prob.problem.f.adtype isa Optimization.AutoFiniteDiff
@test prob.problem.lb == lb == ProfileLikelihood.get_lower_bounds(prob)
@test prob.problem.ub == ub == ProfileLikelihood.get_upper_bounds(prob)
@test ProfileLikelihood.has_upper_bounds(prob)
@test ProfileLikelihood.has_lower_bounds(prob)
@test !ProfileLikelihood.finite_lower_bounds(prob)
@test ProfileLikelihood.finite_upper_bounds(prob)
@test !ProfileLikelihood.finite_bounds(prob)
@test ProfileLikelihood.has_bounds(prob)
@test ProfileLikelihood.number_of_parameters(prob) == 1

## Test the indexing 
loglik = (θ, p) -> θ[1] * p[1] + θ[2]
θ₀ = [5.0, 2.0]
syms = [:a, :b]
prob = ProfileLikelihood.LikelihoodProblem(loglik, θ₀; syms)
@test SciMLBase.sym_to_index(:a, prob) == 1
@test SciMLBase.sym_to_index(:b, prob) == 2
@test prob[1] == 5.0
@test prob[2] == 2.0
@test prob[:a] == 5.0
@test prob[:b] == 2.0
@test prob[[1, 2]] == [5.0, 2.0]
@test prob[1:2] == [5.0, 2.0]
@test prob[[:a, :b]] == [5.0, 2.0]
@test_throws BoundsError prob[:c]

## Test that we can replace the initial estimate
new_θ = [2.0, 3.0]
new_prob = ProfileLikelihood.update_initial_estimate(prob, new_θ)
@test new_prob.θ₀ == new_θ
@test new_prob.problem.u0 == new_θ

## Checking that a parameter is inbounds 
lb = [Inf, Inf]
ub = [3.0, -3.0]
p = [1.0, 3.0]
ode_alg = Rosenbrock23(autodiff=false)
dat = [2.0, 3.0]
syms = [:u, :b]
prob = ProfileLikelihood.LikelihoodProblem(loglik, u₀, f, u₀, tspan;
    data=dat, syms=syms,
    ode_alg, ode_kwargs=(saveat=0.25,), ode_parameters=p,
    prob_kwargs=(lb=lb, ub=ub), f_kwargs=(adtype=adtype,))
@test !ProfileLikelihood.parameter_is_inbounds(prob, [2.0, 3.0])

loglik = (θ, p) -> θ[1] * p[1] + θ[2]
θ₀ = [5.0, 2.0]
syms = [:a, :b]
prob = ProfileLikelihood.LikelihoodProblem(loglik, θ₀; syms)
@test ProfileLikelihood.parameter_is_inbounds(prob, [2.0, 3.0])
@test ProfileLikelihood.parameter_is_inbounds(prob, randn(2))

lb = [-1.0, 1.0]
ub = [5.7, 3.3]
prob = ProfileLikelihood.LikelihoodProblem(loglik, θ₀; syms, prob_kwargs=(lb=lb,ub=ub))
@test ProfileLikelihood.parameter_is_inbounds(prob, [2.0, 3.0])
@test !ProfileLikelihood.parameter_is_inbounds(prob, [0.0,0.0])
@test !ProfileLikelihood.parameter_is_inbounds(prob.problem, [0.0,0.0])
@test ProfileLikelihood.parameter_is_inbounds(prob.problem, [0.0,2.0])

prob = ProfileLikelihood.LikelihoodProblem(loglik, θ₀; syms)
@test ProfileLikelihood.parameter_is_inbounds(prob, [2.0, 3.0])
@test ProfileLikelihood.parameter_is_inbounds(prob, [0.0,0.0])
@test ProfileLikelihood.parameter_is_inbounds(prob.problem, [0.0,0.0])
@test ProfileLikelihood.parameter_is_inbounds(prob.problem, [0.0,2.0])

lb = [-Inf, 1.0]
ub = [5.7, Inf]
prob = ProfileLikelihood.LikelihoodProblem(loglik, θ₀; syms, prob_kwargs=(lb=lb,ub=ub))
@test ProfileLikelihood.parameter_is_inbounds(prob, [2.0, 3.0])
@test ProfileLikelihood.parameter_is_inbounds(prob, [2.0, 5.1])
@test ProfileLikelihood.parameter_is_inbounds(prob, [0.0,3.0])
@test !ProfileLikelihood.parameter_is_inbounds(prob.problem, [0.0,0.0])
@test !ProfileLikelihood.parameter_is_inbounds(prob.problem, [10.0,2.0])
