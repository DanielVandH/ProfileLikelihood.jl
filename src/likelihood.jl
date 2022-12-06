using SciMLBase
using InvertedIndices

export LikelihoodProblem
export mle

######################################################
## AbstractLikelihoodProblem
######################################################
abstract type AbstractLikelihoodProblem end

get_problem(prob::AbstractLikelihoodProblem) = prob.problem
get_data(prob::AbstractLikelihoodProblem) = prob.data
get_log_likelihood_function(prob::AbstractLikelihoodProblem) = prob.log_likelihood_function
get_θ₀(prob::AbstractLikelihoodProblem) = prob.θ₀
get_syms(prob::AbstractLikelihoodProblem) = prob.syms

get_lower_bounds(prob::OptimizationProblem) = prob.lb
get_upper_bounds(prob::OptimizationProblem) = prob.ub
get_lower_bounds(prob::AbstractLikelihoodProblem) = (get_lower_bounds ∘ get_problem)(prob)
get_upper_bounds(prob::AbstractLikelihoodProblem) = (get_upper_bounds ∘ get_problem)(prob)

has_lower_bounds(prob) = !isnothing(get_lower_bounds(prob))
has_upper_bounds(prob) = !isnothing(get_upper_bounds(prob))

finite_lower_bounds(prob) = has_lower_bounds(prob) && all(isfinite, get_lower_bounds(prob))
finite_upper_bounds(prob) = has_upper_bounds(prob) && all(isfinite, get_upper_bounds(prob))
finite_bounds(prob) = finite_lower_bounds(prob) && finite_upper_bounds(prob)

######################################################
## LikelihoodProblem
######################################################
Base.@kwdef struct LikelihoodProblem{P,D,L,Θ,N} <: AbstractLikelihoodProblem
    problem::P
    data::D
    log_likelihood_function::L
    θ₀::Θ
    syms::N
end

function LikelihoodProblem(loglik::Function, θ₀;
    syms=eachindex(θ₀), data=SciMLBase.NullParameters(),
    f_kwargs=nothing, prob_kwargs=nothing)
    negloglik = negate_loglik(loglik)
    opt_f = isnothing(f_kwargs) ? construct_optimisation_function(negloglik, syms) : construct_optimisation_function(negloglik, syms; f_kwargs...)
    opt_prob = isnothing(prob_kwargs) ? construct_optimisation_problem(opt_f, θ₀, data) : construct_optimisation_problem(opt_f, θ₀, data; prob_kwargs...)
    return LikelihoodProblem(opt_prob, data, loglik, θ₀, syms)
end
function LikelihoodProblem(loglik::Function, θ₀, integrator;
    syms=eachindex(θ₀), data=SciMLBase.NullParameters(),
    f_kwargs=nothing, prob_kwargs=nothing)
    _loglik = @inline (θ, p) -> loglik(θ, p, integrator)
    return LikelihoodProblem(_loglik, θ₀; syms, data, f_kwargs, prob_kwargs)
end
function LikelihoodProblem(loglik::Function, θ₀,
    ode_function, u₀, tspan;
    syms=eachindex(θ₀), data=SciMLBase.NullParameters(),
    ode_parameters=SciMLBase.NullParameters(), ode_alg,
    ode_kwargs=nothing, f_kwargs=nothing, prob_kwargs=nothing)
    integrator = isnothing(ode_kwargs) ? construct_integrator(ode_function, u₀, tspan, ode_parameters, ode_alg) : construct_integrator(ode_function, u₀, tspan, ode_parameters, ode_alg; ode_kwargs...)
    return LikelihoodProblem(loglik, θ₀, integrator;
        syms, data, f_kwargs, prob_kwargs)
end

@inline negate_loglik(loglik) = @inline (θ, p) -> -loglik(θ, p)
@inline negate_loglik(loglik, integrator) = @inline (θ, p) -> -loglik(θ, p, integrator)

function construct_optimisation_function(negloglik, syms; f_kwargs...)
    if :adtype ∈ keys(f_kwargs)
        return OptimizationFunction(negloglik, f_kwargs[:adtype]; syms=syms, f_kwargs[Not(:adtype)]...)
    else
        return OptimizationFunction(negloglik, SciMLBase.NoAD(); syms=syms, f_kwargs...)
    end
end
function construct_optimisation_problem(opt_f, θ₀, data; prob_kwargs...)
    return OptimizationProblem(opt_f, θ₀, data; prob_kwargs...)
end

function construct_integrator(prob, ode_alg; ode_kwargs...)
    return SciMLBase.init(prob, ode_alg; ode_kwargs...)
end
function construct_integrator(f, u₀, tspan, p, ode_alg; ode_kwargs...)
    prob = ODEProblem(f, u₀, tspan, p; ode_kwargs...)
    return construct_integrator(prob, ode_alg; ode_kwargs...)
end

######################################################
## AbstractLikelihoodSolution
######################################################
abstract type AbstractLikelihoodSolution end 

get_mle(sol::AbstractLikelihoodSolution) = sol.mle 
get_problem(sol::AbstractLikelihoodSolution) = sol.problem 
get_optimiser(sol::AbstractLikelihoodSolution) = sol.optimiser 
get_maximum(sol::AbstractLikelihoodSolution) = sol.maximum 
get_retcode(sol::AbstractLikelihoodSolution) = sol.retcode 

######################################################
## LikelihoodSolution 
######################################################
Base.@kwdef struct LikelihoodSolution{Θ, P, M, R, A} <: AbstractLikelihoodSolution
    mle::Θ
    problem::P 
    optimiser::A
    maximum::M 
    retcode::R 
end
function LikelihoodSolution(sol::SciMLBase.OptimizationSolution, prob::AbstractLikelihoodProblem)
    return LikelihoodSolution(sol.u, prob, sol.alg, -sol.minimum, sol.retcode)
end

######################################################
## MLE
######################################################
function mle(prob::LikelihoodProblem, alg, args...; kwargs...)
    opt_prob = get_problem(prob)
    opt_sol = solve(opt_prob, alg, args...; kwargs...)
    return LikelihoodSolution(opt_sol, prob)
end