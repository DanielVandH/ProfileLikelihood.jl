Base.@kwdef struct LikelihoodProblem{N,P,D,L,Θ,S} <: AbstractLikelihoodProblem{N,L}
    problem::P
    data::D
    log_likelihood_function::L
    θ₀::Θ
    syms::S
end

"""
    LikelihoodProblem{N,P,D,L,Θ,S} <: AbstractLikelihoodProblem

Struct representing a likelihood problem. 
    
# Fields 
- `problem::P`

The associated `OptimizationProblem`.
- `data::D`

The argument `p` used in the log-likelihood function. 
- `log_likelihood_function::L`

The log-likelihood function, taking the form `ℓ(θ, p)`.
- `θ₀::Θ`

Initial estimates for the MLE `θ`.
- `syms::S`

Variable names for the parameters.
    
The extra parameter `N` is the number of parameters.

# Constructors

## Standard

    LikelihoodProblem(loglik::Function, θ₀;
        syms=eachindex(θ₀), data=SciMLBase.NullParameters(),
        f_kwargs=nothing, prob_kwargs=nothing)

Constructor for the [`LikelihoodProblem`](@ref).

### Arguments 
- `loglik::Function`: The log-likelihood function, taking the form `ℓ(θ, p)`.
- `θ₀`: The estimates estimates for the MLEs.

### Keyword Arguments 
- `syms=eachindex(θ₀)`: Names for each parameter. 
- `data=SciMLBase.NullParameters()`: The parameter `p` in the log-likelihood function. 
- `f_kwargs=nothing`: Keyword arguments, passed as a `NamedTuple`, for the `OptimizationFunction`.
- `prob_kwargs=nothing`: Keyword arguments, passed as a `NamedTuple`, for the `OptimizationProblem`.

### Outputs 
Returns the [`LikelihoodProblem`](@ref) problem object.

## With arguments for a differential equation problem

    LikelihoodProblem(loglik::Function, θ₀,
        ode_function, u₀, tspan;
        syms=eachindex(θ₀), data=SciMLBase.NullParameters(),
        ode_parameters=SciMLBase.NullParameters(), ode_alg,
        ode_kwargs=nothing, f_kwargs=nothing, prob_kwargs=nothing)

Constructor for the [`LikelihoodProblem`](@ref) for a differential equation problem.

### Arguments 
- `loglik::Function`: The log-likelihood function, taking the form `ℓ(θ, p, integrator)`.
- `θ₀`: The estimates estimates for the MLEs.
- `ode_function`: The function `f(du, u, p, t)` or `f(u, p, t)` for the differential equation.
- `u₀`: The initial condition for the differential equation. 
- `tspan`: The time-span to solve the differential equaton over. 

### Keyword Arguments 
- `syms=eachindex(θ₀)`: Names for each parameter. 
- `data=SciMLBase.NullParameters()`: The parameter `p` in the log-likelihood function. 
- `ode_parameters=SciMLBase.NullParameters()`: The parameter `p` in `ode_function`.
- `ode_alg`: The algorithm used for solving the differential equatios.
- `ode_kwargs=nothing`: Extra keyword arguments, passed as a `NamedTuple`, to pass into the integrator; see `construct_integrator`.
- `f_kwargs=nothing`: Keyword arguments, passed as a `NamedTuple`, for the `OptimizationFunction`.
- `prob_kwargs=nothing`: Keyword arguments, passed as a `NamedTuple`, for the `OptimizationProblem`.

#### Outputs 
Returns the [`LikelihoodProblem`](@ref) problem object.

## With an integrator 
    LikelihoodProblem(loglik::Function, θ₀, integrator;
        syms=eachindex(θ₀), data=SciMLBase.NullParameters(),
        f_kwargs=nothing, prob_kwargs=nothing)

Constructor for the [`LikelihoodProblem`](@ref) for a differential equation problem 
with associated `integrator`.

### Arguments 
- `loglik::Function`: The log-likelihood function, taking the form `ℓ(θ, p, integrator)`.
- `θ₀`: The estimates estimates for the MLEs.
- `integrator`: The integrator for the differential equation problem. See also `construct_integrator`.

### Keyword Arguments 
- `syms=eachindex(θ₀)`: Names for each parameter. 
- `data=SciMLBase.NullParameters()`: The parameter `p` in the log-likelihood function. 
- `f_kwargs=nothing`: Keyword arguments, passed as a `NamedTuple`, for the `OptimizationFunction`.
- `prob_kwargs=nothing`: Keyword arguments, passed as a `NamedTuple`, for the `OptimizationProblem`.

#### Outputs 
Returns the [`LikelihoodProblem`](@ref) problem object.
"""
function LikelihoodProblem end

function LikelihoodProblem(loglik::Function, θ₀;
    syms=eachindex(θ₀), data=SciMLBase.NullParameters(),
    f_kwargs=nothing, prob_kwargs=nothing)
    Base.require_one_based_indexing(θ₀)
    negloglik = negate_loglik(loglik)
    opt_f = isnothing(f_kwargs) ? construct_optimisation_function(negloglik, syms) : construct_optimisation_function(negloglik, syms; f_kwargs...)
    opt_prob = isnothing(prob_kwargs) ? construct_optimisation_problem(opt_f, θ₀, data) : construct_optimisation_problem(opt_f, θ₀, data; prob_kwargs...)
    return LikelihoodProblem{length(θ₀),typeof(opt_prob),
        typeof(data),typeof(loglik),
        typeof(θ₀),typeof(syms)}(opt_prob, data, loglik, θ₀, syms)
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
    if :kwargs ∈ keys(prob_kwargs)
        return OptimizationProblem(opt_f, θ₀, data; prob_kwargs[Not(:kwargs)]..., prob_kwargs[:kwargs]...)
    else
        return OptimizationProblem(opt_f, θ₀, data; prob_kwargs...)
    end
end

function construct_integrator(prob, ode_alg; ode_kwargs...)
    return SciMLBase.init(prob, ode_alg; ode_kwargs...)
end
function construct_integrator(f, u₀, tspan, p, ode_alg; ode_kwargs...)
    prob = ODEProblem(f, u₀, tspan, p; ode_kwargs...)
    return construct_integrator(prob, ode_alg; ode_kwargs...)
end

function update_initial_estimate(prob::LikelihoodProblem{N,P,D,L,Θ,S}, θ::Θ) where {N,P,D,L,Θ,S}
    new_prob = update_initial_estimate(get_problem(prob), θ)
    return LikelihoodProblem{N,P,D,L,Θ,S}(new_prob, get_data(prob), get_log_likelihood_function(prob),
        θ, get_syms(prob))
end