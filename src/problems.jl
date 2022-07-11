"""
    abstract type AbstractLikelihoodProblem 

Abstract type for defining likelihood problems. See `subtypes(AbstractLikelihoodProblem)`.
"""
abstract type AbstractLikelihoodProblem end

"""
    num_params(prob::OptimizationProblem)
    num_params(prob::AbstractLikelihoodProblem) 

Returns the number of parameters being estimated for the likelihood/optimisation problem `prob`.
"""
@inline num_params(prob::OptimizationProblem) = length(prob.u0)
@inline num_params(prob::AbstractLikelihoodProblem) =num_params(prob.prob)

"""
    data(prob::AbstractLikelihoodProblem)

Returns the known parameters used for the likelihood problem `prob`.
"""
@inline data(prob::AbstractLikelihoodProblem) = prob.data

"""
    names(prob::AbstractLikelihoodProblem)

Returns the vector of names used for the likelihood problem `prob` for plotting.
"""
@inline Base.names(prob::AbstractLikelihoodProblem) = prob.names

"""
    lower_bounds(prob::Union{OptimizationProblem, AbstractLikelihoodProblem})
    lower_bounds(prob::Union{OptimizationProblem, AbstractLikelihoodProblem}, i) 
    upper_bounds(prob::Union{OptimizationProblem, AbstractLikelihoodProblem})
    upper_bounds(prob::Union{OptimizationProblem, AbstractLikelihoodProblem}, i) 
    bounds(prob::Union{OptimizationProblem, AbstractLikelihoodProblem}, i)
    bounds(prob::Union{OptimizationProblem, AbstractLikelihoodProblem}, i)

Returns the lower or upper bounds used for the constraints in the likelihood / optimisation
problem `prob`. `bounds` instead returns a tuple of the lower and upper bounds, or a vector of 
these tuples of `i` is not given.
"""
function bounds end 
@doc (@doc bounds) @inline lower_bounds(prob::OptimizationProblem) = prob.lb 
@doc (@doc bounds) @inline lower_bounds(prob::AbstractLikelihoodProblem) = lower_bounds(prob.prob)
@doc (@doc bounds) @inline upper_bounds(prob::OptimizationProblem) = prob.ub 
@doc (@doc bounds) @inline upper_bounds(prob::AbstractLikelihoodProblem) = upper_bounds(prob.prob)
@doc (@doc bounds) @inline lower_bounds(prob::OptimizationProblem, i) = prob.lb[i]
@doc (@doc bounds) @inline lower_bounds(prob::AbstractLikelihoodProblem, i) = lower_bounds(prob.prob, i)
@doc (@doc bounds) @inline upper_bounds(prob::OptimizationProblem, i) = prob.ub[i] 
@doc (@doc bounds) @inline upper_bounds(prob::AbstractLikelihoodProblem, i) = upper_bounds(prob.prob, i)
@doc (@doc bounds) @inline function bounds(::OptimizationProblem{iip,FF,θType,P,Nothing,LC,UC,Sns,K}, i) where {iip,AD,G,H,HV,C,CJ,CH,HP,CJP,CHP,S,HCV,CJCV,CHCV,EX,CEX,F,FF<:OptimizationFunction{iip,AD,F,G,H,HV,C,CJ,CH,HP,CJP,CHP,S,HCV,CJCV,CHCV,EX,CEX},θType,P,LC,UC,Sns,K}
    (nothing, nothing)
end
@doc (@doc bounds) @inline bounds(prob::OptimizationProblem, i) = (lower_bounds(prob, i), upper_bounds(prob, i))
@doc (@doc bounds) @inline bounds(prob::OptimizationProblem) = [bounds(prob, i) for i in 1:num_params(prob)]
@doc (@doc bounds) @inline bounds(prob::AbstractLikelihoodProblem, i) = bounds(prob.prob, i)
@doc (@doc bounds) @inline bounds(prob::AbstractLikelihoodProblem) = bounds(prob.prob)

"""
    finite_bounds(prob::OptimizationProblem)
    finite_bounds(prob::AbstractLikelihoodProblem)

Returns `true` if the given `prob` has finite lower and upper bounds, and `false`
otherwise. Note that `nothing` bounds are interpreted as unbounded.
"""
function finite_bounds end
function finite_bounds(prob::OptimizationProblem{iip,FF,θType,P,B,LC,UC,Sns,K}) where {iip,AD,G,H,HV,C,CJ,CH,HP,CJP,CHP,S,HCV,CJCV,CHCV,EX,CEX,F,FF<:OptimizationFunction{iip,AD,F,G,H,HV,C,CJ,CH,HP,CJP,CHP,S,HCV,CJCV,CHCV,EX,CEX},θType,P,B,LC,UC,Sns,K}
    all(isfinite, prob.lb) && all(isfinite, prob.ub)
end
function finite_bounds(prob::OptimizationProblem{iip,FF,θType,P,Nothing,LC,UC,Sns,K}) where {iip,AD,G,H,HV,C,CJ,CH,HP,CJP,CHP,S,HCV,CJCV,CHCV,EX,CEX,F,FF<:OptimizationFunction{iip,AD,F,G,H,HV,C,CJ,CH,HP,CJP,CHP,S,HCV,CJCV,CHCV,EX,CEX},θType,P,LC,UC,Sns,K}
    false
end
function finite_bounds(prob::AbstractLikelihoodProblem)
    finite_bounds(prob.prob)
end

"""
    sym_names(prob::AbstractLikelihoodProblem)

Return the symbolic names used for the likelihood problem `prob`, or creates them 
if none were provided in the construction of `prob`.
"""
@inline function sym_names(prob::AbstractLikelihoodProblem)
    nm = prob.prob.f.syms
    !isnothing(nm) && return nm
    rt = Vector{String}(undef, num_params(prob))
    for i in 1:num_params(prob)
        rt[i] = "θ" * subscriptnumber(i)
    end
    return rt
end

"""
    choose_θ₀(::Nothing, lb, ub, num_params)
    choose_θ₀(θ₀::T, args...) where {T<:AbstractVector}

Chooses the initial guess `θ₀` based on the provided lower bounds `lb` and upper bounds 
`ub`, with a number of parameters `num_params`, if no `θ₀` is provided. Otherwise, simply 
returns `θ₀`.
"""
function choose_θ₀ end
@inline function choose_θ₀(::Nothing, lb, ub, num_params)
    if isnothing(lb) && isnothing(ub)
        θ₀ = zeros(num_params)
    elseif !isnothing(lb) && !isnothing(ub)
        if all(isfinite, lb) && all(isfinite, ub)
            θ₀ = similar(lb)
            for i in eachindex(θ₀)
                θ₀[i] = 0.5(lb[i] + ub[i])
            end
        elseif all(isfinite, lb) && !all(isfinite, ub)
            θ₀ = copy(lb)
        elseif !all(isfinite, lb) && all(isfinite, ub)
            θ₀ = copy(ub)
        else
            θ₀ = zeros(num_params)
        end
    else
        error("If any of `lb` or `ub` is provided, both must be provided.")
    end
    return θ₀
end
@inline choose_θ₀(θ₀::T, args...) where {T<:AbstractVector} = θ₀

"""
    negate_loglik(loglik)
    negate_loglik(loglik, integrator)

Returns `(θ, p) -> -loglik(θ, p)`, or `(θ, p) -> -loglik(θ, p, integrator)` if an 
`integrator` is provided.
"""
function negate_loglik end
@inline negate_loglik(loglik) = @inline (θ, p) -> -loglik(θ, p)
@inline negate_loglik(loglik, integrator) = @inline (θ, p) -> -loglik(θ, p, integrator)

"""
construct_optimisation_function(negloglik, adtype,
    grad, hess, hv,
    cons, cons_j, cons_h,
    hess_prototype, cons_jac_prototype, cons_hess_prototype,
    syms,
    hess_colorvec, cons_jac_colorvec, cons_hess_colorvec,
    expr, cons_expr, f_kwargs)

Constructs an optimisation function with objective `negloglik`. 
See [`OptimizationFunction`](@ref).
"""
@inline function construct_optimisation_function(negloglik, adtype,
    grad, hess, hv,
    cons, cons_j, cons_h,
    hess_prototype, cons_jac_prototype, cons_hess_prototype,
    syms,
    hess_colorvec, cons_jac_colorvec, cons_hess_colorvec,
    expr, cons_expr, f_kwargs)
    optfcall = (
        grad=grad, hess=hess, hv=hv,
        cons=cons, cons_j=cons_j, cons_h=cons_h,
        hess_prototype=hess_prototype, cons_jac_prototype=cons_jac_prototype, cons_hess_prototype=cons_hess_prototype,
        syms=syms,
        hess_colorvec=hess_colorvec, cons_jac_colorvec=cons_jac_colorvec, cons_hess_colorvec=cons_hess_colorvec,
        expr=expr, cons_expr=cons_expr
    )
    optfcall_cleaned = clean_named_tuple(optfcall)
    if !isnothing(optfcall_cleaned)
        if isnothing(f_kwargs)
            f = OptimizationFunction(negloglik, adtype; optfcall_cleaned...)
        else
            f = OptimizationFunction(negloglik, adtype; optfcall_cleaned..., f_kwargs...)
        end
    else
        if isnothing(f_kwargs)
            f = OptimizationFunction(negloglik, adtype)
        else
            f = OptimizationFunction(negloglik, adtype; f_kwargs...)
        end
    end
    return f
end

"""
construct_optimisation_problem(f, θ₀, data,
    lb, ub, 
    lcons, ucons, 
    sense, 
    prob_kwargs)

Constructs an optimisation problem. See [`OptimizationProblem`](@ref).
"""
@inline function construct_optimisation_problem(f, θ₀, data,
    lb, ub,
    lcons, ucons,
    sense,
    prob_kwargs)
    optprobcall = (
        lb=lb, ub=ub,
        lcons=lcons, ucons=ucons,
        sense=sense
    )
    optprobcall_cleaned = clean_named_tuple(optprobcall)
    if !isnothing(optprobcall_cleaned)
        if isnothing(prob_kwargs)
            prob = OptimizationProblem(f, θ₀, data; optprobcall_cleaned...)
        else
            prob = OptimizationProblem(f, θ₀, data; optprobcall_cleaned..., prob_kwargs...)
        end
    else
        if isnothing(prob_kwargs)
            prob = OptimizationProblem(f, θ₀, data)
        else
            prob = OptimizationProblem(f, θ₀, data; prob_kwargs...)
        end
    end
    return prob
end

"""
    LikelihoodProblem{ST,iip,F,θType,P,B,LC,UC,S,K,D,θ₀Type,ℓ<:Function} <: AbstractLikelihoodProblem

A struct that is used to define a likelihood problem. 

# Fields 
- `prob::OptimizationProblem{iip,F,θType,P,B,LC,UC,S,K}`

The optimisation problem, with `OptimizationProblem` coming from `Optimization.jl`. 

!!! warning "Objective function"

    Note that `OptimizationProblem` requires a function that is being minimised, so if you use this 
    struct directly with a provided `OptimizationProblem`, ensure that you have taken the negative of your 
    (log-)likelihood function in the objective function, but keep `loglik` as documented below.
    See also [`negate_loglik`](@ref).

- `data::D`

The data provided for the optimization problem, i.e. the argument `p` in `loglik(θ, p)` (see below).
- `loglik::F`

This is the log-likelihood function. Do not apply a negative to this function, it is done by the 
constructor. The function should take arguments `(θ, p)`, where the 
`θ` are the parameters to optimise, and `p` are known parameters. If you are defining a 
problem whose likelihood is a function of an ODE, it should instead 
take arguments `(θ, p, integrator)`; see the constructors below for more information. See also 
[`negate_loglik`](@ref).
- `θ₀::θ₀Type`

The `θ₀` parameter, used as the initial guess in the optimiser.
- `names::ST`

Names to use for plotting the variables. This is different from `syms` in `OptimizationFunction`.

# Constructors 

## Methods 

We provide two types of constructors: one for a standard likelihood problem, and one where the 
likelihood function is a function of a solution to an ODE. The signatures are listed below, followed 
by more detail.

    [1] LikelihoodProblem(loglik::F, num_params::Integer; <keyword arguments>) where {F<:Function}
    [2] LikelihoodProblem(loglik::F, num_params::Integer, integrator::Ti; <keyword arguments>) where {F<:Function,Ti<:SciMLBase.AbstractODEIntegrator}
    [3] LikelihoodProblem(loglik::F, num_params::Integer, ode_function::ODEf, u₀::u0, tspan::TS, ode_p::P, times::T; <keyword arguments>) where {F<:Function,T,ODEf,u0,TS,P}
    [4] LikelihoodProblem(; prob, loglik, θ₀, names)

The first method is for standard likelihood problems (e.g. for multiple linear regression), the second 
method is for likelihood problems that depend on the solution to an ODE (and whose `integrator` has 
already been defined), and the third method uses the function for an ODE, an initial condition, a timespan, known 
parameters, and times to call the solution at to define an `integrator` and call the second method. Finally, 
the last method is defined so that you can use `remake`, e.g. `remake(likprob; prob = optprob)`.

## Arguments 

We now give a detailed list for the arguments above.

- `loglik::F`: This is the log-likelihood function, same as above.
- `num_params::Integer`: The number of parameters being optimised over.
- `integrator::Ti`: The ODE integrator. See the [`DifferentialEquations.jl`] documentation for more information about this integrator interface. See also [`setup_integrator`](@ref). It is assumed that this integrator has the correct information about the initial condition, time span, the known parameters, and has a callback indicating which times to save the problem at.

If no integrator has been provided, then the third method above allows for the following additional arguments to be given in place 
of `integrator`, using [`setup_integrator`](@ref) to construct the `integrator`.

- `ode_function::ODEf`: The function for the `ODEProblem`, taking the form `(du, u, p, t)` or `(u, p, t)`. See the `DifferentialEquations.jl` docs for more information.
- `u₀::u0`: The initial condition for the `ODEProblem`.
- `tspan::TS`: The time span to solve the ODEs over.
- `ode_p`: The parameters `p` for the known parameters, to be used in `ode_function`.
- `times::T`: The times to return the solution ta.

## Keyword Arguments 

There are many keyword arguments. We give these in groups below.

### `OptimizationProblem` Keyword Arguments 

- `θ₀=nothing`: This is the initial guess for the maximum. If `isnothing(θ₀)`, then this initial guess is decided based on `lb` and `ub`. See [`choose_θ₀`](@ref).
- `data=SciMLBase.NullParameters()`: These are the known parameters, typically the data, to use for `loglik`.
- `lb=nothing`: Lower bounds for the parameters (or `nothing` if there are none).
- `ub=nothing`: Upper bounds for the parameters (or `nothing` if there are none).
- `lcons, ucons, sense, prob_kwargs`: See the documentation for `OptimizationProblem` at http://optimization.sciml.ai/stable/API/optimization_problem/. 

See also [`construct_optimisation_problem`](@ref).

### `OptimizationFunction` Keyword Arguments 

We include options for all the keyword arguments for `OptimizationFunction`. We use 
their same defaults, with additional keyword arguments given by `f_kwargs`. The only difference is that for `adtype`, our 
default is `AutoForwardDiff()` rather than `NoAD()`. See also [`construct_optimisation_function`](@ref).

See the documentation for `OptimizationFunction` http://optimization.sciml.ai/stable/API/optimization_function/

### Naming

- `names=[L"\\theta_%\$i" for i in 1:num_params]`: Names that are used when plotting results.

### ODE-Specific Keyword Arguments 

The third constructor also allows for the following two keyword arguments.

- `ode_alg=nothing`: Algorithm to use for solving the ODEs. If `nothing`, one is chosen automatically.
- `ode_kwargs=nothing`: Additional keyword arguments for the integrator interface.
"""
Base.@kwdef struct LikelihoodProblem{ST,iip,F,θType,P,B,LC,UC,S,K,D,θ₀Type,ℓ<:Function} <: AbstractLikelihoodProblem
    prob::OptimizationProblem{iip,F,θType,P,B,LC,UC,S,K}
    data::D
    loglik::ℓ
    θ₀::θ₀Type
    names::ST
end
function LikelihoodProblem(
    loglik::F, num_params::Integer;
    names=[L"\theta_%$i" for i in 1:num_params],
    θ₀=nothing, data=SciMLBase.NullParameters(),
    lb=nothing, ub=nothing,
    lcons=nothing, ucons=nothing,
    sense=nothing,
    adtype::SciMLBase.AbstractADType=Optimization.AutoForwardDiff(),
    grad=nothing, hess=nothing, hv=nothing,
    cons=nothing, cons_j=nothing, cons_h=nothing,
    hess_prototype=nothing, cons_jac_prototype=nothing, cons_hess_prototype=nothing,
    syms=nothing,
    hess_colorvec=nothing, cons_jac_colorvec=nothing, cons_hess_colorvec=nothing,
    expr=nothing, cons_expr=nothing,
    f_kwargs=nothing, prob_kwargs=nothing
) where {F<:Function}
    if !isnothing(lb) && num_params ≠ length(lb)
        error("The length of `lb` must equal `num_params = $num_params`.")
    end
    if !isnothing(ub) && num_params ≠ length(ub)
        error("The length of `ub` must equal `num_params = $num_params`.")
    end
    if !isnothing(θ₀) && length(θ₀) ≠ num_params
        error("The length of `θ₀` must equal `num_params = $num_params`.")
    end
    θ = choose_θ₀(θ₀, lb, ub, num_params)
    negloglik = negate_loglik(loglik)
    f = construct_optimisation_function(negloglik, adtype,
        grad, hess, hv,
        cons, cons_j, cons_h,
        hess_prototype, cons_jac_prototype, cons_hess_prototype,
        syms,
        hess_colorvec, cons_jac_colorvec, cons_hess_colorvec,
        expr, cons_expr, f_kwargs)
    prob = construct_optimisation_problem(f, θ, data,
        lb, ub,
        lcons, ucons,
        sense,
        prob_kwargs)
    return LikelihoodProblem{typeof(names),typeof(data),typeof(prob).parameters...,typeof(θ),typeof(loglik)}(prob, data, loglik, θ, names)
end
function LikelihoodProblem(loglik::F, num_params::Integer, integrator::Ti;
    adtype::SciMLBase.AbstractADType=Optimization.AutoFiniteDiff(), kwargs...) where {F<:Function,Ti<:SciMLBase.AbstractODEIntegrator}
    return LikelihoodProblem((θ, p) -> loglik(θ, p, integrator), num_params; adtype, kwargs...)
end
function LikelihoodProblem(loglik::F, num_params::Integer,
    ode_function::ODEf, u₀::u0, tspan::TS, ode_p::P, times::T;
    ode_alg=nothing, ode_kwargs=nothing,
    adtype::SciMLBase.AbstractADType=Optimization.AutoFiniteDiff(), kwargs...) where {F<:Function,T,ODEf,u0,TS,P}
    if isnothing(ode_kwargs)
        integrator = setup_integrator(ode_function, u₀, tspan, ode_p, times, ode_alg)
    else
        integrator = setup_integrator(ode_function, u₀, tspan, ode_p, times, ode_alg; ode_kwargs...)
    end
    return LikelihoodProblem(loglik, num_params, integrator; adtype, kwargs...)
end
