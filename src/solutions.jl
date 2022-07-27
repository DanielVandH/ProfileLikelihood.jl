"""
    abstract type AbstractLikelihoodSolution

Abstract type for the solution to a likelihood problem. See `subtypes(AbstractLikelihoodSolution)`. The main 
reason to have this separate struct is to avoid type piracy.
"""
abstract type AbstractLikelihoodSolution end

for op in (:num_params, :data, :sym_names, :(Base.names))
    @eval @inline $op(sol::AbstractLikelihoodSolution) = $op(sol.prob)
end
@inline lower_bounds(sol::AbstractLikelihoodSolution; make_open = false) = lower_bounds(sol.prob; make_open)
@inline upper_bounds(sol::AbstractLikelihoodSolution; make_open = false) = upper_bounds(sol.prob; make_open)
@inline Base.maximum(sol::AbstractLikelihoodSolution) = sol.maximum
@inline mle(sol::AbstractLikelihoodSolution) = sol.θ
@inline retcode(sol::AbstractLikelihoodSolution) = sol.retcode

Base.@propagate_inbounds @inline function Base.getindex(sol::AbstractLikelihoodSolution, i::Int)
    sol.θ[i]
end
SciMLBase.sym_to_index(sym, sol::AbstractLikelihoodSolution) = SciMLBase.sym_to_index(sym, sol.prob.prob.f.syms)
Base.@propagate_inbounds @inline function Base.getindex(sol::AbstractLikelihoodSolution, sym)
    if SciMLBase.issymbollike(sym)
        i = SciMLBase.sym_to_index(sym, sol)
    else
        i = sym
    end
    if i === nothing
        throw(BoundsError(sol, sym))
    else
        sol[i]
    end
end

"""
    LikelihoodSolution{θType₁,θType₂,θType₃,A,Tf,O,ST,iip,F,P,B,LC,UC,S,K,ℓ<:Function} <: AbstractLikelihoodSolution <: AbstractLikelihoodSolution

Struct for a solution to a `LikelihoodProblem`. This is directly modified from 
`SciMLBase.OptimizationSolution`, noting that we are finding maxima rather than 
minima.

# Fields 
- `θ::θType₃` 

The maximum likelihood estimate (MLE).
- `prob::LikelihoodProblem{ST,iip,F,θType₁,P,B,LC,UC,S,K,θType₂,ℓ}`

The [`LikelihoodProblem`](@ref).
- `alg::A`

The algorithm used for solving the problem.
- `maximum::Tf`

The value of the log-likelihood function at the MLE.
- `retcode::Symbol`

The return code. For more details, see the return code section of the `DifferentialEquations.jl` documentation.
- `original::O`

If the solver is wrapped from an alternative solver ecosystem, such as `Optim.jl`, then this is the original return from said solver library.

# Constructor 
We define the constructor 

    LikelihoodSolution(sol::SciMLBase.AbstractOptimizationSolution, prob::LikelihoodProblem; scale=true, alg::A)

that constructs a [`LikelihoodSolution`](@ref) struct from a solution from `Optimization.jl`. `scale` is the amount to divide the final 
maximum likelihood by, and `alg` is the algorithm used.
"""
Base.@kwdef struct LikelihoodSolution{θType₁,θType₂,θType₃,A,Tf,O,ST,iip,F,P,B,LC,UC,S,K,D,ℓ<:Function} <: AbstractLikelihoodSolution
    θ::θType₃
    prob::LikelihoodProblem{ST,iip,F,θType₁,P,B,LC,UC,S,K,D,θType₂,ℓ}
    alg::A
    maximum::Tf
    retcode::Symbol
    original::O
end
@inline function LikelihoodSolution(sol::T, prob::LikelihoodProblem{ST,iip,F,θType₁,P,B,LC,UC,S,K,D,θType₂,ℓ}; scale=true, alg::A) where {T<:SciMLBase.AbstractOptimizationSolution,ST,iip,F,θType₁,θType₂,P,B,LC,UC,S,K,D,ℓ,A}
    return LikelihoodSolution{θType₁,θType₂,typeof(sol.u),A,typeof(sol.minimum),typeof(sol.original),ST,iip,F,P,B,LC,UC,S,K,D,ℓ}(sol.u, prob, alg, -sol.minimum / scale, sol.retcode, sol.original)
end

"""
    ProfileLikelihoodSolution{I,V,LP<:AbstractLikelihoodProblem,LS<:AbstractLikelihoodSolution,Spl,CT,CF,OM} <: AbstractLikelihoodSolution

Struct for the normalised profile log-likelihood. See [`profile`](@ref) for a constructor.

# Fields 
- `θ::Dict{I, V}`

This is a dictionary such that `θ[i]` gives the parameter values used for the normalised profile log-likelihood of the `i`th variable.
- `profile::Dict{I, V}`

This is a dictionary such that `profile[i]` gives the values of the normalised profile log-likelihood function at the corresponding values in `θ[i]`.
- `prob::LP`

The original [`LikelihoodProblem`](@ref).
- `mle::LS`

The solution to the full problem.
- `spline::Dict{I, Spl}`

This is a dictionary such that `spline[i]` is a spline through the data `(θ[i], profile[i])`. This spline can be evaluated at a point `ψ` for the `i`th variable by calling an instance of the struct with arguments `(ψ, i)`. See also [`spline_profile`](@ref).
- `confidence_intervals::Dict{I, Tuple{T, T}}`

This is a dictonary such that `confidence_intervals[i]` is a confidence interval for the `i`th parameter.

- `other_mles::OM`

This is a dictionary such that `other_mles[i]` gives the vector for the MLEs of the other parameters not being profiled, for each datum.
# Spline evaluation 

This struct is callable. We define the method 

    (prof::ProfileLikelihoodSolution)(θ, i)

that evaluates the spline through the `i`th profile at the point `θ`.
"""
Base.@kwdef struct ProfileLikelihoodSolution{I,V,LP<:AbstractLikelihoodProblem,LS<:AbstractLikelihoodSolution,Spl,CT,CF,OM} <: AbstractLikelihoodSolution
    θ::Dict{I,V}
    profile::Dict{I,V}
    prob::LP
    mle::LS
    spline::Dict{I,Spl}
    confidence_intervals::Dict{I,ConfidenceInterval{CT,CF}}
    other_mles::OM
end
(prof::ProfileLikelihoodSolution)(θ, i) = prof.spline[i](θ)

@inline Base.maximum(sol::ProfileLikelihoodSolution) = maximum(sol.mle)
@inline mle(sol::ProfileLikelihoodSolution) = mle(sol.mle)
@inline confidence_intervals(sol::ProfileLikelihoodSolution) = sol.confidence_intervals
@inline confidence_intervals(sol::ProfileLikelihoodSolution, i) = confidence_intervals(sol)[i]
@inline retcode(sol::ProfileLikelihoodSolution) = retcode(sol.mle)

Base.@kwdef struct ProfileLikelihoodSolutionView{I,PLS,V,LP,LS,Spl,CT,CF,OM} <: AbstractLikelihoodSolution
    parent::PLS
    θ::V
    profile::V
    prob::LP
    mle::LS
    spline::Spl
    confidence_intervals::ConfidenceInterval{CT,CF}
    other_mles::OM
end

Base.@propagate_inbounds @inline function Base.getindex(sol::ProfileLikelihoodSolution, i::Int)
    ProfileLikelihoodSolutionView{i,typeof(sol),promote_type(typeof(sol.θ[i]), typeof(sol.profile[i])),
        typeof(sol.prob),typeof(sol.mle[i]),typeof(sol.spline[i]),
        typeof(sol.confidence_intervals[i].lower),typeof(sol.confidence_intervals[i].upper),typeof(sol.other_mles[i])}(sol,
        sol.θ[i], sol.profile[i], sol.prob, sol.mle[i], sol.spline[i], sol.confidence_intervals[i], sol.other_mles[i])
end
(prof::ProfileLikelihoodSolutionView)(θ) = prof.spline(θ)
