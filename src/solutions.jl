"""
    abstract type AbstractLikelihoodSolution

Abstract type for the solution to a likelihood problem. See `subtypes(AbstractLikelihoodSolution)`. The main 
reason to have this separate struct is to avoid type piracy.
"""
abstract type AbstractLikelihoodSolution end

for op in (:num_params, :data, :lower_bounds, :upper_bounds, :sym_names, :(Base.names)) 
    @eval @inline $op(sol::AbstractLikelihoodSolution) = $op(sol.prob)
end
@inline Base.maximum(sol::AbstractLikelihoodSolution) = sol.maximum 
@inline mle(sol::AbstractLikelihoodSolution) = sol.θ 
@inline retcode(sol::AbstractLikelihoodSolution) = sol.retcode

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

    LikelihoodSolution(sol::SciMLBase.AbstractOptimizationSolution, prob::LikelihoodProblem)

that constructs a [`LikelihoodSolution`](@ref) struct from a solution from `Optimization.jl`.
"""
Base.@kwdef struct LikelihoodSolution{θType₁,θType₂,θType₃,A,Tf,O,ST,iip,F,P,B,LC,UC,S,K,D,ℓ<:Function} <: AbstractLikelihoodSolution
    θ::θType₃
    prob::LikelihoodProblem{ST,iip,F,θType₁,P,B,LC,UC,S,K,D,θType₂,ℓ}
    alg::A
    maximum::Tf
    retcode::Symbol
    original::O
end
@inline function LikelihoodSolution(sol::T, prob::LikelihoodProblem{ST,iip,F,θType₁,P,B,LC,UC,S,K,D,θType₂,ℓ}; alg::A) where {T<:SciMLBase.AbstractOptimizationSolution,ST,iip,F,θType₁,θType₂,P,B,LC,UC,S,K,D,ℓ,A}
    return LikelihoodSolution{θType₁,θType₂,typeof(sol.u),A,typeof(sol.minimum),typeof(sol.original),ST,iip,F,P,B,LC,UC,S,K,D,ℓ}(sol.u, prob, alg, -sol.minimum, sol.retcode, sol.original)
end

"""
    ProfileLikelihoodSolution{I,V,LP<:AbstractLikelihoodProblem,LS<:AbstractLikelihoodSolution,Spl<:Spline1D,CT,CF} <: AbstractLikelihoodSolution

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

This is a dictionary such that `spline[i]` is a spline through the data `(θ[i], profile[i])`. This spline can be evaluated at a point `ψ` for the `i`th variable by calling an instance of the struct with arguments `(ψ, i)`.
- `confidence_intervals::Dict{I, Tuple{T, T}}`

This is a dictonary such that `confidence_intervals[i]` is a confidence interval for the `i`th parameter.

# Spline evaluation 

This struct is callable. We define the method 

    (prof::ProfileLikelihoodSolution)(θ, i)

that evaluates the spline through the `i`th profile at the point `θ`.
"""
Base.@kwdef struct ProfileLikelihoodSolution{I,V,LP<:AbstractLikelihoodProblem,LS<:AbstractLikelihoodSolution,Spl<:Spline1D,CT,CF} <: AbstractLikelihoodSolution
    θ::Dict{I, V}
    profile::Dict{I, V}
    prob::LP
    mle::LS
    spline::Dict{I, Spl}
    confidence_intervals::Dict{I, ConfidenceInterval{CT, CF}}
end
(prof::ProfileLikelihoodSolution)(θ, i) = prof.spline[i](θ) 

@inline Base.maximum(sol::ProfileLikelihoodSolution) = maximum(sol.mle)
@inline mle(sol::ProfileLikelihoodSolution) = mle(sol.mle)
@inline confidence_intervals(sol::ProfileLikelihoodSolution) = sol.confidence_intervals
@inline confidence_intervals(sol::ProfileLikelihoodSolution, i)  = confidence_intervals(sol)[i]
@inline retcode(sol::ProfileLikelihoodSolution)  = retcode(sol.mle)