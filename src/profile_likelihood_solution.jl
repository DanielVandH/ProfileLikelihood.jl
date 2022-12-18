"""
    ProfileLikelihoodSolution{I,V,LP,LS,Spl,CT,CF,OM} 

Struct for the normalised profile log-likelihood. See [`profile`](@ref) for a constructor.

# Fields 
- `θ::Dict{I, V}`: This is a dictionary such that `θ[i]` gives the parameter values used for the normalised profile log-likelihood of the `i`th variable.
- `profile::Dict{I, V}`: This is a dictionary such that `profile[i]` gives the values of the normalised profile log-likelihood function at the corresponding values in `θ[i]`.
- `prob::LP`: The original [`LikelihoodProblem`](@ref).
- `mle::LS`: The solution to the full problem.
- `spline::Dict{I, Spl}`: This is a dictionary such that `spline[i]` is a spline through the data `(θ[i], profile[i])`. This spline can be evaluated at a point `ψ` for the `i`th variable by calling an instance of the struct with arguments `(ψ, i)`. See also [`spline_profile`](@ref).
- `confidence_intervals::Dict{I, Tuple{T, T}}`: This is a dictonary such that `confidence_intervals[i]` is a confidence interval for the `i`th parameter.
- `other_mles::OM`: This is a dictionary such that `other_mles[i]` gives the vector for the MLEs of the other parameters not being profiled, for each datum.

# Spline evaluation 

This struct is callable. We define the method 

    (prof::ProfileLikelihoodSolution)(θ, i)

that evaluates the spline through the `i`th profile at the point `θ`.
"""
Base.@kwdef struct ProfileLikelihoodSolution{I,V,LP,LS,Spl,CT,CF,OM} 
    parameter_values::Dict{I,V}
    profile_values::Dict{I,V}
    likelihood_problem::LP
    likelihood_solution::LS
    splines::Dict{I,Spl}
    confidence_intervals::Dict{I,ConfidenceInterval{CT,CF}}
    other_mles::OM
end

get_parameter_values(prof::ProfileLikelihoodSolution) = prof.parameter_values 
get_parameter_values(prof::ProfileLikelihoodSolution, i) = get_parameter_values(prof)[i]
get_parameter_values(prof::ProfileLikelihoodSolution, sym::Symbol) = get_parameter_values(prof, SciMLBase.sym_to_index(sym, prof))
get_profile_values(prof::ProfileLikelihoodSolution) = prof.profile_values
get_profile_values(prof::ProfileLikelihoodSolution, i) = get_profile_values(prof)[i]
get_profile_values(prof::ProfileLikelihoodSolution, sym::Symbol) = get_profile_values(prof, SciMLBase.sym_to_index(sym, prof))
get_likelihood_problem(prof::ProfileLikelihoodSolution) = prof.likelihood_problem
get_likelihood_solution(prof::ProfileLikelihoodSolution) = prof.likelihood_solution
get_splines(prof::ProfileLikelihoodSolution) = prof.splines 
get_splines(prof::ProfileLikelihoodSolution, i) = get_splines(prof)[i] 
get_splines(prof::ProfileLikelihoodSolution, sym::Symbol) = get_splines(prof, SciMLBase.sym_to_index(sym, prof)) 
get_confidence_intervals(prof::ProfileLikelihoodSolution) = prof.confidence_intervals
get_confidence_intervals(prof::ProfileLikelihoodSolution, i) = get_confidence_intervals(prof)[i]
get_confidence_intervals(prof::ProfileLikelihoodSolution, sym::Symbol) = get_confidence_intervals(prof, SciMLBase.sym_to_index(sym, prof))
get_other_mles(prof::ProfileLikelihoodSolution) = prof.other_mles
get_other_mles(prof::ProfileLikelihoodSolution, i) = get_other_mles(prof)[i]
get_other_mles(prof::ProfileLikelihoodSolution, sym::Symbol) = get_other_mles(prof, SciMLBase.sym_to_index(sym, prof))
get_syms(prof::ProfileLikelihoodSolution) = get_syms(get_likelihood_problem(prof))
get_syms(prof::ProfileLikelihoodSolution, i) = get_syms(prof)[i]
profiled_parameters(prof::ProfileLikelihoodSolution) = (sort ∘ collect ∘ keys ∘ get_confidence_intervals)(prof)::Vector{Int64}
number_of_profiled_parameters(prof::ProfileLikelihoodSolution) = length(profiled_parameters(prof))

struct ProfileLikelihoodSolutionView{N,PLS} 
    parent::PLS
end

Base.getindex(prof::PLS, i::Integer) where {PLS<:ProfileLikelihoodSolution} = ProfileLikelihoodSolutionView{i,PLS}(prof)
SciMLBase.sym_to_index(sym, prof::ProfileLikelihoodSolution) = SciMLBase.sym_to_index(sym, get_syms(prof))
function Base.getindex(prof::ProfileLikelihoodSolution, sym)
    if SciMLBase.issymbollike(sym)
        i = SciMLBase.sym_to_index(sym, prof)
    else
        i = sym
    end
    if i === nothing
        throw(BoundsError(prof, sym))
    else
        prof[i]
    end
end

get_parent(prof::ProfileLikelihoodSolutionView) = prof.parent 
get_index(::ProfileLikelihoodSolutionView{N,PLS}) where {N,PLS} = N
get_parameter_values(prof::ProfileLikelihoodSolutionView{N,PLS}) where {N,PLS} = get_parameter_values(get_parent(prof), N)
get_parameter_values(prof::ProfileLikelihoodSolutionView{N,PLS}, i)  where {N,PLS}= get_parameter_values(prof)[i]
get_profile_values(prof::ProfileLikelihoodSolutionView{N,PLS})  where {N,PLS}= get_profile_values(get_parent(prof), N)
get_profile_values(prof::ProfileLikelihoodSolutionView{N,PLS}, i) where {N,PLS} = get_profile_values(prof)[i]
get_likelihood_problem(prof::ProfileLikelihoodSolutionView{N,PLS}) where {N,PLS} = get_likelihood_problem(get_parent(prof))
get_likelihood_solution(prof::ProfileLikelihoodSolutionView{N,PLS}) where {N,PLS} = get_likelihood_solution(get_parent(prof))
get_splines(prof::ProfileLikelihoodSolutionView{N,PLS})  where {N,PLS}= get_splines(get_parent(prof), N)
get_confidence_intervals(prof::ProfileLikelihoodSolutionView{N,PLS}) where {N,PLS} = get_confidence_intervals(get_parent(prof), N)
get_confidence_intervals(prof::ProfileLikelihoodSolutionView{N,PLS}, i)  where {N,PLS}= get_confidence_intervals(prof)[i]
get_other_mles(prof::ProfileLikelihoodSolutionView{N,PLS})  where {N,PLS}= get_other_mles(get_parent(prof), N)
get_other_mles(prof::ProfileLikelihoodSolutionView{N,PLS}, i) where {N,PLS} = get_other_mles(prof)[i]
get_syms(prof::ProfileLikelihoodSolutionView{N,PLS})  where {N,PLS}= get_syms(get_parent(prof), N)

eval_spline(prof::ProfileLikelihoodSolutionView, θ) = get_splines(prof)(θ)
(prof::ProfileLikelihoodSolutionView)(θ) = eval_spline(prof, θ)
eval_spline(prof::ProfileLikelihoodSolution, θ, i) = prof[i](θ)
(prof::ProfileLikelihoodSolution)(θ, i) = prof[i](θ)