"""
    abstract type AbstractLikelihoodProblem{N, L}

Abstract type of a likelihood problem, where `N` is the number of parameters and 
`L` is the type of the likelihood function.
"""
abstract type AbstractLikelihoodProblem{N, L} end

get_problem(prob::AbstractLikelihoodProblem) = prob.problem
get_data(prob::AbstractLikelihoodProblem) = prob.data
get_log_likelihood_function(prob::AbstractLikelihoodProblem) = prob.log_likelihood_function
get_θ₀(prob::AbstractLikelihoodProblem) = prob.θ₀
get_θ₀(prob::AbstractLikelihoodProblem, i) = prob.θ₀[i]
get_syms(prob::AbstractLikelihoodProblem) = prob.syms

get_lower_bounds(prob::OptimizationProblem) = prob.lb
get_upper_bounds(prob::OptimizationProblem) = prob.ub
get_lower_bounds(prob::OptimizationProblem, i) = prob.lb[i]
get_upper_bounds(prob::OptimizationProblem, i) = prob.ub[i]
get_lower_bounds(prob::AbstractLikelihoodProblem) = (get_lower_bounds ∘ get_problem)(prob)
get_upper_bounds(prob::AbstractLikelihoodProblem) = (get_upper_bounds ∘ get_problem)(prob)

has_lower_bounds(prob) = !isnothing(get_lower_bounds(prob))
has_upper_bounds(prob) = !isnothing(get_upper_bounds(prob))
has_bounds(prob) = has_lower_bounds(prob) && has_upper_bounds(prob)

finite_lower_bounds(prob) = has_lower_bounds(prob) && all(isfinite, get_lower_bounds(prob))
finite_upper_bounds(prob) = has_upper_bounds(prob) && all(isfinite, get_upper_bounds(prob))
finite_bounds(prob) = finite_lower_bounds(prob) && finite_upper_bounds(prob)

number_of_parameters(prob::OptimizationProblem) = length(prob.u0)
number_of_parameters(::AbstractLikelihoodProblem{N, L}) where {N, L} = N

Base.getindex(prob::AbstractLikelihoodProblem, i::Integer) = get_θ₀(prob, i)
SciMLBase.sym_to_index(sym, prob::AbstractLikelihoodProblem) = SciMLBase.sym_to_index(sym, get_syms(prob))
SciMLBase.sym_to_index(i::Integer, ::AbstractLikelihoodProblem) = i
function Base.getindex(prob::AbstractLikelihoodProblem, sym)
    if SciMLBase.issymbollike(sym)
        i = SciMLBase.sym_to_index(sym, prob)
    else
        i = sym
    end
    if i === nothing
        throw(BoundsError(get_θ₀(prob), sym))
    else
        get_θ₀(prob, i)
    end
end
function Base.getindex(prob::AbstractLikelihoodProblem, sym::AbstractVector{Symbol})
    idx = SciMLBase.sym_to_index.(sym, Ref(prob))
    return get_θ₀(prob, idx)
end

function parameter_is_inbounds(prob, θ)
    !has_bounds(prob) && return true
    for (i,lb,ub) in zip(eachindex(θ), get_lower_bounds(prob), get_upper_bounds(prob))
        (θ[i] < lb || θ[i] > ub) && return false 
    end
    return true
end

######################################################
## AbstractLikelihoodSolution
######################################################
"""
    abstract type AbstractLikelihoodSolution{N, P}

Type representing the solution to a likelihood problem, where `N` is the 
number of parameters and `P` is the type of the likelihood problem.
"""
abstract type AbstractLikelihoodSolution{N, P} end

get_mle(sol::AbstractLikelihoodSolution) = sol.mle
get_mle(sol::AbstractLikelihoodSolution, i) = sol.mle[i]
get_problem(sol::AbstractLikelihoodSolution) = sol.problem
get_optimiser(sol::AbstractLikelihoodSolution) = sol.optimiser
get_maximum(sol::AbstractLikelihoodSolution) = sol.maximum
get_retcode(sol::AbstractLikelihoodSolution) = sol.retcode
get_syms(sol::AbstractLikelihoodSolution) = get_syms(get_problem(sol))

number_of_parameters(::AbstractLikelihoodSolution{N,P}) where {N,P}=N

Base.getindex(sol::AbstractLikelihoodSolution, i::Integer) = get_mle(sol, i)
SciMLBase.sym_to_index(sym, sol::AbstractLikelihoodSolution) = SciMLBase.sym_to_index(sym, get_syms(sol))
function Base.getindex(sol::AbstractLikelihoodSolution, sym)
    if SciMLBase.issymbollike(sym)
        i = SciMLBase.sym_to_index(sym, sol)
    else
        i = sym
    end
    if i === nothing
        throw(BoundsError(get_mle(sol), sym))
    else
        get_mle(sol, i)
    end
end
function Base.getindex(sol::AbstractLikelihoodSolution, sym::AbstractVector{Symbol})
    idx = SciMLBase.sym_to_index.(sym, Ref(sol))
    return get_mle(sol, idx)
end

update_initial_estimate(prob::AbstractLikelihoodProblem, sol::AbstractLikelihoodSolution) = update_initial_estimate(prob, get_mle(sol))

######################################################
## AbstractGrid
######################################################
"""
    abstract type AbstractGrid{N,B,T}

Type representing a grid, where `N` is the number of parameters, `B` is the type for the 
bounds, and `T` is the number type.
"""
abstract type AbstractGrid{N,B,T} end

@inline get_lower_bounds(grid::AbstractGrid) = grid.lower_bounds
@inline get_lower_bounds(grid::AbstractGrid, i) = get_lower_bounds(grid)[i]
@inline get_upper_bounds(grid::AbstractGrid) = grid.upper_bounds
@inline get_upper_bounds(grid::AbstractGrid, i) = get_upper_bounds(grid)[i]

@inline finite_lower_bounds(grid::AbstractGrid) = all(isfinite, get_lower_bounds(grid))
@inline finite_upper_bounds(grid::AbstractGrid) = all(isfinite, get_upper_bounds(grid))
@inline finite_bounds(grid::AbstractGrid) = finite_lower_bounds(grid) && finite_upper_bounds(grid)

@inline number_type(::Type{A}) where {N,B,T,A<:AbstractGrid{N,B,T}} = T
@inline number_type(::A) where {N,B,T,A<:AbstractGrid{N,B,T}} = number_type(A)
@inline number_of_parameters(::Type{A}) where {N,B,T,A<:AbstractGrid{N,B,T}} = N
@inline number_of_parameters(::A) where {N,B,T,A<:AbstractGrid{N,B,T}} = number_of_parameters(A)
