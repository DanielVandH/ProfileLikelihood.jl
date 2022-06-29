"""
    mle(prob::LikelihoodProblem[, alg = PolyOpt()], args...; kwargs...) 

Computes the maximum likelihood estimates for the given [`LikelihoodProblem`](@ref).

## Arguments 
- `prob::LikelihoodProblem`: The [`LikelihoodProblem`](@ref).
- `alg = PolyOpt()`: The solver to use from `Optimization.jl`.
- `args...`: Arguments for `Optimization.solve`. 

## Keyword Arguments 
- `kwargs...`: Keyword arguments for `Optimization.solve`.

## Output 
The output is a [`LikelihoodSolution`](@ref) struct. See [`LikelihoodSolution`](@ref) for more details. 
"""
@inline function mle(prob::LikelihoodProblem, alg=PolyOpt(), args...; kwargs...) 
    sol = solve(prob.prob, alg, args...; kwargs...)
    return LikelihoodSolution(sol, prob; alg)
end
