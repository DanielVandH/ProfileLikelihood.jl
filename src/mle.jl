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

"""
    refine(sol::LikelihoodSolution, args...; n = 100, local_method = NLopt.LN_NELDERMEAD(), kwargs...)

Given a [`LikelihoodSolution`](@ref) `sol` (from [`mle`](@ref), say), refines the result using 
a multi-start method.

# Arguments 
- `sol::LikelihoodSolution`: A [`LikelihoodSolution`](@ref).
- `args...`: Extra arguments to use in [`mle`](@ref).

# Keyword Arguments 
- `n = 100`: Number of Sobol points to use in [`MultistartOptimization.TikTak`](@ref).
- `local_method = NLopt.LN_NELDERMEAD()`: Local optimiser to use in [`MultistartOptimization.TikTak`](@ref).
- `kwargs...`: Extra keyword arguments to use in [`mle`](@ref).

# Output 
The output is another [`LikelihoodSolution`](@ref) with the refined results.
"""
function refine(sol::LikelihoodSolution, args...; n=100, local_method=NLopt.LN_NELDERMEAD(), kwargs...)
    likprob = sol.prob
    !finite_bounds(likprob) && error("Problem must have finite lower/upper bounds to use multistart methods. Consider using `remake` to choose new bounds on the problem.")
    optprob = likprob.prob
    prob_newu0 = remake(optprob; u0=sol.θ)
    new_likprob = remake(likprob; prob=prob_newu0)
    return mle(new_likprob, MultistartOptimization.TikTak(n), local_method, args...; kwargs...)
end
