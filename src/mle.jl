"""
    mle(prob::LikelihoodProblem, alg, args...; kwargs...)
    mle(prob::LikelihoodProblem, alg::Tuple, args...; kwargs...)

Given the likelihood problem `prob` and an optimiser `alg`, finds the MLEs and returns a 
[`LikelihoodSolution`](@ref) object. Extra arguments and keyword arguments for `solve` can be passed 
through `args...` and `kwargs...`.

If `alg` is a `Tuple`, then the problem is re-optimised after each algorithm with the next element in alg, 
starting from `alg[1]`, with initial estimate coming from the solution with the 
previous algorithm (starting with `get_initial_estimate(prob)`).
"""
function mle(prob::LikelihoodProblem, alg, args...; kwargs...)
    opt_prob = get_problem(prob)
    opt_sol = solve(opt_prob, alg, args...; kwargs...)
    return LikelihoodSolution(opt_sol, prob)
end
function mle(prob::LikelihoodProblem, alg::Tuple, args...; kwargs...)
    opt_prob = get_problem(prob)
    opt_sol = solve(opt_prob, alg[begin], args...; kwargs...)
    for i in 2:length(alg)
        if any(isnan, opt_sol.u)
            @warn "Encountered a NaN in the MLE solution, $(opt_sol.u), for the algorithm $(alg[i-1]). Ignoring its solution."
            updated_problem = opt_prob
        else
            updated_problem = update_initial_estimate(opt_prob, opt_sol)
        end
        _alg = alg[i]
        opt_sol = solve(updated_problem, _alg, args...; kwargs...)
    end
    return LikelihoodSolution(opt_sol, prob; alg=alg)
end