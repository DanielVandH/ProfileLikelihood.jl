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
    refine(sol::LikelihoodSolution, args...; method = :tiktak, kwargs...)

Given a [`LikelihoodSolution`](@ref) `sol` (from [`mle`](@ref), say), refines the result using [`refine_tiktak`](@ref)
or [`refine_lhc`](@ref).

# Arguments 
- `sol::LikelihoodSolution`: A [`LikelihoodSolution`](@ref).
- `args...`: Extra positional arguments that get passed into the refinement method; see [`refine_tiktak`](@ref) or [`refine_lhc`](@ref) for the options.

# Keyword Arguments
- `method = :tiktak`: Method to use. Can be one of `tiktak` or `lhc`.
- `kwargs...`: Extra keyword arguments that get passed into the refinement method; see [`refine_tiktak`](@ref) or [`refine_lhc`](@ref) for the options.

# Output 
The output is another [`LikelihoodSolution`](@ref) with the refined results.
"""
function refine(sol::LikelihoodSolution, args...; method = :tiktak, kwargs...)
    likprob = sol.prob 
    !finite_bounds(likprob) && error("Problem must have finite lower/upper bounds to use refinement methods. Consider using `remake` to choose new bounds on the problem.")
    optprob = likprob.prob
    prob_newu0 = remake(optprob; u0=sol.θ)
    new_likprob = remake(likprob; prob=prob_newu0)
    if method == :tiktak
        return refine_tiktak(new_likprob, args...; kwargs...)
    elseif method == :lhc
        return refine_lhc(new_likprob, args...; kwargs...)
    else 
        throw(ArgumentError("The provided method, $method, is invalid. It must be one of `:tiktak` or `:lhc`. See `?refine`."))
    end
end

"""
    refine_tiktak(prob::OptimizationProblem, args...; n = 100, local_method = NLopt.LN_NELDERMEAD(), use_threads = false, kwargs...)
    refine_tiktak(prob::LikelihoodProblem, args...; n = 100, local_method = NLopt.LN_NELDERMEAD(), use_threads = false, kwargs...)

Optimises the given problem `prob` using the `TikTak` method from `MultistartOptimization.jl`.

See also [`mle`](@ref), [`refine`](@ref), and [`refine_lhc`](@ref).

# Arguments 
- `prob`: The [`LikelihoodProblem`](@ref) or `OptimizationProblem`.
- `args...`: Extra positional arguments that get passed to `Optimization.solve`.

# Keyword Arguments 
- `n = 100`: Number of Sobol points.
- `local_method = NLopt.LN_NELDERMEAD()`: Local method to use in the multistart method.
- `use_threads = false`: Whether to use multithreading.
- `kwargs...`: Extra keyword arguments that get passed to `Optimization.solve`.

# Output 
The output is a new `OptimizationSolution` or [`LikelihoodSolution`](@ref).
"""
function refine_tiktak end 
function refine_tiktak(prob::OptimizationProblem, args...; n = 100, local_method = NLopt.LN_NELDERMEAD(), use_threads = false, kwargs...)
    solve(prob, MultistartOptimization.TikTak(n), local_method, args...; use_threads = use_threads, kwargs...)
end
function refine_tiktak(prob::LikelihoodProblem, args...; n = 100, local_method = NLopt.LN_NELDERMEAD(), use_threads = false, kwargs...)
    sol = refine_tiktak(prob.prob, args...; n, local_method, use_threads, kwargs...) 
    LikelihoodSolution(sol, prob; sol.alg)
end

"""
    refine_lhc(prob::OptimizationProblem, alg, args...; n = 25, gens = 1000, kwargs...)
    refine_lhc(prob::LikelihoodProblem, alg, args...; n = 25, gens = 1000, kwargs...)

Optimises the given problem `prob` by solving at many points specified by sampling from a Latin hypercube.

See also [`mle`](@ref), [`refine`](@ref), and [`refine_tiktak`](@ref).

# Arguments 
- `prob`: The [`LikelihoodProblem`](@ref) or `OptimizationProblem`.
- `alg`: Algorithm to use for optimising.
- `args...`: Extra positional arguments that get passed to `Optimization.solve`.

# Keyword Arguments 
- `n = 25`: Number of points to sample from the hypercube.
- `gens = 1000`: Generations to use; see [`get_lhc_params`](@ref).
- `kwargs...`: Extra keyword arguments that get passed to `Optimization.solve`.

# Output 
The output is a new `OptimizationSolution` or [`LikelihoodSolution`](@ref).
"""
function refine_lhc end 
function refine_lhc(prob::OptimizationProblem, alg, args...; n = 25, gens = 1000, use_threads = false, kwargs...)
    if n == 1
        throw(ArgumentError("The provided number of restarts, $n, must be greater than 1."))
    end
    new_params = get_lhc_params(prob, n, gens; use_threads = use_threads)
    opt_sol = solve(prob, alg, args...; kwargs...)
    min_obj = opt_sol.minimum
    for j in 1:n 
        new_prob = @views remake(prob; u0 = new_params[:, j])
        new_sol = solve(new_prob, alg, args...; kwargs...)
        if new_sol.minimum < min_obj 
            opt_sol = new_sol 
            min_obj = new_sol.minimum
        end
    end
    return opt_sol
end
function refine_lhc(prob::LikelihoodProblem, alg = NLopt.LN_NELDERMEAD, args...; n = 25, gens = 1000,  use_threads = false, kwargs...)
    sol = refine_lhc(prob.prob, alg, args...; n, gens, use_threads, kwargs...)
    LikelihoodSolution(sol, prob; alg)
end
