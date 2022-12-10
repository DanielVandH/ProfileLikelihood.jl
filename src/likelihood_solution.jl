"""
    struct LikelihoodSolution{Θ,P,M,R,A} <: AbstractLikelihoodSolution
    
Struct for a solution to a [`LikelihoodProblem`](@ref).

# Fields 
- `mle::Θ`: The MLEs.
- `problem::P`: The [`LikelihoodProblem`](@ref).
- `optimiser::A`: The algorithm used for solving the optimisation problem. 
- `maximum::M`: The maximum likelihood. 
- `retcode::R`: The `SciMLBase.ReturnCode`.
"""
Base.@kwdef struct LikelihoodSolution{Θ,P,M,R,A} <: AbstractLikelihoodSolution
    mle::Θ
    problem::P
    optimiser::A
    maximum::M
    retcode::R
end
function LikelihoodSolution(sol::SciMLBase.OptimizationSolution, prob::AbstractLikelihoodProblem; alg=sol.alg)
    return LikelihoodSolution(sol.u, prob, alg, -sol.minimum, sol.retcode)
end
