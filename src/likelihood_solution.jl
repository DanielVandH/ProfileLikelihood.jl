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
Base.@kwdef struct LikelihoodSolution{N,Θ,P,M,R,A} <: AbstractLikelihoodSolution{N,P}
    mle::Θ
    problem::P
    optimiser::A
    maximum::M
    retcode::R
end
function LikelihoodSolution(sol::SciMLBase.OptimizationSolution, prob::AbstractLikelihoodProblem; alg=sol.alg)
    return LikelihoodSolution{number_of_parameters(prob),typeof(sol.u),typeof(prob),
        typeof(-sol.objective),typeof(sol.retcode),typeof(alg)}(sol.u, prob, alg, -sol.objective, sol.retcode)
end