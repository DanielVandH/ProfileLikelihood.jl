module ProfileLikelihood

using SciMLBase
using InvertedIndices
using FunctionWrappers
using PreallocationTools
using StatsFuns
using SimpleNonlinearSolve
using Interpolations
using Requires
using Printf

include("utils.jl")
include("problem_updates.jl")
include("abstract_type_definitions.jl")
include("likelihood_problem.jl")
include("likelihood_solution.jl")
include("confidence_interval.jl")
include("profile_likelihood_solution.jl")
include("mle.jl")
include("grid_search.jl")
include("profile_likelihood.jl")
include("display.jl")

export LikelihoodProblem
export mle
export GridSearch
export grid_search
export RegularGrid
export IrregularGrid
export profile
export construct_profile_ranges
export get_confidence_intervals
export gaussian_loglikelihood
export update_initial_estimate

function __init__()
    @require CairoMakie = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0" begin
        @require LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f" begin
            include("plotting.jl")
            export plot_profiles
        end
    end
end

end
# new line