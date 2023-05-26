module ProfileLikelihood

using SciMLBase
using InvertedIndices
using FunctionWrappers
using PreallocationTools
using StatsFuns
using SimpleNonlinearSolve
using Interpolations
using Printf
using OffsetArrays
using PolygonInbounds
using Contour
using ChunkSplitters

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
include("prediction_intervals.jl")
include("bivariate.jl")

export LikelihoodProblem
export mle
export GridSearch
export grid_search
export RegularGrid
export IrregularGrid
export profile
export construct_profile_ranges
export get_confidence_intervals
export get_confidence_regions
export get_prediction_intervals
export gaussian_loglikelihood
export update_initial_estimate
export construct_integrator
export get_mle
export get_maximum
export get_lower_bounds
export get_upper_bounds
export get_x
export get_y
export get_parameter_values
export get_profile_values
export get_range
export replace_profile!
export construct_profile_grids
export bivariate_profile
export refine_profile!

function plot_profiles end
function plot_profiles! end
export plot_profiles
export plot_profiles!
function choose_grid_layout end
SciMLBase.sym_to_index(vars::Integer, prof::ProfileLikelihoodSolution) = vars

@static if !isdefined(Base, :get_extension)
    using Requires
end

@static if !isdefined(Base, :get_extension)
    function __init__()
        @require Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a" include("../ext/ProfileLikelihoodMakieExt.jl")
        @require DelaunayTriangulation = "927a84f5-c5f4-47a5-9785-b46e178433df" include("../ext/ProfileLikelihoodDelaunayTriangulationExt.jl")
    end
end



end # module ProfileLikelihood