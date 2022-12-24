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
using ChunkSplitters
using OffsetArrays
using Contour

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
export eval_prediction_function
export replace_profile!
export construct_profile_grids
export bivariate_profile

function __init__()
    @require CairoMakie = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0" begin
        @require LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f" begin
            include("plotting.jl")
            export plot_profiles
        end
    end

    @require Surrogates = "6fc51010-71bc-11e9-0e15-a3fcc6593c49" begin
        function _set_next_initial_estimate_radial!(sub_cache, combined_grid, other_mles, layer, I, grid)
            grid_data = @views combined_grid[(-layer+1):(layer-1), (-layer+1):(layer-1)]
            grid_values = @views other_mles[(-layer+1):(layer-1), (-layer+1):(layer-1)]
            lb = get_parameters(grid, (-layer, -layer))
            ub = get_parameters(grid, (layer, layer))
            basis = Surrogates.RadialBasis(grid_data, grid_values, lb, ub)
            sub_cache .= basis(combined_grid[I])
            return nothing
        end
    end
end

end
# new line