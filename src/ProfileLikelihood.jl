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

    @require DelaunayTriangulation = "927a84f5-c5f4-47a5-9785-b46e178433df" begin
        @inline threshold_intersection(τ, uᵢ, uⱼ) = (τ - uᵢ) / (uⱼ - uᵢ)
        @inline threshold_intersection_exists(τ, uᵢ, uⱼ) = (uᵢ < τ && uⱼ > τ) || (uᵢ > τ && uⱼ < τ)
        function _get_confidence_regions_delaunay!(confidence_regions, n, range_1::AbstractArray{T}, range_2, profile_values, threshold, conf_level) where {T}
            grid_xy = vec([(x, y) for x in range_1, y in range_2])
            tri = DelaunayTriangulation.triangulate_bowyer(grid_xy)
            conf_contour = NTuple{2,T}[]
            DG = DelaunayTriangulation.get_graph(tri)
            DelaunayTriangulation.delete_point!(DG, DelaunayTriangulation.BoundaryIndex)
            for (u, v) in DelaunayTriangulation.edges(DG)
                u₁, u₂ = profile_values[u], profile_values[v]
                if threshold_intersection_exists(threshold, u₁, u₂)
                    p₁ = grid_xy[u]
                    p₂ = grid_xy[v]
                    t = threshold_intersection(threshold, u₁, u₂)
                    p = @. p₁ + t * (p₂ - p₁)
                    push!(conf_contour, Tuple(p))
                end
            end
            θ = zeros(length(conf_contour))
            for j in eachindex(conf_contour)
                x, y = conf_contour[j]
                θ[j] = atan(y - range_2[0], x - range_1[0])
            end
            sort_idx = sortperm(θ)
            permute!(conf_contour, sort_idx)
            confidence_regions[n] = ConfidenceRegion(getindex.(conf_contour, 1), getindex.(conf_contour, 2), conf_level)
            return nothing
        end
    end
end

end
# new line