module ProfileLikelihoodDelaunayTriangulationExt

using ProfileLikelihood
@static if isdefined(Base, :get_extension)
    using DelaunayTriangulation
else
    using ..DelaunayTriangulation
end

@inline threshold_intersection(τ, uᵢ, uⱼ) = (τ - uᵢ) / (uⱼ - uᵢ)
@inline threshold_intersection_exists(τ, uᵢ, uⱼ) = (uᵢ < τ && uⱼ > τ) || (uᵢ > τ && uⱼ < τ)
function ProfileLikelihood._get_confidence_regions_delaunay!(confidence_regions, n, range_1::AbstractArray{T}, range_2, profile_values, threshold, conf_level) where {T}
    grid_xy = vec([(x, y) for x in range_1, y in range_2])
    tri = DelaunayTriangulation.triangulate(grid_xy)
    conf_contour = NTuple{2,T}[]
    DelaunayTriangulation.delete_boundary_vertices_from_graph!(tri)
    for (u, v) in DelaunayTriangulation.get_edges(tri)
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
    confidence_regions[n] = ProfileLikelihood.ConfidenceRegion(getindex.(conf_contour, 1), getindex.(conf_contour, 2), conf_level)
    return nothing
end

end