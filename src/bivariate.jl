function construct_profile_grids(n::NTuple{N,NTuple{2,I}}, sol, lower_bounds, upper_bounds, resolutions) where {N,I}
    grids = Dict{NTuple{2,I},FusedRegularGrid{
        2,
        Vector{number_type(lower_bounds)},
        Vector{I},
        Vector{number_type(lower_bounds)},
        number_type(lower_bounds),
        Vector{number_type(lower_bounds)}
    }}()
    res = get_resolution_tuples(resolutions, number_of_parameters(sol))
    for (n₁, n₂) in n
        if any(isinf, lower_bounds[[n₁, n₂]]) || any(isinf, upper_bounds[[n₁, n₂]])
            throw("The provided parameter bounds for $n₁ and $n₂ must be finite.")
        end
        grids[(n₁, n₂)] = FusedRegularGrid(
            [lower_bounds[n₁], lower_bounds[n₂]],
            [upper_bounds[n₁], upper_bounds[n₂]],
            [sol[n₁], sol[n₂]],
            maximum([res[n₁], res[n₂]])
        )
    end
    return grids
end

function layer_iterator(fused_grid::FusedRegularGrid{2,B,R,S,T}, layer_number) where {B,R,S,T}
    res = get_resolutions(get_negative_grid(fused_grid))
end