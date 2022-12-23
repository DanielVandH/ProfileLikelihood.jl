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
            maximum([res[n₁], res[n₂]]);
            store_original_resolutions=true
        )
    end
    return grids
end

struct LayerIterator{N,B,T} # Don't use Iterators.flatten as it cannot infer when we use repeated (this was originally implemented as being a collection of `zip`s, e.g. the bottom row was `zip(-layer_number:layer_number, Iterators.repeated(-layer_number, 2layer_number+1))`, but this returns `Any` type)
    bottom::B
    right::B
    top::T
    left::T
    function LayerIterator(layer_number)
        itr1 = -layer_number:layer_number # UnitRanges
        itr2 = (-layer_number+1):layer_number 
        itr3 = (layer_number-1):-1:-layer_number # StepRanges
        itr4 = (layer_number-1):-1:(-layer_number+1)
        return new{layer_number,typeof(itr1),typeof(itr3)}(itr1, itr2, itr3, itr4)
    end
end
Base.eltype(::Type{LayerIterator{N,B,T}}) where {N,B,T} = NTuple{2,Int64}
Base.length(::LayerIterator{N,B,T}) where {N,B,T} = 8N
function Base.iterate(layer::LayerIterator{N,B,T}, state=1) where {N,B,T}
    if 1 ≤ state ≤ 2N + 1
        return ((layer.bottom[state], -N), state + 1)
    elseif 2N + 2 ≤ state ≤ 4N + 1
        return ((N, layer.right[state-2N-1]), state + 1)
    elseif 4N + 2 ≤ state ≤ 6N + 1
        return ((layer.top[state-4N-1], N), state + 1)
    elseif 6N + 2 ≤ state ≤ 8N
        return ((-N, layer.left[state-6N-1]), state + 1)
    else
        return nothing
    end
end