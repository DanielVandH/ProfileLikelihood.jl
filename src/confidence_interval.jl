"""
    struct ConfidenceInterval{T, F}

Struct representing a confidence interval. 

# Fields 
- `lower::T`

The lower bound of the confidence interval. 
- `upper::T`

The upper bound of the confidence interval. 
- `level::F`

The level of the confidence interval.
"""
Base.@kwdef struct ConfidenceInterval{T,F}
    lower::T
    upper::T
    level::F
end

get_lower(CI::ConfidenceInterval) = CI.lower
get_upper(CI::ConfidenceInterval) = CI.upper
get_level(CI::ConfidenceInterval) = CI.level
get_bounds(CI::ConfidenceInterval{T,F}) where {T,F} = NTuple{2,T}((get_lower(CI), get_upper(CI)))

@inline Base.firstindex(CI) = 1
@inline Base.lastindex(CI) = 2
@inline Base.@propagate_inbounds function Base.getindex(CI::ConfidenceInterval, i::Int)
    if i == 1
        return get_lower(CI)
    elseif i == 2
        return get_upper(CI)
    else
        throw(BoundsError(CI, i))
    end
end

@inline Base.length(CI::ConfidenceInterval) = get_upper(CI) - get_lower(CI)
@inline Base.iterate(CI::ConfidenceInterval, state=1) = state == 1 ? (get_lower(CI), 2) : (state == 2 ? (get_upper(CI), nothing) : nothing)

Base.in(x, CI::ConfidenceInterval) = get_lower(CI) ≤ x ≤ get_upper(CI)

"""
    struct ConfidenceRegion{T, F}

Struct representing a confidence region. 

# Fields 
- `x::T`

The `x`-coordinates for the region's boundary.
- `y::T`

The `y`-coordinates for the region's boundary.
- `level::F`

The level of the confidence region.
"""
Base.@kwdef struct ConfidenceRegion{T,F}
    x::T
    y::T
    level::F
end

get_x(CI::ConfidenceRegion) = CI.x
get_y(CI::ConfidenceRegion) = CI.y
get_level(CI::ConfidenceRegion) = CI.level

Base.length(CI::ConfidenceRegion) = length(get_x(CI))
Base.eltype(::Type{ConfidenceRegion{T,F}}) where {T,F} = NTuple{2,number_type(T)}
@inline Base.iterate(CI::ConfidenceRegion) = iterate(zip(get_x(CI), get_y(CI)))
@inline Base.iterate(CI::ConfidenceRegion, state) = iterate(zip(get_x(CI), get_y(CI)), state)

function get_nodes_and_edges(x, y)
    if x[end] ≠ x[1] || y[end] ≠ y[1]
        nodes = [x y]
        edges = zeros(Int64, length(x), 2)
        edges[:, 1] .= 1:length(x)
        edges[1:end-1, 2] .= 2:length(x)
        edges[end, 2] = 1
    else
        nodes = [x[1:end-1] y[1:end-1]]
        edges = zeros(Int64, length(x) - 1, 2)
        edges[:, 1] .= 1:length(x)-1
        edges[1:end-1, 2] .= 2:length(x)-1
        edges[end, 2] = 1
    end
    return nodes, edges
end

function Base.in(x, CR::ConfidenceRegion)
    conf_x = get_x(CR)
    conf_y = get_y(CR)
    nodes, edges = get_nodes_and_edges(conf_x, conf_y)
    if typeof(x) <: AbstractVector
        res = inpoly2(x, nodes, edges)
        return res[:, 1]
    else
        res = inpoly2([x], nodes, edges)
        return res[1, 1]
    end
end