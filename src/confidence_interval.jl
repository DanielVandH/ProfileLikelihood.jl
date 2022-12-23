"""
    struct ConfidenceInterval{T, F}

Struct representing a confidence interval. 

# Fields 
- `lower::T`: The lower bound of the confidence interval. 
- `upper::T`: The upper bound of the confidence interval. 
- `level::F`: The level of the confidence interval.
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
