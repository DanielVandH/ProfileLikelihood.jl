"""
    ConfidenceInterval{T,F}

Struct for a confidence interval.

# Fields 
- `lower::T`

Lower bound of the confidence interval.

- `upper::T`

Upper bound of the confidence interval.

- `level::F`

Confidence level for the confidence interval.
"""
struct ConfidenceInterval{T,F}
    lower::T 
    upper::T 
    level::F 
end

"""
    lower(CI::ConfidenceInterval)
    upper(CI::ConfidenceInterval)

Returns the `lower` or `upper` bound of the confidence interval `CI`.
"""
@inline lower(CI::ConfidenceInterval) = CI.lower 
@doc (@doc lower) @inline upper(CI::ConfidenceInterval) = CI.upper 

"""
    level(CI::ConfidenceInterval)

Returns the confidence level of the confidence interval `CI`.
"""
@inline level(CI::ConfidenceInterval) = CI.level

"""
    bounds(ci::ConfidenceInterval) 

Return, as a `Tuple`, the lower and upper bounds of the confidence interval `ci` in the form `(L, U)`.
"""
@inline bounds(CI::ConfidenceInterval) = (lower(CI), upper(CI))

"""
    getindex(CI::ConfidenceInterval, i::Int) 

Indexes the given confidence interval `CI`. If `i==1` then the lower bound is returned, if 
`i == 2` then the upper bound is returned. Otherwise, a `BoundsError` is thrown.
"""
@inline Base.Base.@propagate_inbounds function Base.getindex(CI::ConfidenceInterval, i::Int)
    if i == 1 
        return lower(CI)
    elseif i == 2
        return upper(CI)
    else
        throw(BoundsError(CI, i))
    end
end

"""
    in(x, CI::ConfidenceInterval) 

Determines if `x` is in the confidence interval `CI`. Returns `true` if so and
`false` otherwise.
"""
Base.in(x, CI::ConfidenceInterval) = lower(CI) ≤ x ≤ upper(CI)

@inline Base.Base.@propagate_inbounds Base.iterate(CI::ConfidenceInterval, state=1) = state > 2 ? nothing : (CI[state], state + 1)
@inline Base.eltype(::ConfidenceInterval{T,F}) where {T,F} = T 
@inline Base.length(::ConfidenceInterval) = 2

"""
    confidence_intervals(θ, prof, i; conf_level = 0.99, spline = true)  
    confidence_intervals(θ, prof; conf_level = 0.99, spline = true)

Computes a confidence interval for given normalised profile log-likelihood values. If `i` is provided, 
computes the interval for the `i`th variable, otherwise computes all of them.

## Arguments 
- `θ`: Parameter values.
- `prof`: Values of the normalised profile log-likelihood function corresponding to the parameter values in `θ`.
- `i`: The parameter that the interval is being found for; `θ` and `prof` above are indexed accordingly.

## Keyword Arguments 
- `conf_level = 0.99`: The confidence level for the interval.
- `spline = true`: Whether a spline should be used for finding the interval endpoints.

## Output 
- `CI`: A [`ConfidenceInterval`](@ref) with the lower and upper bounds. If confidence intervals were computed for all variables, this is instead a dictionary of such objects.
"""
function confidence_intervals end
function confidence_intervals(θ, prof, i; conf_level = 0.99, spline = true)
    conf_val = -0.5quantile(Chisq(1), conf_level)
    if spline 
        itp = Spline1D(θ[i], prof[i] .- conf_val)
        ab = sort(roots(itp))
        return ConfidenceInterval(extrema(ab)..., conf_level)
    else
        conf_region = θ[i] .≥ conf_val 
        idx = findall(conf_region)
        ab = extrema(θ[i][idx])
        return ConfidenceInterval(ab..., conf_level)
    end
end
function confidence_intervals(θ, prof; conf_level = 0.99, spline = true)
    conf_ints = Dict{Int64, ConfidenceInterval{eltype(θ[1]), typeof(conf_level)}}([])
    sizehint!(conf_ints, length(θ))
    for n in eachindex(θ)
        conf_ints[n] = confidence_intervals(θ, prof, n; conf_level, spline)
    end
    return conf_ints
end

