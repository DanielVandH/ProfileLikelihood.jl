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
Base.@kwdef struct ConfidenceInterval{T,F}
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

@inline Base.@propagate_inbounds Base.iterate(CI::ConfidenceInterval, state=1) = state > 2 ? nothing : (CI[state], state + 1)
@inline Base.eltype(::ConfidenceInterval{T,F}) where {T,F} = T
@inline Base.length(::ConfidenceInterval) = 2

"""
    itp_roots(itp)

Finds the roots of the interpolant `itp`, assumed to come from a normalised profile log-likelihood minus the threshold.
"""
function itp_roots(itp)
    ## Left value 
    left_bracket = (itp.knots[begin], itp.knots[end ÷ 2]) # MLE is at end ÷ 2
    ℓ = find_zero(itp, left_bracket, Bisection())
    ## Right value 
    right_bracket = (itp.knots[end ÷ 2], itp.knots[end])
    u = find_zero(itp, right_bracket, Bisection())
    return [ℓ, u]
end

"""
    confidence_intervals(θ, prof, i; conf_level=0.99, spline=true, spline_alg = FritschCarlsonMonotonicInterpolation, extrap = Throw)
    confidence_intervals(θ, prof, i; conf_level=0.99, spline=true, spline_alg = FritschCarlsonMonotonicInterpolation, extrap = Throw)

Computes a confidence interval for given normalised profile log-likelihood values. If `i` is provided, 
computes the interval for the `i`th variable, otherwise computes all of them.

## Arguments 
- `θ`: Parameter values.
- `prof`: Values of the normalised profile log-likelihood function corresponding to the parameter values in `θ`.
- `i`: The parameter that the interval is being found for; `θ` and `prof` above are indexed accordingly.

## Keyword Arguments 
- `conf_level = 0.99`: The confidence level for the interval.
- `spline = true`: Whether a spline should be used for finding the interval endpoints.
- `spline_alg=FritschCarlsonMonotonicInterpolation`: The spline algorithm.
- `extrap=Throw`: The extrapolation algorithm.

## Output 
- `CI`: A [`ConfidenceInterval`](@ref) with the lower and upper bounds. If confidence intervals were computed for all variables, this is instead a dictionary of such objects.
"""
function confidence_intervals end
function confidence_intervals(θ, prof, i; conf_level=0.99, spline=true, spline_alg=FritschCarlsonMonotonicInterpolation, extrap=Throw)
    conf_val = -0.5quantile(Chisq(1), conf_level)
    if spline
        itp = spline_profile(θ[i], prof[i] .- conf_val; alg=spline_alg, extrap)
        try
            ℓ, u = itp_roots(itp.itp) # itp is an Extrapolation, so itp.itp is the interpolant
            res = ConfidenceInterval(ℓ, u, conf_level)
            return res
        catch
            @warn("Failed to find a valid confidence interval for parameter $i. Attempting to find confidence interval without using a spline.")
            res = confidence_intervals(θ, prof, i; conf_level, spline=false)
            return res
        end
    else
        conf_region = θ[i] .≥ conf_val
        idx = findall(conf_region)
        ab = extrema(θ[i][idx])
        try
            res = ConfidenceInterval(ab..., conf_level)
            return res
        catch
            @warn("Failed to find a valid confidence interval for parameter $i. Returning the extrema of the parameter values.")
            res = ConfidenceInterval(extrema(θ)..., conf_level)
            return res
        end
    end
end
function confidence_intervals(θ, prof; conf_level=0.99, spline=true, spline_alg=FritschCarlsonMonotonicInterpolation, extrap=Throw)
    conf_ints = Dict{Int64,ConfidenceInterval{eltype(θ[1]),typeof(conf_level)}}([])
    sizehint!(conf_ints, length(θ))
    for n in eachindex(θ)
        conf_ints[n] = confidence_intervals(θ, prof, n; conf_level, spline, spline_alg, extrap)
    end
    return conf_ints
end

