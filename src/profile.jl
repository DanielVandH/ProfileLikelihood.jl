"""
    prepare_profile(prob::OptimizationProblem{iip,F,θType,P,B,LC,UC,S,K}, θₘₗₑ, max_steps, i) where {iip,F,θType,P,B,LC,UC,S,K}

Prepares the cache arrays and `prob` for profiling.

# Arguments
- `prob`: The `OptimizationProblem`.
- `Θₘₗₑ`: The MLEs.
- `max_steps`: The maximum number of steps to take in either direction.
- `i`: The variable being profiled.

# Outputs 
- `prob`: This is the `OptimizationProblem`, except now updated so that the lower and upper bounds exclude the ith variable and also has an initial guess with the updated dimensions.
- `param_vals`: This is the vector which will store the parameter values for the `i`th variable.
- `profile_vals`: This is the vector which will store the values of the normalised profile log-likelihood function at the corresponding values in `param_vals`.
- `cache`: Cache array for storing the `i`th variable along with the other variables. See its use in [`construct_new_f`](@ref).
- `θ₀`: This is the vector which will continually be updated with the current initial guess.
"""
function prepare_profile(prob::OptimizationProblem{iip,F,θType,P,B,LC,UC,S,K}, θₘₗₑ, max_steps, i) where {iip,F,θType,P,B,LC,UC,S,K}
    param_vals = θType([])
    profile_vals = Vector{Float64}([])
    cache = dualcache(zeros(length(θₘₗₑ)))
    θ₀ = θₘₗₑ[Not(i)]
    sizehint!(param_vals, 2max_steps)
    sizehint!(profile_vals, 2max_steps)
    prob = update_prob(prob, i) # Update the lower and upper bounds to exclude i     
    prob = update_prob(prob, θ₀) # This is replaced immediately, but just helps the problem start with the correct dimensions 
    return prob, param_vals, profile_vals, cache, θ₀
end

"""
    profile(prob::OptimizationProblem, θₘₗₑ, ℓₘₐₓ, i, min_steps, max_steps, threshold, Δθ, alg) 
    profile(prob::LikelihoodProblem, sol::LikelihoodSolution, i; <keyword arguments)
    profile(prob::LikelihoodProblem[, sol::LikelihoodSolution=mle(prob, alg)]; <keyword argments>)

Computes the normalised profile log-likelihood function for the given `LikelihoodProblem` for all variables, or only for 
the `i`th variable if `i` is provided.

# Arguments 
- `prob`: The [`LikelihoodProblem`](@ref).
- `sol=mle(prob, alg)]`: The solution to `prob`, given in a [`LikelihoodSolution`](@ref) struct, where `alg` is one of the keyword arguments below. The [`OptimizationProblem`](@ref) method uses this structure to define 
    - `θₘₗₑ`: The MLEs, `θₘₗₑ = mle(sol)`.
    - `ℓₘₐₓ`: The log-likelihood function evaluated at the MLEs, `ℓₘₐₓ = maximum(sol)`.
- `i`: The variable to profile.

The remaining arguments for the first method, the one with the [`OptimizationProblem`](@ref), are those in the 
following keyword arguments. 

# Keyword Arguments 
- `min_steps = 15`: Minimum number of steps. If the process terminates before this number is reached, then the function is computed at equidistant points (with `min_steps` points) between the extrema of the used points.
- `max_steps = 100`: Maximum number of steps to take before terminating. Will return an error if this number of steps is reached.
- `Δθ = abs.(mle(sol) / 100)` or `Δθ=abs(mle(sol)[i] / 100)`: Value to increment the `i`th value by, computing `θᵢ = param_vals[end] + Δθ[i]`.
- `alg = PolyOpt()`: Algorithm to use for optimising `prob`.
- `conf_level = 0.99`: The confidence level for the confidence intervals.
- `threshold = -0.5quantile(Chisq(1), conf_level)`: When the normalised profile log-likelihood function drops below this `threshold`, stop.
- `spline=true` (exclusive to the last method): Whether a spline should be used for computing the confidence intervals. See [`confidence_intervals`](@ref).

# Outputs 
If `i` is provided, then we return 

- `profile_vals`: Values for the normalised profile log-likelihood function.
- `param_vals`: The values for the `i`th parameter that correspond to the values in `profile_vals`.

Otherwise, returns a [`ProfileLikelihoodSolution`](@ref) struct.
"""
function profile end
@doc (@doc profile) function profile(prob::OptimizationProblem, θₘₗₑ, ℓₘₐₓ, i, min_steps, max_steps, threshold, Δθ, alg) 
    prob, param_vals, profile_vals, cache, θ₀ = prepare_profile(prob, θₘₗₑ, max_steps, i)
    ## Start at the MLE 
    profile!(prob, ℓₘₐₓ, i, θₘₗₑ[i], param_vals, profile_vals, θ₀, cache; alg)
    ## Start by going left 
    find_endpoint!(prob, profile_vals, threshold, min_steps, max_steps, ℓₘₐₓ, i, param_vals, θ₀, cache, alg, -Δθ)
    ## Now go right
    reverse!(param_vals) # This is needed so that we can use [end]
    reverse!(profile_vals)
    find_endpoint!(prob, profile_vals, threshold, min_steps, max_steps, ℓₘₐₓ, i, param_vals, θ₀, cache, alg, Δθ)
    ## Sort the final results 
    sort_idx = sortperm(param_vals)
    permute!(param_vals, sort_idx)
    permute!(profile_vals, sort_idx)
    ## Sometimes we get duplicate indices. Need to remove them, and need to make sure length(param_vals)=length(profile_vals)
    idx = unique(i -> param_vals[i], eachindex(param_vals))
    keepat!(param_vals, idx)
    keepat!(profile_vals, idx)
    return profile_vals, param_vals
end
@doc (@doc profile) function profile(prob::LikelihoodProblem, sol::LikelihoodSolution, i;
    min_steps=15, max_steps=100, Δθ=abs(mle(sol)[i] / 100), alg=PolyOpt(),
    conf_level=0.99, threshold=-0.5quantile(Chisq(1), conf_level))
    return profile(prob.prob, sol.θ, sol.maximum, i, min_steps, max_steps, threshold, Δθ, alg)
end
@doc (@doc profile) function profile(prob::LikelihoodProblem{ST,iip,F,θType,P,B,LC,UC,S,K,θ₀Type,ℓ}, sol::LikelihoodSolution=mle(prob, PolyOpt());
    min_steps=15, max_steps=100, Δθ=abs.(mle(sol) / 100), alg=PolyOpt(),
    conf_level=0.99, spline=true, threshold=-0.5quantile(Chisq(1), conf_level)) where {ST,iip,F,θType,P,B,LC,UC,S,K,θ₀Type,ℓ<:Function}
    N = num_params(prob)
    if Δθ isa Number
        Δθ = repeat([Δθ], N)
    end
    θ = Dict{Int64,θType}([])
    prof = Dict{Int64,Vector{Float64}}([])
    splines = Vector{Spline1D}([])
    sizehint!(θ, N)
    sizehint!(prof, N)
    sizehint!(splines, N)
    for n in 1:N
        profile_vals, param_vals = profile(prob, sol, n; min_steps, max_steps, threshold, Δθ=Δθ[n], alg)
        θ[n] = param_vals
        prof[n] = profile_vals
        push!(splines, Spline1D(param_vals, profile_vals))
    end
    splines = Dict(1:N .=> splines) # We define the Dict here rather than above to make sure we get the types right
    conf_ints = confidence_intervals(θ, prof; conf_level, spline)
    profile_sol = ProfileLikelihoodSolution(θ, prof, prob, sol, splines, conf_ints)
    return profile_sol
end
