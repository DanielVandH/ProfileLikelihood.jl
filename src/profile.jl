"""
    prepare_profile(prob::OptimizationProblem, θₘₗₑ, n, param_ranges) 

Prepares the cache arrays and `prob` for profiling.

# Arguments
- `prob`: The `OptimizationProblem`.
- `Θₘₗₑ`: The MLEs.
- `n`: The variable being profiled.
- `param_ranges`: A 2-tuple whose first entry gives the ranges for the profile going left, and second for the right.

# Outputs 
- `prob`: This is the `OptimizationProblem`, except now updated so that the lower and upper bounds exclude the `n`th variable and also has an initial guess with the updated dimensions.
- `left`/`right_profile`: This is the vector which will store the values of the normalised profile log-likelihood function at the corresponding values in `left`/`right_param_vals`.
- `left`/`right_param_vals`: This is the vector which will store the parameter values for the `n`th variable going `left`/`right`.
- `cache`: Cache array for storing the `i`th variable along with the other variables. See its use in [`construct_new_f`](@ref).
- `θ₀`: This is the vector which will continually be updated with the current initial guess.
- `combined_profiles`: Vector that combines the normalised profile log-likelihood function values going left and right.
- `combined_param_vals`: Vector that combines the parameter values going left and right.
"""
function prepare_profile(prob::OptimizationProblem, θₘₗₑ, n, param_ranges)
    left_profiles = Vector{Float64}([])
    right_profiles = Vector{Float64}([])
    left_param_vals = Vector{Float64}([])
    right_param_vals = Vector{Float64}([])
    combined_profiles = Vector{Float64}([])
    combined_param_vals = Vector{Float64}([])
    sizehint!(left_profiles, length(param_ranges[1]))
    sizehint!(right_profiles, length(param_ranges[2]))
    sizehint!(combined_profiles, length(param_ranges[1]) + length(param_ranges[2]))
    sizehint!(combined_param_vals, length(param_ranges[1]) + length(param_ranges[2]))
    cache = dualcache(zeros(length(θₘₗₑ)))
    θ₀ = θₘₗₑ[Not(n)]
    prob = update_prob(prob, n) # Update the lower and upper bounds to exclude n     
    prob = update_prob(prob, θ₀) # This is replaced immediately, but just helps the problem start with the correct dimensions 
    return prob, left_profiles, right_profiles, left_param_vals, right_param_vals, cache, θ₀, combined_profiles, combined_param_vals
end

"""
    construct_profile_ranges(prob::LikelihoodProblem, sol::LikelihoodSolution, resolution::Int; param_bounds = bounds(prob))

Construct ranges for each parameter in `prob` to use for profiling.

# Arguments 
- `prob::LikelihoodProblem`: The [`LikelihoodProblem`](@ref).
- `sol::LikelihoodSolution`: The MLEs for `prob`, see [`mle`](@ref).
- `resolution::Int`: How many points to use in each direction.

# Keyword Arguments 
- `param_bounds = bounds(prob)`: The bounds for each parameter, given as a 2-vector (or a 2-tuple).

# Output 
The output is a `Vector{NTuple{2, LinRange{Float64, Int64}}}` of length `num_params(prob)`, with the `i`th entry 
giving the values to use a tuple, with the first tuple being the values used when going to the left of the MLE, with 
the first entry the MLE, and similarly for the second tuple.
"""
function construct_profile_ranges(prob::LikelihoodProblem, sol::LikelihoodSolution, resolution; param_bounds=bounds(prob))
    if resolution isa Number
        resolution = repeat([resolution], num_params(prob))
    end
    param_ranges = Vector{NTuple{2,LinRange{Float64,Int64}}}(undef, num_params(prob))
    mles = mle(sol)
    for i in eachindex(mles)
        θᵢ = mles[i]
        ℓ, u = param_bounds[i]
        if ⊻(isinf(ℓ), isinf(u))
            error("The provided parameter bounds must be finite.")
        end
        left_range = LinRange(θᵢ, ℓ, resolution[i])
        right_range = LinRange(θᵢ, u, resolution[i])
        param_ranges[i] = (left_range, right_range)
    end
    param_ranges
end

"""
    profile(prob::OptimizationProblem, θₘₗₑ, ℓₘₐₓ, n, alg, threshold, param_ranges; kwargs...)
    profile(prob::LikelihoodProblem, sol::LikelihoodSolution, n, alg, threshold, param_ranges; kwargs...)
    profile!(prob::LikelihoodProblem, sol::LikelihoodSolution, n, alg, threshold, param_ranges, profile_vals, param_vals, splines; kwargs...)
    profile(prob::LikelihoodProblem{ST,iip,F,θType,P,B,LC,UC,S,K,D,θ₀Type,ℓ}, sol::LikelihoodSolution; <keyword arguments>)

Computes the normalised profile log-likelihood function for the given `LikelihoodProblem` for all variables, or only for 
the `n`th variable if `n` is provided.

# Arguments 
- `prob`: The [`LikelihoodProblem`](@ref) or `OptimizationProblem`.
- `sol`: The [`LikelihoodSolution`](@ref); see also [`mle`](@ref).
- `n`: The parameter to profile.
- `alg`: The algorithm to use for optimising.
- `threshold`: When to stop profiling.
- `param_ranges`: Parameter ranges for the `n`th parameter; see [`construct_profile_ranges`](@ref) (and below).
- `θₘₗₑ`: The maximum likelihood estimates, `mle(sol)`.
- `ℓₘₐₓ`: The maximum log-likelihood, `maximum(sol)`.
- `splines`: Dictionary holding the splines of the data.

The third method also has keyword arguments, given below.

# Keyword Arguments 
- `alg`: The algorithm to use for optimising.
- `conf_level = 0.95`: The confidence level for the confidence intervals.
- `spline = true`: Whether a spline should be used for computing the confidence intervals. See [`confidence_intervals`](@ref).
- `threshold = -0.5quantile(Chisq(1), conf_level)`: When the normalised profile log-likelihood function drops below this `threshold`, stop.
- `resolution = 200`: Number of gridpoints to use in each direction in `param_ranges` (if the default is used).
- `param_ranges = construct_profile_ranges(prob, sol, resolution)`: A `Vector{NTuple{2, LinRange{Float64, Int64}}}` of length `num_params(prob)`, with the `i`th entry 
giving the values to use a tuple, with the first tuple being the values used when going to the left of the MLE, with 
the first entry the MLE, and similarly for the second tuple. See [`construct_profile_ranges`](@ref). 
- `use_threads = false`: Whether to profile all parameters using multithreading. This thread only applies to the profiling of all parameters, i.e. you could profile 
multiple parameters at the same time, but the individual computations within a single parameter's profile are performed serially. (Doesn't actually do anything currently.)
- `kwargs...`: Extra keyword arguments to pass to the optimisers.

# Outputs 
If `n` is provided, then we return 

- `profile_vals`: Values for the normalised profile log-likelihood function.
- `param_vals`: The values for the `i`th parameter that correspond to the values in `profile_vals`.

Otherwise, returns a [`ProfileLikelihoodSolution`](@ref) struct.
"""
function profile end
function profile(prob::OptimizationProblem, θₘₗₑ, ℓₘₐₓ, n, alg, threshold, param_ranges; kwargs...)
    prob, left_profiles, right_profiles,
    left_param_vals, right_param_vals,
    cache, θ₀, combined_profiles, combined_param_vals = prepare_profile(prob, θₘₗₑ, n, param_ranges)
    find_endpoint!(prob, left_param_vals, left_profiles, copy(θ₀), ℓₘₐₓ, n, alg, threshold, cache, param_ranges[1]; kwargs...) # Go left
    find_endpoint!(prob, right_param_vals, right_profiles, copy(θ₀), ℓₘₐₓ, n, alg, threshold, cache, param_ranges[2]; kwargs...) # Go right
    append!(combined_profiles, left_profiles, right_profiles) # Now combine the profiles 
    append!(combined_param_vals, left_param_vals, right_param_vals) # Now combine the parameter values 
    # Make sure the results are sorted 
    sort_idx = sortperm(combined_param_vals)
    permute!(combined_param_vals, sort_idx)
    permute!(combined_profiles, sort_idx)
    # Some cleanup duplicate indices
    idx = unique(i -> combined_param_vals[i], eachindex(combined_param_vals))
    keepat!(combined_param_vals, idx)
    keepat!(combined_profiles, idx)
    return combined_profiles, combined_param_vals
end
@inline function profile(prob::LikelihoodProblem, sol::LikelihoodSolution, n, alg, threshold, param_ranges; kwargs...)
    return profile(prob.prob, mle(sol), maximum(sol), n, alg, threshold, param_ranges; kwargs...)
end
@inline function profile!(prob::LikelihoodProblem, sol::LikelihoodSolution, n, alg, threshold, param_ranges, profile_vals, param_vals, splines; kwargs...)
    _prof, _θ = profile(prob, sol, n, alg, threshold, param_ranges[n]; kwargs...)
    param_vals[n] = _θ
    profile_vals[n] = _prof
    try
        splines[n] = Spline1D(_θ, _prof)
    catch
        error("Error creating the spline. Try increasing the grid resolution for parameter $n.")
    end
    return nothing
end
function profile(prob::LikelihoodProblem, sol::LikelihoodSolution;
    alg=sol.alg,
    conf_level=0.95,
    spline=true,
    threshold=-0.5quantile(Chisq(1), conf_level),
    resolution=200,
    param_ranges=construct_profile_ranges(prob, sol, resolution),
    use_threads=false,
    kwargs...)
    N = num_params(prob)
    θ = Dict{Int64,Vector{Float64}}([])
    prof = Dict{Int64,Vector{Float64}}([])
    splines = Vector{Spline1D}(undef, N)
    sizehint!(θ, N)
    sizehint!(prof, N)
    #if use_threads
    #    Threads.@threads for n in 1:N
    #        profile!(prob, sol, n, alg, threshold, param_ranges, prof, θ, splines)
    #    end
    #else
    for n in 1:N
        profile!(prob, sol, n, alg, threshold, param_ranges, prof, θ, splines; kwargs...)
    end
    #end
    splines = Dict(1:N .=> splines) # We define the Dict here rather than above to make sure we get the types right
    conf_ints = confidence_intervals(θ, prof; conf_level, spline)
    profile_sol = ProfileLikelihoodSolution(θ, prof, prob, sol, splines, conf_ints)
    return profile_sol
end

"""
    profile!(prob::OptimizationProblem, profile_vals, n, θₙ, θ₀, ℓₘₐₓ, alg, cache; kwargs...)

Computes the normalised profile log-likelihood function for the `n`th variable at `θₙ`.

# Arguments 
- `prob::OptimizationProblem`: The `OptimizationProblem`. This should be already updated from [`update_prob`](@ref).
- `profile_vals`: Values of the normalised profile log-likelihood function. Gets updated with the new value at the index `i`.
- `n`: The index of the parameter being profiled.
- `θₙ`: The value to fix the `n`th variable at. 
- `θ₀`: Initial guesses for the parameters to optimise over (i.e. initial guesses for the parameters without the `n`th variable). This gets updated in-place with the new guesses.
- `ℓₘₐₓ`: The value of the log-likelihood function at the MLEs.
- `alg`: Algorithm to use for optimising `prob`.
- `cache`: Cache array for storing the `n`th variable along with the other variables. See its use in [`construct_new_f`](@ref).

# Keyword Arguments
- `kwargs...`: Extra keyword arguments to pass to the optimisers.

# Output 
There is no output, but `profile_vals` gets updated (via `push!`) with the new normalised profile log-likelihood value at `θₙ`.
"""
function profile!(prob::OptimizationProblem, profile_vals, n, θₙ, θ₀, ℓₘₐₓ, alg, cache; kwargs...)
    prob = update_prob(prob, n, θₙ, cache, θ₀) # Update the objective function and initial guess 
    soln = solve(prob, alg; kwargs...)
    for j in eachindex(θ₀)
        θ₀[j] = soln.u[j]
    end
    push!(profile_vals, -soln.minimum - ℓₘₐₓ)
    return nothing
end

"""
    find_endpoint!(prob::OptimizationProblem, param_vals, profile_vals, θ₀, ℓₘₐₓ, n, alg, threshold, cache, param_ranges; kwargs...)

Optimises the profile likelihood until exceeding `threshold`, going in the direction specified by `Δθ`.

# Arguments 
- `prob`: The `OptimizationProblem`. This should be already updated from [`update_prob`](@ref).
- `param_vals`: Parameter values to use for the normalised profile log-likelihood function.
- `profile_vals`: Values of the normalised profile log-likelihood function at the corresponding values in `param_vals`.
- `θ₀`: Initial guesses for the parameters to optimise over (i.e. initial guesses for the parameters without the `n`th variable).
- `ℓₘₐₓ`: The value of the log-likelihood function at the MLEs.
- `n`: The variable being profiled.
- `alg`: Algorithm to use for optimising `prob`.
- `threshold`: When the normalised profile log-likelihood function drops below this `threshold`, stop.
- `cache`: Cache array for storing the `n`th variable along with the other variables. See its use in [`construct_new_f`](@ref).
- `param_ranges`: Values to try for profiling. See also [`construct_profile_ranges`](@ref).

# Keyword Arguments 
- `kwargs...`: Extra keyword arguments to pass to the optimisers.

# Outputs 
Returns nothing, but `profile_vals` gets updated with the found values and `param_vals` with the used parameter values.
"""
function find_endpoint!(prob::OptimizationProblem, param_vals, profile_vals, θ₀, ℓₘₐₓ, n, alg, threshold, cache, param_ranges; kwargs...)
    for θₙ in param_ranges
        push!(param_vals, θₙ)
        profile!(prob, profile_vals, n, θₙ, θ₀, ℓₘₐₓ, alg, cache; kwargs...)
        if profile_vals[end] ≤ threshold 
            return nothing 
        end
    end
    return nothing 
end