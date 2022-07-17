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
- `left`/`right_other_mles`: This is the vector which will store the MLEs for the other parameters for the `n`th variable going `left`/`right`.
- `cache`: Cache array for storing the `i`th variable along with the other variables. See its use in [`construct_new_f`](@ref).
- `θ₀`: This is the vector which will continually be updated with the current initial guess.
- `combined_profiles`: Vector that combines the normalised profile log-likelihood function values going left and right.
- `combined_param_vals`: Vector that combines the parameter values going left and right.
- `combined_other_mles`: Vector that combines the parameter values going left and right.
"""
function prepare_profile(prob::OptimizationProblem, θₘₗₑ, n, param_ranges)
    left_profiles = Vector{Float64}([])
    right_profiles = Vector{Float64}([])
    left_param_vals = Vector{Float64}([])
    right_param_vals = Vector{Float64}([])
    left_other_mles = Vector{Vector{Float64}}([])
    right_other_mles = Vector{Vector{Float64}}([])
    combined_profiles = Vector{Float64}([])
    combined_param_vals = Vector{Float64}([])
    combined_other_mles = Vector{Vector{Float64}}([])
    sizehint!(left_profiles, length(param_ranges[1]))
    sizehint!(right_profiles, length(param_ranges[2]))
    sizehint!(combined_profiles, length(param_ranges[1]) + length(param_ranges[2]))
    sizehint!(combined_param_vals, length(param_ranges[1]) + length(param_ranges[2]))
    sizehint!(combined_other_mles, length(param_ranges[1]) + length(param_ranges[2]))
    cache = dualcache(zeros(length(θₘₗₑ)))
    θ₀ = θₘₗₑ[Not(n)]
    prob = update_prob(prob, n) # Update the lower and upper bounds to exclude n     
    prob = update_prob(prob, θ₀) # This is replaced immediately, but just helps the problem start with the correct dimensions 
    return prob, left_profiles, right_profiles, left_param_vals, right_param_vals, left_other_mles, right_other_mles,
    cache, θ₀,
    combined_profiles, combined_param_vals, combined_other_mles
end

"""
    construct_profile_ranges(sol::LikelihoodSolution, resolution, n, param_bounds)
    construct_profile_ranges(prob::LikelihoodProblem, sol::LikelihoodSolution, resolution::Int; param_bounds = bounds(prob))

Construct ranges for each parameter in `prob` to use for profiling. `n` can be provided to construct a range only for the `n`th parameter.

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
function construct_profile_ranges(sol::LikelihoodSolution, resolution, n, param_bounds)
    θᵢ = mle(sol)[n]
    ℓ, u = param_bounds
    if ⊻(isinf(ℓ), isinf(u))
        error("The provided parameter bounds must be finite.")
    end
    left_range = LinRange(θᵢ, ℓ, resolution)
    right_range = LinRange(θᵢ, u, resolution)
    param_ranges = (left_range, right_range)
    return param_ranges
end
function construct_profile_ranges(prob::LikelihoodProblem, sol::LikelihoodSolution, resolution; param_bounds=bounds(prob))
    if resolution isa Number
        resolution = repeat([resolution], num_params(prob))
    end
    param_ranges = Vector{NTuple{2,LinRange{Float64,Int64}}}(undef, num_params(prob))
    mles = mle(sol)
    for i in eachindex(mles)
        param_ranges[i] = construct_profile_ranges(sol, resolution[i], i, param_bounds[i])
    end
    param_ranges
end


"""
    profile!(prob::OptimizationProblem, profile_vals, other_mles, n, θₙ, θ₀, ℓₘₐₓ, alg, cache; kwargs...)

Computes the normalised profile log-likelihood function for the `n`th variable at `θₙ`.

# Arguments 
- `prob::OptimizationProblem`: The `OptimizationProblem`. This should be already updated from [`update_prob`](@ref).
- `profile_vals`: Values of the normalised profile log-likelihood function. Gets updated with the new value at the index `i`.
- `other_mles`: This will get updated with the MLEs for the other parameters not being profiled.
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
function profile!(prob::OptimizationProblem, profile_vals, other_mles, n, θₙ, θ₀, ℓₘₐₓ, alg, cache; kwargs...)
    prob = update_prob(prob, n, θₙ, cache, θ₀) # Update the objective function and initial guess 
    soln = solve(prob, alg; kwargs...)
    for j in eachindex(θ₀)
        θ₀[j] = soln.u[j]
    end
    push!(profile_vals, -soln.minimum - ℓₘₐₓ)
    push!(other_mles, soln.u)
    return nothing
end

"""
    profile(prob::OptimizationProblem, n, θₙ, θ₀, ℓₘₐₓ, alg, cache; kwargs...)

Similar to [`profile!`](@ref), except returns the `OptimizationSolution`.
"""
function profile(prob::OptimizationProblem, n, θₙ, θ₀, ℓₘₐₓ, alg, cache; kwargs...)
    prob = update_prob(prob, n, θₙ, cache, θ₀)
    soln = solve(prob, alg; kwargs...)
    return soln
end

"""
    find_endpoint!(prob::OptimizationProblem, param_vals, profile_vals, other_mles, θ₀, ℓₘₐₓ, n, alg, threshold, cache, param_ranges; kwargs...)

Optimises the profile likelihood until exceeding `threshold`, going in the direction specified by `Δθ`.

# Arguments 
- `prob`: The `OptimizationProblem`. This should be already updated from [`update_prob`](@ref).
- `param_vals`: Parameter values to use for the normalised profile log-likelihood function.
- `profile_vals`: Values of the normalised profile log-likelihood function at the corresponding values in `param_vals`.
- `other_mles`: This will get updated with the MLEs for the other parameters not being profiled.
- `θ₀`: Initial guesses for the parameters to optimise over (i.e. initial guesses for the parameters without the `n`th variable).
- `ℓₘₐₓ`: The value of the log-likelihood function at the MLEs.
- `n`: The variable being profiled.
- `alg`: Algorithm to use for optimising `prob`.
- `threshold`: When the normalised profile log-likelihood function drops below this `threshold`, stop.
- `cache`: Cache array for storing the `n`th variable along with the other variables. See its use in [`construct_new_f`](@ref).
- `param_ranges`: Values to try for profiling. See also [`construct_profile_ranges`](@ref).
- `min_steps`: Minimum number of steps to take.

# Keyword Arguments 
- `kwargs...`: Extra keyword arguments to pass to the optimisers.

# Outputs 
Returns nothing, but `profile_vals` gets updated with the found values and `param_vals` with the used parameter values.
"""
function find_endpoint!(prob::OptimizationProblem, param_vals, profile_vals, other_mles, θ₀, ℓₘₐₓ, n, alg, threshold, cache, param_ranges, min_steps; kwargs...)
    steps = 1
    old_θ₀ = copy(θ₀)
    for θₙ in param_ranges
        push!(param_vals, θₙ)
        profile!(prob, profile_vals, other_mles, n, θₙ, old_θ₀, ℓₘₐₓ, alg, cache; kwargs...)
        steps += 1
        if profile_vals[end] ≤ threshold
            break
        end
    end
    if steps ≤ min_steps
        new_range = LinRange(param_vals[1], param_vals[end], min_steps)
        !isempty(param_vals) && empty!(param_vals)
        !isempty(profile_vals) && empty!(profile_vals)
        !isempty(other_mles) && empty!(other_mles)
        find_endpoint!(prob, param_vals, profile_vals, other_mles, θ₀, ℓₘₐₓ, n, alg, typemin(threshold), cache, new_range, 0; kwargs...)
        return nothing
    end
    return nothing
end

"""
    [1] profile(prob::OptimizationProblem, θₘₗₑ, ℓₘₐₓ, n, alg, threshold, param_ranges; kwargs...)
    [2] profile(prob::LikelihoodProblem, sol::LikelihoodSolution, n, alg, threshold, param_ranges; kwargs...)
    [3] profile!(prob::LikelihoodProblem, sol::LikelihoodSolution, n, alg, threshold, param_ranges, profile_vals, param_vals, other_mles, splines; kwargs...)
    [4] profile(prob::LikelihoodProblem, sol::LikelihoodSolution; <keyword arguments>)
    [5] profile!(prof::ProfileLikelihoodSolution, n; <keyword arguments>)
    [6] profile!(prof::ProfileLikelihoodSolution; n = 1:num_params(prof.prob), alg = prof.mle.alg, spline=true, use_threads=false, conf_level, kwargs...)

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
- `min_steps`: Minimum number of steps to take in each direction.

The fourth method also has keyword arguments, given below.

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
- `min_steps = 10`: Minimum number of steps to take in each direction.
- `kwargs...`: Extra keyword arguments to pass to the optimisers.
- `n = 1:num_params(prof.prob)`: Parameters to refine, if using the sixth method.

# Outputs 
If `n` is provided, then we return 

- `profile_vals`: Values for the normalised profile log-likelihood function.
- `param_vals`: The values for the `i`th parameter that correspond to the values in `profile_vals`.
- `other_mles`: The MLEs for the other parameters, i.e. those not being profiled, corresponding to the above data.

Otherwise, returns a [`ProfileLikelihoodSolution`](@ref) struct.

# Refinement 
If you want to refine the profile for a parameter, potentially using more gridpoints, you can use the fifth method. 
The keyword arguments in this case are the same as for the fourth method, with the exception that `param_ranges` should now be 
specific to the `n`th parameter.

If you instead want to keep the same gridpoints and just re-optimise at each parameter value, you can use the last method. This method simply 
goes over each existing parameter value for the provided `n` and olves the optimisation problem again. Of course, if you do not change 
the algorithm or any of the solver keyword arguments, then nothing will change, so be sure that you use different settings if you decide to use 
this method.
"""
function profile end
function profile(prob::OptimizationProblem, θₘₗₑ, ℓₘₐₓ, n, alg, threshold, param_ranges, min_steps; kwargs...)
    prob, left_profiles, right_profiles,
    left_param_vals, right_param_vals,
    left_other_mles, right_other_mles,
    cache, θ₀,
    combined_profiles, combined_param_vals, combined_other_mles = prepare_profile(prob, θₘₗₑ, n, param_ranges)
    find_endpoint!(prob, left_param_vals, left_profiles, left_other_mles, copy(θ₀), ℓₘₐₓ, n, alg, threshold, cache, param_ranges[1], min_steps; kwargs...) # Go left
    find_endpoint!(prob, right_param_vals, right_profiles, right_other_mles, copy(θ₀), ℓₘₐₓ, n, alg, threshold, cache, param_ranges[2], min_steps; kwargs...) # Go right
    append!(combined_profiles, left_profiles, right_profiles) # Now combine the profiles 
    append!(combined_param_vals, left_param_vals, right_param_vals) # Now combine the parameter values 
    append!(combined_other_mles, left_other_mles, right_other_mles) # Now combine the other MLEs
    # Make sure the results are sorted 
    sort_idx = sortperm(combined_param_vals)
    permute!(combined_param_vals, sort_idx)
    permute!(combined_profiles, sort_idx)
    permute!(combined_other_mles, sort_idx)
    # Some cleanup duplicate indices
    idx = unique(i -> combined_param_vals[i], eachindex(combined_param_vals))
    keepat!(combined_param_vals, idx)
    keepat!(combined_profiles, idx)
    keepat!(combined_other_mles, idx)
    return combined_profiles, combined_param_vals, combined_other_mles
end
@inline function profile(prob::LikelihoodProblem, sol::LikelihoodSolution, n, alg, threshold, param_ranges, min_steps=10; kwargs...)
    return profile(prob.prob, mle(sol), maximum(sol), n, alg, threshold, param_ranges, min_steps; kwargs...)
end
@doc (@doc profile) @inline function profile!(prob::LikelihoodProblem, sol::LikelihoodSolution, n, alg, threshold, param_ranges, profile_vals, param_vals, other_mles, splines, min_steps; kwargs...)
    _prof, _θ, _other_mles = profile(prob, sol, n, alg, threshold, param_ranges, min_steps; kwargs...)
    param_vals[n] = _θ
    profile_vals[n] = _prof
    other_mles[n] = _other_mles
    try
        splines[n] = Spline1D(_θ, _prof)
    catch
        error("Error creating the spline. Try increasing the grid resolution for parameter $n or increasing $min_steps.")
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
    min_steps=10,
    kwargs...)
    N = num_params(prob)
    θ = Dict{Int64,Vector{Float64}}([])
    prof = Dict{Int64,Vector{Float64}}([])
    other_mles = Dict{Int64,Vector{Vector{Float64}}}([])
    splines = Vector{Spline1D}(undef, N)
    sizehint!(θ, N)
    sizehint!(prof, N)
    for n in 1:N
        profile!(prob, sol, n, alg, threshold, param_ranges[n], prof, θ, other_mles, splines, min_steps; kwargs...)
    end
    splines = Dict(1:N .=> splines) # We define the Dict here rather than above to make sure we get the types right
    conf_ints = confidence_intervals(θ, prof; conf_level, spline)
    profile_sol = ProfileLikelihoodSolution(θ, prof, prob, sol, splines, conf_ints, other_mles)
    return profile_sol
end
@doc (@doc profile) function profile!(prof::ProfileLikelihoodSolution, n;
    alg=prof.mle.alg,
    conf_level=0.95,
    spline=true,
    threshold=-0.5quantile(Chisq(1), conf_level),
    resolution=200,
    param_ranges=construct_profile_ranges(prof.prob, prof.mle, resolution)[n],
    use_threads=false,
    min_steps=10,
    kwargs...)
    profile!(prof.prob, prof.mle, n, alg, threshold, param_ranges, prof.profile, prof.θ, prof.other_mles, prof.spline, min_steps; kwargs...)
    prof.confidence_intervals[n] = confidence_intervals(prof.θ, prof.profile, n; conf_level, spline)
    return nothing
end
@doc (@doc profile) function profile!(prof::ProfileLikelihoodSolution;
    n=1:num_params(prof.prob),
    alg=prof.mle.alg,
    spline=true,
    use_threads=false,
    conf_level=0.95,
    kwargs...)
    ## Setup
    cache = dualcache(zeros(num_params(prof.prob)))
    likprob = prof.prob
    ℓₘₐₓ = maximum(prof.mle)
    ## Optimise
    for _n in n
        optprob = update_prob(likprob.prob, _n)
        for (i, other_θ) in pairs(prof.other_mles[_n])
            soln = profile(optprob, _n, prof.θ[_n][i], other_θ, ℓₘₐₓ, alg, cache; kwargs...)
            prof.profile[_n][i] = -soln.minimum - ℓₘₐₓ
            prof.other_mles[_n][i] .= soln.u
        end
        ## Get the new spline 
        prof.spline[_n] = Spline1D(prof.θ[_n], prof.profile[_n])
        ## Get the new confidence interval 
        prof.confidence_intervals[_n] = confidence_intervals(prof.θ, prof.profile, _n; conf_level, spline)
    end
    return nothing
end