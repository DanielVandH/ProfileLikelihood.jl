"""
    profile!(prob::OptimizationProblem, profile_vals, n, θₙ, θ₀, ℓₘₐₓ, alg, cache)

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

# Output 
There is no output, but `profile_vals` gets updated (via `push!`) with the new normalised profile log-likelihood value at `θₙ`.
"""
function profile!(prob::OptimizationProblem, profile_vals, n, θₙ, θ₀, ℓₘₐₓ, alg, cache)
    prob = update_prob(prob, n, θₙ, cache, θ₀) # Update the objective function and initial guess 
    soln = solve(prob, alg)
    for j in eachindex(θ₀)
        θ₀[j] = soln.u[j]
    end
    push!(profile_vals, -soln.minimum - ℓₘₐₓ)
    return nothing
end

"""
    find_endpoint!(prob::OptimizationProblem, param_vals, profile_vals, θ₀, ℓₘₐₓ, n, alg, threshold, cache, param_ranges)

Optimises the profile likelihood until exceeding `threshold`, going in the direction specified by `Δθ`.

## Arguments 
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

## Outputs 
Returns nothing, but `profile_vals` gets updated with the found values and `param_vals` with the used parameter values.
"""
function find_endpoint!(prob::OptimizationProblem, param_vals, profile_vals, θ₀, ℓₘₐₓ, n, alg, threshold, cache, param_ranges)
    for θₙ in param_ranges
        push!(param_vals, θₙ)
        profile!(prob, profile_vals, n, θₙ, θ₀, ℓₘₐₓ, alg, cache)
        if profile_vals[end] ≤ threshold 
            return nothing 
        end
    end
    return nothing 
end