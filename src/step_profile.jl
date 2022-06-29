"""
    step_profile!(prob::OptimizationProblem, ℓₘₐₓ, i, param_vals, profile_vals, θ₀, cache; alg=PolyOpt(), Δθ)
    profile!(prob::OptimizationProblem, ℓₘₐₓ, i, θᵢ, param_vals, profile_vals, θ₀, cache; alg=PolyOpt())

Computes the normalised profile log-likelihood function for the `i`th variable at `θᵢ`, or at 
`param_vals[end] + Δθ` is `θᵢ` is not provided.

# Arguments 
- `prob::OptimizationProblem`: The `OptimizationProblem`. This should be already updated from [`update_prob`](@ref).
- `ℓₘₐₓ`: The value of the log-likelihood function at the MLEs.
- `i`: The index of the variable being profiled. 
- `θᵢ`: The value to fix the `i`th variable at. 
- `param_vals`: Vector of parameter values for the `i`th variable.
- `profile_vals`: Values of the normalised profile log-likelihood function at the corresponding values in `param_vals`.
- `θ₀`: Initial guesses for the parameters to optimise over (i.e. initial guesses for the parameters without the `i`th variable).
- `cache`: Cache array for storing the `i`th variable along with the other variables. See its use in [`construct_new_f`](@ref).

# Keyword Arguments 
- `alg=PolyOpt()`: Algorithm to use for optimising `prob`.
- `Δθ`: The value to increment the `i`th value by, computing `θᵢ = param_vals[end] + Δθ`.

# Output 
There is no output, but `param_vals` and `profile_vals` are updated with `θᵢ` and the normalised profile log-likelihood at `θᵢ`, respectively.
"""
function profile!(prob::OptimizationProblem, ℓₘₐₓ, i, θᵢ, param_vals, profile_vals, θ₀, cache; alg=PolyOpt())
    prob = update_prob(prob, i, θᵢ, cache, θ₀) # Update the objective function and initial guess 
    soln = solve(prob, alg)
    for i in eachindex(θ₀)
        θ₀[i] = soln.u[i]
    end
    push!(param_vals, θᵢ)
    push!(profile_vals, -soln.minimum - ℓₘₐₓ)
    return nothing
end
@doc (@doc profile!) @inline function step_profile!(prob::OptimizationProblem, ℓₘₐₓ, i, param_vals, profile_vals, θ₀, cache; alg=PolyOpt(), Δθ)
    profile!(prob, ℓₘₐₓ, i, param_vals[end] + Δθ, param_vals, profile_vals, θ₀, cache; alg)
    return nothing
end

"""
    find_endpoint!(prob::OptimizationProblem, profile_vals, threshold, min_steps, max_steps, ℓₘₐₓ, i, param_vals, θ₀, cache, alg, Δθ)

Optimises the profile likelihood until exceeding `threshold`, going in the direction specified by `Δθ`.

## Arguments 
- `prob`: The `OptimizationProblem`. This should be already updated from [`update_prob`](@ref).
- `profile_vals`: Values of the normalised profile log-likelihood function at the corresponding values in `param_vals`.
- `threshold`: When the normalised profile log-likelihood function drops below this `threshold`, stop.
- `min_steps`: Minimum number of steps. If the process terminates before this number is reached, then the function is computed at equidistant points (with `min_steps` points) between the extrema of the used points.
- `max_steps`: Maximum number of steps to take before terminating. Will return an error if this number of steps is reached.
- `ℓₘₐₓ`: The value of the log-likelihood function at the MLEs.
- `i`: The variable being profiled.
- `profile_vals`: Values of the normalised profile log-likelihood function at the corresponding values in `param_vals`.
- `θ₀`: Initial guesses for the parameters to optimise over (i.e. initial guesses for the parameters without the `i`th variable).
- `cache`: Cache array for storing the `i`th variable along with the other variables. See its use in [`construct_new_f`](@ref).
- `alg`: Algorithm to use for optimising `prob`.
- `Δθ`: The value to increment the `i`th value by, computing `θᵢ = param_vals[end] + Δθ`.

## Outputs 
Returns nothing, but `param_vals` and `profile_vals` are updated as in [`step_profile!`](@ref).
"""
function find_endpoint!(prob::OptimizationProblem, profile_vals, threshold, min_steps, max_steps, ℓₘₐₓ, i, param_vals, θ₀, cache, alg, Δθ)
    steps = 1
    while profile_vals[end] ≥ threshold && steps ≤ max_steps
        step_profile!(prob, ℓₘₐₓ, i, param_vals, profile_vals, θ₀, cache; alg, Δθ)
        steps += 1
    end
    if steps > max_steps
        direction = sign(Δθ) > 0 ? "right" : "left"
        @warn "Maximum number of steps reached going to the $direction for the $(i)th variable."
    end
    if steps < min_steps
        θ0 = LinRange(param_vals[end], param_vals[1], min_steps)
        for θ in θ0
            profile!(prob, ℓₘₐₓ, i, θ, param_vals, profile_vals, θ₀, cache; alg)
        end
    end
    return nothing
end