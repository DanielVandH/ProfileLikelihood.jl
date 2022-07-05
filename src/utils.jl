"""
    gaussian_loglikelihood(x::AbstractVector{<:Real}, μ::AbstractVector{<:Real}, σ, n) 
    gaussian_loglikelihood(x::AbstractVector{<:AbstractVector{<:Real}}, μ::AbstractVector{<:AbstractVector{<:Real}}, σ, n)

Computes the log-likelihood function for `x ~ N(μ, σ²)`.
"""
function gaussian_loglikelihood end
@inline function gaussian_loglikelihood(x::AbstractVector{<:Real}, μ::AbstractVector{<:Real}, σ, n)
    ℓ = -0.5n * log(2π * σ^2)
    s = zero(eltype(x))
    @turbo for i ∈ eachindex(x, μ)
        s += (x[i] - μ[i])^2
    end
    ℓ = ℓ - 0.5 / σ^2 * s
    return ℓ
end
@inline function gaussian_loglikelihood(x::AbstractVector{<:AbstractVector{<:Real}}, μ::AbstractVector{<:AbstractVector{<:Real}}, σ, n)
    ℓ = -0.5n * log(2π * σ^2)
    s = zero(eltype(eltype(x)))
    for i ∈ eachindex(x, μ)
        @turbo for j ∈ eachindex(x[i], μ[i])
            s += (x[i][j] - μ[i][j])^2
        end
    end
    ℓ = ℓ - 0.5 / σ^2 * s
    return ℓ
end

"""
    check_solution(sol::SciMLBase.AbstractODESolution)

Checks if the solution `sol` was successfully integrated, returning 
`sol.retcode ≠ :Success`.
"""
@inline check_solution(sol::SciMLBase.AbstractODESolution) = sol.retcode ≠ :Success

@inline OptimizationNLopt.algorithm_name(alg) = nameof(typeof(alg))
function OptimizationNLopt.algorithm_name(sol::AbstractLikelihoodSolution)
    if sol.alg isa Optim.AbstractConstrainedOptimizer # need to wrap in Fminbox...
        return string(OptimizationNLopt.algorithm_name(sol.alg)) * "(" * string(nameof(typeof(sol.alg).parameters[1])) * ")" # e.g. Fminbox(BFGS)
    end
    return OptimizationNLopt.algorithm_name(sol.alg)
end
@inline OptimizationNLopt.algorithm_name(sol::ProfileLikelihoodSolution) = OptimizationNLopt.algorithm_name(sol.mle)

"""
    subscriptnumber(i::Int)

Given an integer `i`, returns the subscript `ᵢ`.

Sourced from https://stackoverflow.com/a/64758370.
"""
function subscriptnumber(i::Int)#https://stackoverflow.com/a/64758370
    if i < 0
        c = [Char(0x208B)]
    else
        c = []
    end
    for d in reverse(digits(abs(i)))
        push!(c, Char(0x2080 + d))
    end
    return join(c)
end

"""
    clean_named_tuple(nt::NamedTuple)

Given a `NamedTuple` `nt`, returns another `NamedTuple` with all keys removed that correspond to 
a `Nothing` value.
"""
@inline function clean_named_tuple(nt::NamedTuple)
    good_names = findall(!isnothing, nt)
    nt_cleaned = length(good_names) > 0 ? NamedTuple{Tuple(good_names)}(nt) : nothing
    return nt_cleaned
end

"""
    setup_integrator(f, u₀, tspan, p, t, alg=nothing; ode_problem_kwargs = nothing, kwargs...)
    setup_integrator(prob, t, alg=nothing; kwargs...)

Constructs the `integrator` for solving a differential equation with `DifferentialEquations.jl`, given 
the ODEProblem `prob`, with solutions returned at the times `t`. If `alg = nothing`, then a default algorithm 
is chosen. The second method constructs the `ODEProblem` required, `prob = ODEProblem(f, u₀, tspan, p; ode_problem_kwargs...)`.
"""
function setup_integrator end
@inline function setup_integrator(prob, t, alg=nothing; kwargs...)
    if isnothing(alg) ## Select default algorithm. ...\.julia\packages\DifferentialEquations\4jfQK\src\default_solve.jl
        alg, new_kwargs = DifferentialEquations.default_algorithm(prob; kwargs...)
        alg = DiffEqBase.prepare_alg(alg, prob.u0, prob.p, prob)
        return DifferentialEquations.init(prob, alg, saveat=t; new_kwargs..., kwargs...)
    else
        return DifferentialEquations.init(prob, alg, saveat=t; kwargs...)
    end
end
@inline function setup_integrator(f, u₀, tspan, p, t, alg=nothing; ode_problem_kwargs=nothing, kwargs...)
    prob = isnothing(ode_problem_kwargs) ? ODEProblem(f, u₀, tspan, p) : ODEProblem(f, u₀, tspan, p; ode_problem_kwargs...)
    return setup_integrator(prob, t, alg; kwargs...)
end

"""
    finite_bounds(prob::AbstractLikelihoodProblem)

Returns `true` if the given `prob::AbstractLikelihoodProblem` has finite lower and upper bounds, and `false`
otherwise.
"""
function finite_bounds(prob::AbstractLikelihoodProblem)
    all(isfinite, lower_bounds(prob)) && all(isfinite, upper_bounds(prob))
end
