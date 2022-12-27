##########################################################
##
## UNIVARIATE PROFILE PREDICTION
##
##########################################################
"""
    get_prediction_intervals(q, prof::ProfileLikelihoodSolution, data;
        q_prototype=isinplace(q, 3) ? nothing : build_q_prototype(q, prof, data), resolution=250)

Obtain prediction intervals for the output of the prediction function `q`.

# Arguments 
- `q`: The prediction function, taking either the form `(θ, data)` or `(cache, θ, data)`. The former version is an out-of-place version, returning the full vector, while the latter version is an in-place version, with the output being placed into `cache`. The argument `θ` is the same as the parameters used in the likelihood problem (from `prof`), and the `data` argument is the same `data` as in this function. 
- `prof::ProfileLikelihoodSolution`: The profile likelihood results. 
- `data`: The argument `data` in `q`.

# Keyword Arguments 
- `q_prototype=isinplace(q, 3) ? nothing : build_q_prototype(q, prof, data)`: A prototype for the result of `q`. If you are using the `q(θ, data)` version of `q`, this can be inferred from `build_q_prototype`, but if you are using the in-place version then a `build_q_prototype` is needed. For example, if `q` returns a vector with eltype `Float64` and has length 27, `q_prototype` could be `zeros(27)`.
- `resolution=250`: The amount of curves to evaluate for each parameter.
- `parallel=false`: Whether to use multithreading. Multithreading is used when building `q_vals` below.

# Outputs 
Four values are returned. In order: 

- `individual_intervals`: Prediction intervals for the output of `q`, relative to each parameter. 
- `union_intervals`: The union of the individual prediction intervals from `individual_intervals`.
- `q_vals`: Values of `q` at each parameter considered. The output is a `Dict`, where the parameter index is mapped to a matrix where each column is an output from `q`, with the `j`th column corresponding to the parameter value at `param_ranges[j]`.
- `param_ranges`: Parameter values used for each prediction interval. 
"""
function get_prediction_intervals(q, prof::ProfileLikelihoodSolution, data;
    q_prototype=isinplace(q, 3) ? nothing : build_q_prototype(q, prof, data),
    resolution=250,
    parallel=false)
    ## Test if q is inplace or not
    iip = isinplace(q, 3)::Bool
    (iip && isnothing(q_prototype)) && throw("If the prediction function is inplace, then you must provide a cache for the function's first argument in q_prototype.")

    ## Setup
    prof_idx, param_ranges, splines = prepare_prediction_grid(prof, resolution)
    θ, q_vals, q_lower_bounds, q_upper_bounds, q_union_lower_bounds, q_union_upper_bounds = prepare_prediction_cache(prof, prof_idx, q_prototype, resolution, Val(parallel))

    ## Evaluate the prediction function
    evaluate_prediction_function!(q_vals, param_ranges, splines, θ, prof_idx, q, data, Val(iip))

    ## Get all the bounds 
    update_interval_bounds!(q_lower_bounds, q_upper_bounds, q_union_lower_bounds, q_union_upper_bounds, q_vals, prof_idx)

    ## Convert the results into ConfidenceIntervals 
    level = get_level(get_confidence_intervals(prof[first(prof_idx)]))
    individual_intervals, union_intervals = get_prediction_intervals(prof_idx, q_lower_bounds, q_upper_bounds,
        q_union_lower_bounds, q_union_upper_bounds, level)

    ## Done
    return individual_intervals, union_intervals, q_vals, param_ranges
end

@inline function prepare_prediction_grid(prof::ProfileLikelihoodSolution, resolution)
    prof_idx = profiled_parameters(prof)
    param_ranges = Dict(prof_idx .=> [LinRange(get_confidence_intervals(prof[i])..., resolution) for i in prof_idx])
    splines = spline_other_mles(prof)
    return prof_idx, param_ranges, splines
end

function prepare_prediction_cache(prof::ProfileLikelihoodSolution, prof_idx, q_prototype::AbstractArray{T}, resolution, parallel::Val{B}) where {T,B}
    if !B
        θ = zeros(number_of_parameters(get_likelihood_problem(prof)))
    else
        θ = Dict{Int64,Vector{Vector{T}}}(prof_idx .=> [[zeros(T, number_of_parameters(get_likelihood_problem(prof))) for _ in 1:Base.Threads.nthreads()] for _ in prof_idx])
    end
    q_length = length(q_prototype)
    q_vals = Dict{Int64,Matrix{T}}(prof_idx .=> [zeros(T, q_length, resolution) for _ in prof_idx])
    q_lower_bounds = Dict{Int64,Vector{T}}(prof_idx .=> [typemax(T) * ones(T, q_length) for _ in prof_idx])
    q_upper_bounds = Dict{Int64,Vector{T}}(prof_idx .=> [typemin(T) * ones(T, q_length) for _ in prof_idx])
    q_union_lower_bounds = typemax(T) * ones(T, q_length)
    q_union_upper_bounds = typemin(T) * ones(T, q_length)
    return θ, q_vals, q_lower_bounds, q_upper_bounds, q_union_lower_bounds, q_union_upper_bounds
end

@inline function evaluate_prediction_function!(q_n, range, spline, θ::AbstractVector{T}, n::Integer, q, data, iip::Val{B}=Val(isinplace(q, 3)), parallel::Val{false}=Val(false)) where {B,T<:Number}
    for (j, ψ) in pairs(range)
        build_θ!(θ, n, spline, ψ)
        if B
            @views q(q_n[:, j], θ, data)
        else
            @views q_n[:, j] .= q(θ, data)
        end
    end
    return nothing
end
@inline function evaluate_prediction_function!(q_n, range, spline, θ, n::Integer, q, data, iip::Val{B}=Val(isinplace(q, 3)), parallel::Val{true}=Val(true)) where {B}
    Base.Threads.@threads for j in eachindex(range)
        id = Base.Threads.threadid()
        ψ = range[j]
        build_θ!(θ[id], n, spline, ψ)
        if B
            @views q(q_n[:, j], θ[id], data)
        else
            @views q_n[:, j] .= q(θ[id], data)
        end
    end
    return nothing
end
function evaluate_prediction_function!(q_vals, ranges, splines, θ::AbstractVector, prof_idx::Vector{Int}, q, data, iip::Val{B}=Val(isinplace(q, 3)), parallel::Val{false}=Val(false)) where {B}
    for n in prof_idx
        q_n = q_vals[n]
        range = ranges[n]
        spline = splines[n]
        evaluate_prediction_function!(q_n, range, spline, θ, n, q, data, iip)
    end
    return nothing
end
function evaluate_prediction_function!(q_vals, ranges, splines, θ::Dict, prof_idx::Vector{Int}, q, data, iip::Val{B}=Val(isinplace(q, 3)), parallel::Val{true}=Val(true)) where {B}
    @sync for n in prof_idx
        Base.Threads.@spawn evaluate_prediction_function!(q_vals[n], ranges[n], splines[n], θ[n], n, q, data, iip)
    end
    return nothing
end

function update_interval_bounds!(q_lower_bounds, q_upper_bounds, q_union_lower_bounds, q_union_upper_bounds, q_n, i, j)
    q_n_ij = q_n[i, j]
    q_lb_i = q_lower_bounds[i]
    q_ub_i = q_upper_bounds[i]
    q_union_lb_i = q_union_lower_bounds[i]
    q_union_ub_i = q_union_upper_bounds[i]
    update_lower_bounds!(q_lower_bounds, q_union_lower_bounds, q_n_ij, q_lb_i, q_union_lb_i, i)
    update_upper_bounds!(q_upper_bounds, q_union_upper_bounds, q_n_ij, q_ub_i, q_union_ub_i, i)
    return nothing
end
function update_interval_bounds!(q_lower_bounds, q_upper_bounds, q_union_lower_bounds, q_union_upper_bounds, q_n)
    for j in axes(q_n, 2)
        for i in axes(q_n, 1)
            update_interval_bounds!(q_lower_bounds, q_upper_bounds, q_union_lower_bounds, q_union_upper_bounds, q_n, i, j)
        end
    end
    return nothing
end
function update_interval_bounds!(q_lower_bounds, q_upper_bounds, q_union_lower_bounds, q_union_upper_bounds, q_vals, prof_idx)
    for n in prof_idx
        q_n = q_vals[n]
        q_lb = q_lower_bounds[n]
        q_ub = q_upper_bounds[n]
        update_interval_bounds!(q_lb, q_ub, q_union_lower_bounds, q_union_upper_bounds, q_n)
    end
    return nothing
end

@inline function build_q_prototype(q, prof::ProfileLikelihoodSolution, data)
    pred = q(get_mle(get_likelihood_solution(prof)), data)
    return zero(pred)
end
@inline function build_q_prototype(q, prof::ProfileLikelihoodSolutionView, data)
    return build_q_prototype(q, get_parent(prof), data)
end

@inline function spline_other_mles(prof::ProfileLikelihoodSolutionView)
    data = get_parameter_values(prof)
    other_mles = get_other_mles(prof)
    spline = interpolate((data,), other_mles, Gridded(Linear()))
    return spline
end
@inline function spline_other_mles(prof::ProfileLikelihoodSolution)
    param_idx = profiled_parameters(prof)
    splines = Dict(param_idx .=> [spline_other_mles(prof[i]) for i in param_idx])
    return splines
end

@inline function build_θ!(θ, n::Integer, spline, ψ)
    @views θ[Not(n)] .= spline(ψ)
    θ[n] = ψ
    return nothing
end

@inline function update_lower_bounds!(q_lower_bounds, q_union_lower_bounds, q, lb, union_lb, i)
    if q < lb
        q_lower_bounds[i] = q
        if q < union_lb
            q_union_lower_bounds[i] = q
        end
    end
end
@inline function update_upper_bounds!(q_upper_bounds, q_union_upper_bounds, q, ub, union_ub, i)
    if q > ub
        q_upper_bounds[i] = q
        if q > union_ub
            q_union_upper_bounds[i] = q
        end
    end
end

@inline function get_prediction_intervals(prof_idx, q_lower_bounds, q_upper_bounds,
    q_union_lower_bounds::AbstractArray{F}, q_union_upper_bounds, level::L) where {F,L}
    individual_intervals = Dict{Int64,Vector{ConfidenceInterval{F,L}}}(prof_idx .=> [[ConfidenceInterval{F,L}(a, b, level) for (a, b) in zip(q_lower_bounds[i], q_upper_bounds[i])] for i in prof_idx])
    union_intervals = [ConfidenceInterval{F,L}(a, b, level) for (a, b) in zip(q_union_lower_bounds, q_union_upper_bounds)]
    return individual_intervals, union_intervals
end

