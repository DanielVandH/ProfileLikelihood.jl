"""
    get_prediction_intervals(q, prof::(Bivariate)ProfileLikelihoodSolution, data;
        q_prototype=isinplace(q, 3) ? nothing : build_q_prototype(q, prof, data), resolution=250)

Obtain prediction intervals for the output of the prediction function `q`, assuming `q` returns (or operates in-place on) a vector.

# Arguments 
- `q`: The prediction function, taking either the form `(θ, data)` or `(cache, θ, data)`. The former version is an out-of-place version, returning the full vector, while the latter version is an in-place version, with the output being placed into `cache`. The argument `θ` is the same as the parameters used in the likelihood problem (from `prof`), and the `data` argument is the same `data` as in this function. 
- `prof::(Bivariate)ProfileLikelihoodSolution`: The profile likelihood results. 
- `data`: The argument `data` in `q`.

# Keyword Arguments 
- `q_prototype=isinplace(q, 3) ? nothing : build_q_prototype(q, prof, data)`: A prototype for the result of `q`. If you are using the `q(θ, data)` version of `q`, this can be inferred from `build_q_prototype`, but if you are using the in-place version then a `build_q_prototype` is needed. For example, if `q` returns a vector with eltype `Float64` and has length 27, `q_prototype` could be `zeros(27)`.
- `resolution::Integer=250`: The amount of curves to evaluate for each parameter. This will be the same for each parameter. If `prof isa BivariateProfileLikelihoodSolution`, then `resolution^2` points are defined inside a bounding box for the confidence region, and then we throw away all points outside of the actual confidence region.
- `parallel=false`: Whether to use multithreading. Multithreading is used when building `q_vals` below.

# Outputs 
Four values are returned. In order: 

- `individual_intervals`: Prediction intervals for the output of `q`, relative to each parameter. 
- `union_intervals`: The union of the individual prediction intervals from `individual_intervals`.
- `q_vals`: Values of `q` at each parameter considered. The output is a `Dict`, where the parameter index is mapped to a matrix where each column is an output from `q`, with the `j`th column corresponding to the parameter value at `param_ranges[j]`.
- `param_ranges`: Parameter values used for each prediction interval. 
"""
function get_prediction_intervals(q, prof::Union{ProfileLikelihoodSolution,BivariateProfileLikelihoodSolution}, data;
    q_prototype=isinplace(q, 3) ? nothing : build_q_prototype(q, prof, data),
    resolution::Integer=250,
    parallel=false)
    ## Test if q is inplace or not
    iip = isinplace(q, 3)::Bool
    (iip && isnothing(q_prototype)) && throw("If the prediction function is inplace, then you must provide a cache for the function's first argument in q_prototype.")

    ## Setup
    prof_idx, param_ranges, splines, resolution = prepare_prediction_grid(prof, resolution)
    θ, q_vals, q_lower_bounds, q_upper_bounds, q_union_lower_bounds, q_union_upper_bounds = prepare_prediction_cache(prof, prof_idx, q_prototype, resolution, Val(parallel))

    ## Evaluate the prediction function
    evaluate_prediction_function!(q_vals, param_ranges, splines, θ, prof_idx, q, data, Val(iip))

    ## Get all the bounds 
    update_interval_bounds!(q_lower_bounds, q_upper_bounds, q_union_lower_bounds, q_union_upper_bounds, q_vals, prof_idx)

    ## Convert the results into ConfidenceIntervals 
    level = prof isa ProfileLikelihoodSolution ? get_level(get_confidence_intervals(prof[first(prof_idx)])) : get_level(get_confidence_regions(prof[first(prof_idx)]))
    individual_intervals, union_intervals = get_prediction_intervals(prof_idx, q_lower_bounds, q_upper_bounds,
        q_union_lower_bounds, q_union_upper_bounds, level)

    ## Done
    return individual_intervals, union_intervals, q_vals, param_ranges
end

@inline function prepare_prediction_grid(prof::ProfileLikelihoodSolution, resolution)
    prof_idx = profiled_parameters(prof)
    param_ranges = Dict{Int64,LinRange{Float64,Int64}}(prof_idx .=> [LinRange(get_confidence_intervals(prof[i])..., resolution) for i in prof_idx])
    splines = spline_other_mles(prof)
    return prof_idx, param_ranges, splines, resolution
end
@inline function prepare_prediction_grid(prof::BivariateProfileLikelihoodSolution, resolution)
    prof_idx = profiled_parameters(prof)
    param_ranges = Dict{NTuple{2,Integer},Vector{NTuple{2,Float64}}}()
    # Get the grids
    new_res = resolution^2
    for (n, m) in prof_idx
        a, b, c, d = get_bounding_box(prof[n, m])
        grid_1 = LinRange(a, b, resolution)
        grid_2 = LinRange(c, d, resolution)
        full_grid = vec([(x, y) for x in grid_1, y in grid_2])
        points_in_cr = full_grid ∈ get_confidence_regions(prof[(n, m)])
        reduced_grid = full_grid[points_in_cr]
        param_ranges[(n, m)] = reduced_grid
        new_res = length(reduced_grid) < new_res ? length(reduced_grid) : new_res
    end
    # Now trim the grids so that they all have the same size
    for (n, m) in prof_idx
        grid = param_ranges[(n, m)]
        while length(grid) > new_res
            i = rand(eachindex(grid))
            deleteat!(grid, i)
        end
    end
    splines = spline_other_mles(prof)
    return prof_idx, param_ranges, splines, new_res
end

function prepare_prediction_cache(prof::Union{ProfileLikelihoodSolution,BivariateProfileLikelihoodSolution}, prof_idx::AbstractVector{I}, q_prototype::AbstractArray{T}, resolution, parallel::Val{B}) where {T,B,I}
    if !B
        θ = zeros(number_of_parameters(get_likelihood_problem(prof)))
    else
        θ = Dict{I,Vector{Vector{T}}}(prof_idx .=> [[zeros(T, number_of_parameters(get_likelihood_problem(prof))) for _ in 1:Base.Threads.nthreads()] for _ in prof_idx])
    end
    q_length = length(q_prototype)
    q_vals = Dict{I,Matrix{T}}(prof_idx .=> [zeros(T, q_length, resolution) for _ in prof_idx])
    q_lower_bounds = Dict{I,Vector{T}}(prof_idx .=> [typemax(T) * ones(T, q_length) for _ in prof_idx])
    q_upper_bounds = Dict{I,Vector{T}}(prof_idx .=> [typemin(T) * ones(T, q_length) for _ in prof_idx])
    q_union_lower_bounds = typemax(T) * ones(T, q_length)
    q_union_upper_bounds = typemin(T) * ones(T, q_length)
    return θ, q_vals, q_lower_bounds, q_upper_bounds, q_union_lower_bounds, q_union_upper_bounds
end

@inline function evaluate_prediction_function!(q_n, range, spline, θ::AbstractVector{T}, n, q, data, iip::Val{B}=Val(isinplace(q, 3)), parallel::Val{false}=Val(false)) where {B,T<:Number}
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
@inline function evaluate_prediction_function!(q_n, range, spline, θ::AbstractVector{T}, n, q, data, iip::Val{B}=Val(isinplace(q, 3)), parallel::Val{true}=Val(true)) where {B,T<:AbstractVector}
    chunked_range = chunks(eachindex(range), Base.Threads.nthreads())
    Base.Threads.@threads for (j_range, id) in chunked_range
        for j in j_range
            ψ = range[j]
            build_θ!(θ[id], n, spline, ψ)
            if B
                @views q(q_n[:, j], θ[id], data)
            else
                @views q_n[:, j] .= q(θ[id], data)
            end
        end
    end
    return nothing
end
function evaluate_prediction_function!(q_vals, ranges, splines, θ::AbstractVector{T}, prof_idx::Vector, q, data, iip::Val{B}=Val(isinplace(q, 3)), parallel::Val{false}=Val(false)) where {B,T<:Number}
    for n in prof_idx
        q_n = q_vals[n]
        range = ranges[n]
        spline = splines[n]
        evaluate_prediction_function!(q_n, range, spline, θ, n, q, data, iip)
    end
    return nothing
end
function evaluate_prediction_function!(q_vals, ranges, splines, θ::Dict, prof_idx::Vector, q, data, iip::Val{B}=Val(isinplace(q, 3)), parallel::Val{true}=Val(true)) where {B}
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

@inline function build_q_prototype(q, prof::Union{ProfileLikelihoodSolution,BivariateProfileLikelihoodSolution}, data)
    pred = q(get_mle(get_likelihood_solution(prof)), data)
    return zero(pred)
end
@inline function build_q_prototype(q, prof::Union{ProfileLikelihoodSolutionView,BivariateProfileLikelihoodSolutionView}, data)
    return build_q_prototype(q, get_parent(prof), data)
end

@inline function spline_other_mles(parameter_values, other_mles)
    spline = interpolate((parameter_values,), other_mles, Gridded(Linear()))
    return spline
end
@inline function spline_other_mles(parameter_values_1, parameter_values_2, other_mles)
    spline = interpolate((parameter_values_1, parameter_values_2), other_mles, Gridded(Linear()))
    return spline
end
@inline function spline_other_mles(prof::ProfileLikelihoodSolutionView)
    data = get_parameter_values(prof)
    other_mles = get_other_mles(prof)
    spline = spline_other_mles(data, other_mles)
    return spline
end
@inline function spline_other_mles(prof::BivariateProfileLikelihoodSolutionView)
    grid_1 = get_parameter_values(prof, 1)
    grid_2 = get_parameter_values(prof, 2)
    other_mles = get_other_mles(prof)
    spline = spline_other_mles(grid_1, grid_2, other_mles)
    return spline
end
@inline function spline_other_mles(prof::Union{ProfileLikelihoodSolution,BivariateProfileLikelihoodSolution})
    param_idx = profiled_parameters(prof)
    splines = Dict(param_idx .=> [spline_other_mles(prof[i]) for i in param_idx])
    return splines
end

@inline function build_θ!(θ, n::Integer, spline, ψ)
    @views θ[Not(n)] .= spline(ψ)
    θ[n] = ψ
    return nothing
end
@inline function build_θ!(θ, (n, m), spline, (ψ, ϕ))
    @views θ[Not(n, m)] .= spline(ψ, ϕ)
    θ[n] = ψ
    θ[m] = ϕ
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

@inline function get_prediction_intervals(prof_idx::AbstractVector{T}, q_lower_bounds, q_upper_bounds,
    q_union_lower_bounds::AbstractArray{F}, q_union_upper_bounds, level::L) where {T,F,L}
    individual_intervals = Dict{T,Vector{ConfidenceInterval{F,L}}}(prof_idx .=> [[ConfidenceInterval{F,L}(a, b, level) for (a, b) in zip(q_lower_bounds[i], q_upper_bounds[i])] for i in prof_idx])
    union_intervals = [ConfidenceInterval{F,L}(a, b, level) for (a, b) in zip(q_union_lower_bounds, q_union_upper_bounds)]
    return individual_intervals, union_intervals
end