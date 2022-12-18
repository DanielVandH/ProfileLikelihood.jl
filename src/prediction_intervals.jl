"""
    get_prediction_intervals(q, prof::ProfileLikelihoodSolution, data;
        q_type=get_q_type(q, prof, data), resolution=250)

Given a prediction of the form `q(θ, data)`, where `θ` has the same size as the `θ` used in 
the profile likelihood solution `prof`, and `data` is the argument `data`, computes the prediction 
intervals for `q` (possibly at each point if it outputs a vector) using the confidence intervals from `prof`.
You can set the output of the prediction function using `q_type`, and the grid resolution when evaluating the prediction 
function for each parameter via `resolution`.

Two results are produced:

- `parameterwise_cis`: This is a `Dict` mapping parameter indices to a a vector of confidence intervals from each output of `q` for the corresponding parameter. 
- `union_cis`: This gives the union of the intervals from `parameterwise_cis` (just taking the extrema over each interal) at each output of `q`.

We also return `all_curves`, a `Dict` mapping parameter indices to the vector of `q` values for each parameter, 
and these parameter ranges are given in `param_ranges`. So, the final output looks like:

`(parameterwise_cis, union_cis, all_curves, param_ranges)`.
"""
function get_prediction_intervals(q, prof::ProfileLikelihoodSolution, data;
    q_type=get_q_type(q, prof, data),
    resolution=250)
    ## Evaluate the family of curves
    prof_idx = profiled_parameters(prof)
    param_ranges = Dict(prof_idx .=> [LinRange(get_confidence_intervals(prof[i])..., 250) for i in prof_idx])
    all_curves = eval_prediction_function(q, prof, data; param_ranges, q_type)
    q_type_num = number_type(all_curves[first(prof_idx)]) # assuming q_type is same for each parameter
    all_curves_matrix = Dict(prof_idx .=> [Matrix(reduce(hcat, all_curves[i])') for i in prof_idx]) # transpose to make following memory access contiguous 
    q_length = size(all_curves_matrix[first(prof_idx)], 2) # assuming length of each q is the same
    ## Prepare the bound caches
    lower_bounds = Dict(prof_idx .=> [typemax(q_type_num) .* ones(q_type_num, q_length) for _ in prof_idx])
    upper_bounds = Dict(prof_idx .=> [typemin(q_type_num) .* ones(q_type_num, q_length) for _ in prof_idx])
    union_lower_bounds = typemax(q_type_num) .* ones(q_type_num, q_length)
    union_upper_bounds = typemin(q_type_num) .* ones(q_type_num, q_length)
    for k in prof_idx
        for j in 1:q_length
            for i in 1:resolution
                if all_curves_matrix[k][i, j] < lower_bounds[k][j]
                    lower_bounds[k][j] = all_curves_matrix[k][i, j]
                    if lower_bounds[k][j] < union_lower_bounds[j]
                        union_lower_bounds[j] = lower_bounds[k][j]
                    end
                end # keeps the Ifs disjoint so that the initial lb/ub get replaced
                if all_curves_matrix[k][i, j] > upper_bounds[k][j]
                    upper_bounds[k][j] = all_curves_matrix[k][i, j]
                    if upper_bounds[k][j] > union_upper_bounds[j]
                        union_upper_bounds[j] = upper_bounds[k][j]
                    end
                end
            end
        end
    end
    ## Build the prediction intervals 
    level = get_level(get_confidence_intervals(prof[first(prof_idx)]))
    parameterwise_cis = Dict(prof_idx .=> [[ConfidenceInterval(a, b, level) for (a, b) in zip(lower_bounds[i], upper_bounds[i])] for i in prof_idx])
    union_cis = [ConfidenceInterval(a, b, level) for (a, b) in zip(union_lower_bounds, union_upper_bounds)]
    return parameterwise_cis, union_cis, all_curves, param_ranges
end

"""
    eval_prediction_function(q, prof::ProfileLikelihoodSolution, data;
        resolution=250,
        param_ranges=Dict(profiled_parameters(prof) .=> [LinRange(get_confidence_intervals(prof[i])..., resolution) for i in profiled_parameters(prof)]),
        q_type=get_q_type(q, prof, data))

Given a prediction of the form `q(θ, data)`, where `θ` has the same size as the `θ` used in 
the profile likelihood solution `prof`, and `data` is the argument `data`, and for each parameter index
`i`: Evaluates `q((ψ, ωˢ(ψ)), data)`, where `ψ` ranges over `param_ranges[i]` and `ωˢ(ψ)` are the parameter values 
that lead to the value of the profile likelihood function where `θ[i] = ψ`. 

You can set the type of the output from the prediction function `q` using `q_type`.

The output is a `Dict` mapping the profiled parameter indices (from [`profiled_parameters`](@ref)) to the outputs from `q` at each 
corresponding parameter in `param_ranges`.
"""
function eval_prediction_function(q, prof::ProfileLikelihoodSolution, data;
    resolution=250,
    param_ranges=Dict(profiled_parameters(prof) .=> [LinRange(get_confidence_intervals(prof[i])..., resolution) for i in profiled_parameters(prof)]),
    q_type=get_q_type(q, prof, data))
    prof_idx = profiled_parameters(prof)
    q_vals = Dict{Int64,Vector{q_type}}()
    for i in prof_idx
        q_vals[i] = eval_prediction_function(q, prof[i], param_ranges[i], data; q_type)
    end
    return q_vals
end

function eval_prediction_function(q, prof::ProfileLikelihoodSolutionView{N,PLS}, ψ, data;
    q_type=get_q_type(q, prof, data)) where {N,PLS}
    splines = spline_other_mles(prof)
    θ = zeros(number_of_parameters(get_parent(prof).likelihood_problem))
    q_vals = Vector{q_type}(undef, length(ψ))
    for (i, ψ) in pairs(ψ)
        build_θ!(θ, N, splines, ψ)
        q_vals[i] = q(θ, data)
    end
    return q_vals
end

function spline_other_mles(prof::ProfileLikelihoodSolutionView{N,PLS}) where {N,PLS}
    data = get_parameter_values(prof)
    other_mles = reduce(hcat, get_other_mles(prof))
    splines = Vector{Interpolations.GriddedInterpolation{Float64,1,Vector{Float64},Gridded{Linear{Throw{OnGrid}}},Tuple{Vector{Float64}}}}(undef, size(other_mles, 1))
    for i in axes(other_mles, 1)
        parameter_vals = other_mles[i, :]
        itp = interpolate((data,), parameter_vals, Gridded(Linear()))
        splines[i] = itp
    end
    return splines
end
function spline_other_mles(prof::ProfileLikelihoodSolution)
    param_idx = profiled_parameters(prof)
    other_mle_splines = Dict(param_idx .=> [spline_other_mles(prof[i]) for i in param_idx])
    return other_mle_splines
end

function eval_other_mles_spline!(other_mles, splines::AbstractVector, ψ)
    @inbounds for j in eachindex(splines)
        other_mles[j] = splines[j](ψ)
    end
    return nothing
end

function build_θ!(θ, n, splines::AbstractVector, ψ)
    @views eval_other_mles_spline!(θ[Not(n)], splines, ψ)
    θ[n] = ψ
    return nothing
end

function get_q_type(q, prof::ProfileLikelihoodSolution, data)
    q_type = typeof(q(prof.likelihood_solution.mle, data))
    return q_type
end
function get_q_type(q, prof::ProfileLikelihoodSolutionView{N,PLS}, data) where {N,PLS}
    return get_q_type(q, get_parent(prof), data)
end