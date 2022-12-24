function construct_profile_grids(n::NTuple{N,NTuple{2,I}}, sol, lower_bounds, upper_bounds, resolutions) where {N,I}
    grids = Dict{NTuple{2,I},FusedRegularGrid{
        2,
        Vector{number_type(lower_bounds)},
        Vector{I},
        Vector{number_type(lower_bounds)},
        number_type(lower_bounds),
        Vector{number_type(lower_bounds)}
    }}()
    res = get_resolution_tuples(resolutions, number_of_parameters(sol))
    for (n₁, n₂) in n
        if any(isinf, lower_bounds[[n₁, n₂]]) || any(isinf, upper_bounds[[n₁, n₂]])
            throw("The provided parameter bounds for $n₁ and $n₂ must be finite.")
        end
        grids[(n₁, n₂)] = FusedRegularGrid(
            [lower_bounds[n₁], lower_bounds[n₂]],
            [upper_bounds[n₁], upper_bounds[n₂]],
            [sol[n₁], sol[n₂]],
            max(res[n₁][1], res[n₁][2], res[n₂][1], res[n₂][2]);
            store_original_resolutions=true
        )
    end
    return grids
end

struct LayerIterator{N,B,T} # Don't use Iterators.flatten as it cannot infer when we use repeated (this was originally implemented as being a collection of `zip`s, e.g. the bottom row was `zip(-layer_number:layer_number, Iterators.repeated(-layer_number, 2layer_number+1))`, but this returns `Any` type)
    bottom::B
    right::B
    top::T
    left::T
    function LayerIterator(layer_number)
        itr1 = -layer_number:layer_number # UnitRanges
        itr2 = (-layer_number+1):layer_number
        itr3 = (layer_number-1):-1:-layer_number # StepRanges
        itr4 = (layer_number-1):-1:(-layer_number+1)
        return new{layer_number,typeof(itr1),typeof(itr3)}(itr1, itr2, itr3, itr4)
    end
end
Base.eltype(::Type{LayerIterator{N,B,T}}) where {N,B,T} = CartesianIndex{2}
Base.length(::LayerIterator{N,B,T}) where {N,B,T} = 8N
function Base.iterate(layer::LayerIterator{N,B,T}, state=1) where {N,B,T}
    if 1 ≤ state ≤ 2N + 1
        return (CartesianIndex(layer.bottom[state], -N), state + 1)
    elseif 2N + 2 ≤ state ≤ 4N + 1
        return (CartesianIndex(N, layer.right[state-2N-1]), state + 1)
    elseif 4N + 2 ≤ state ≤ 6N + 1
        return (CartesianIndex(layer.top[state-4N-1], N), state + 1)
    elseif 6N + 2 ≤ state ≤ 8N
        return (CartesianIndex(-N, layer.left[state-6N-1]), state + 1)
    else
        return nothing
    end
end

function prepare_bivariate_profile_results(N, T, F)
    θ = Dict{NTuple{2,Int64},NTuple{2,OffsetVector{T,Vector{T}}}}([])
    prof = Dict{NTuple{2,Int64},OffsetMatrix{T,Matrix{T}}}([])
    other_mles = Dict{NTuple{2,Int64},OffsetMatrix{Vector{T},Matrix{Vector{T}}}}([])
    interpolants = Dict{NTuple{2,Int64},Interpolations.GriddedInterpolation{T,2,OffsetMatrix{T,Matrix{T}},Gridded{Linear{Throw{OnGrid}}},Tuple{OffsetVector{T,Vector{T}},OffsetVector{T,Vector{T}}}}}([])
    confidence_regions = Dict{NTuple{2,Int64},ConfidenceRegion{Vector{T},F}}([])
    sizehint!(θ, N)
    sizehint!(prof, N)
    sizehint!(other_mles, N)
    sizehint!(interpolants, N)
    sizehint!(confidence_regions, N)
    return θ, prof, other_mles, interpolants, confidence_regions
end

function prepare_cache_vectors(n, mles::AbstractVector{T}, res, num_params, normalise, ℓmax) where {T}
    profile_vals = OffsetArray(zeros(T, 2res + 1, 2res + 1), -res:res, -res:res)
    other_mles = OffsetArray([zeros(T, num_params - 2) for _ in 1:(2res+1), _ in 1:(2res+1)], -res:res, -res:res)
    cache = DiffCache(zeros(T, num_params))
    sub_cache = zeros(T, num_params - 2)
    sub_cache .= mles[Not(n[1], n[2])]
    fixed_vals = zeros(T, 2)
    profile_vals[0, 0] = normalise ? zero(T) : ℓmax
    other_mles[0, 0] .= mles[Not(n[1], n[2])]
    fixed_value_pairs = OffsetArray([(zero(T), zero(T)) for _ in 1:(2res+1), _ in 1:(2res+1)], -res:res, -res:res)
    fixed_value_pairs[0, 0] = (mles[n[1]], mles[n[2]])
    return profile_vals, other_mles, cache, sub_cache, fixed_vals, fixed_value_pairs
end

function bivariate_profile(prob::LikelihoodProblem, sol::LikelihoodSolution, n::NTuple{M,NTuple{2,Int64}};
    alg=get_optimiser(sol),
    conf_level::F=0.95,
    confidence_region_method=:contour,
    threshold=get_chisq_threshold(conf_level, 2),
    resolution=200,
    grids=construct_profile_grids(n, sol, get_lower_bounds(prob), get_upper_bounds(prob), resolution),
    min_layers=10,
    outer_layers=0,
    normalise::Bool=true,
    parallel=false,
    next_initial_estimate_method=:mle,
    kwargs...) where {M,F}
    ## Extract the problem and solution 
    opt_prob, mles, ℓmax = extract_problem_and_solution(prob, sol)

    ## Prepare the profile results 
    T = number_type(mles)
    θ, prof, other_mles, interpolants, confidence_regions = prepare_bivariate_profile_results(M, T, F)
    num_params = number_of_parameters(opt_prob)

    ## Normalise the objective function 
    shifted_opt_prob = normalise_objective_function(opt_prob, ℓmax, normalise)

    ## Profile each parameter 
    for _n in n
        profile_single_pair!(θ, prof, other_mles, confidence_regions, interpolants, grids, _n,
            mles, num_params, normalise, ℓmax, shifted_opt_prob, alg, threshold, outer_layers, min_layers,
            conf_level, confidence_region_method, next_initial_estimate_method)
    end
    results = BivariateProfileLikelihoodSolution(θ, prof, prob, sol, interpolants, confidence_regions, other_mles)
    return results
end

function profile_single_pair!(θ, prof, other_mles, confidence_regions, interpolants, grids, n,
    mles, num_params, normalise, ℓmax, shifted_opt_prob, alg, threshold, outer_layers, min_layers,
    conf_level, confidence_region_method, next_initial_estimate_method)
    ## Setup
    grid = grids[n]
    res = grid.resolutions
    profile_vals, other_mle, cache, sub_cache, fixed_vals, combined_grid = prepare_cache_vectors(n, mles, res, num_params, normalise, ℓmax)
    restricted_prob = exclude_parameter(shifted_opt_prob, n)

    ## Evolve outwards 
    layer = 1
    final_layer = res
    outer_layer = 0
    for i in 1:res
        any_above_threshold = expand_layer!(fixed_vals, profile_vals, other_mle, cache, layer, n,
            grid, restricted_prob, alg, ℓmax, normalise, threshold, sub_cache, next_initial_estimate_method,
            combined_grid)
        if !any_above_threshold
            final_layer = layer
            outer_layer += 1
            outer_layer ≥ outer_layers && layer ≥ min_layers && break
        end
        layer += 1
    end

    ## Resize the arrays 
    range_1 = get_range(grid, 1, -final_layer, final_layer)
    range_2 = get_range(grid, 2, -final_layer, final_layer)
    resize_results!(θ, prof, other_mles, n, final_layer, profile_vals, other_mle, range_1, range_2)
    get_confidence_regions!(confidence_regions, n, range_1, range_2, prof[n], threshold, conf_level, final_layer, confidence_region_method)
    interpolate_profile!(interpolants, n, range_1, range_2, prof[n])
    return nothing
end

function expand_layer!(fixed_vals, profile_vals, other_mle, cache, layer, n, grid, restricted_prob, alg,
    ℓmax, normalise, threshold, sub_cache, next_initial_estimate_method, combined_grid)
    layer_iterator = LayerIterator(layer)
    any_above_threshold = false
    for I in layer_iterator
        get_parameters!(fixed_vals, grid, I)
        combined_grid[I] = (fixed_vals[1], fixed_vals[2])
        set_next_initial_estimate!(sub_cache, other_mle, I, fixed_vals, grid, layer, combined_grid; next_initial_estimate_method)
        fixed_prob = construct_fixed_optimisation_function(restricted_prob, n, fixed_vals, cache)
        fixed_prob.u0 .= sub_cache
        soln = solve(fixed_prob, alg)
        profile_vals[I] = -soln.objective - ℓmax * !normalise
        other_mle[I] = soln.u
        if !any_above_threshold && profile_vals[I] > threshold
            any_above_threshold = true
        end
    end
    return any_above_threshold
end

function resize_results!(parameter_values, profile_values, other_mles, n, final_layer, prof, other_mle, range_1, range_2)
    profile_values[n] = OffsetArray(prof[-final_layer:final_layer, -final_layer:final_layer], -final_layer:final_layer, -final_layer:final_layer)
    other_mles[n] = OffsetArray(other_mle[-final_layer:final_layer, -final_layer:final_layer], -final_layer:final_layer, -final_layer:final_layer)
    parameter_values[n] = (range_1, range_2)
    return nothing
end

function get_confidence_regions!(confidence_regions, n, range_1, range_2, profile_values, threshold, conf_level, final_layer, method)
    if method == :contour
        _get_confidence_regions_contour!(confidence_regions, n, range_1, range_2, profile_values, threshold, conf_level, final_layer)
    else
        throw("Invalid confidence region method, $method, specified.")
    end
    return nothing
end

function _get_confidence_regions_contour!(confidence_regions, n, range_1, range_2, profile_values, threshold, conf_level, final_layer)
    c = Contour.contour(range_1, range_2, profile_values, threshold)
    all_coords = reduce(vcat, [reduce(hcat, coordinates(xy)) for xy in Contour.lines(c)])
    region_x = all_coords[:, 1]
    region_y = all_coords[:, 2]
    reverse!(region_x)
    reverse!(region_y)
    pop!(region_x)
    pop!(region_y) # contour keeps the last value as being the same as the first
    confidence_regions[n] = ConfidenceRegion(region_x, region_y, conf_level)
    return nothing
end

function interpolate_profile!(interpolants, n, range_1, range_2, profile_values)
    interpolants[n] = interpolate((range_1, range_2), profile_values, Gridded(Linear()))
    return nothing
end

function set_next_initial_estimate!(sub_cache, other_mles, I::CartesianIndex, fixed_vals, grid, layer, combined_grid; next_initial_estimate_method=:mle)
    if next_initial_estimate_method == :mle
        sub_cache .= other_mles[0, 0]
        return nothing
    end
end