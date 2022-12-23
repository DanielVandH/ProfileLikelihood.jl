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
    profile_vals = OffsetArray(Matrix{T}(undef, 2res + 1, 2res + 1), -res:res, -res:res)
    other_mles = OffsetArray(Matrix{Vector{T}}(undef, 2res + 1, 2res + 1), -res:res, -res:res)
    cache = DiffCache(zeros(T, num_params))
    sub_cache = zeros(T, num_params - 2)
    sub_cache .= mles[Not(n[1], n[2])]
    fixed_vals = zeros(T, 2)
    profile_vals[0, 0] = normalise ? zero(T) : ℓmax
    other_mles[0, 0] = mles[Not(n[1], n[2])]
    return profile_vals, other_mles, cache, sub_cache, fixed_vals
end

function bivariate_profile(prob::LikelihoodProblem, sol::LikelihoodSolution, n::NTuple{M,NTuple{2,I}};
    alg=get_optimiser(sol),
    conf_level::F=0.95,
    confidence_interval_method=:contour,
    threshold=get_chisq_threshold(conf_level, 2),
    resolution=200,
    param_ranges=construct_profile_grids(n, sol, get_lower_bounds(prob), get_upper_bounds(prob), resolution),
    min_layers=10,
    normalise::Bool=true,
    parallel=false,
    next_initial_estimate_method=:mle,
    kwargs...) where {M,I,F}
    ## Extract the problem and solution 
    opt_prob, mles, ℓmax = extract_problem_and_solution(prob, sol)

    ## Prepare the profile results 
    T = number_type(mles)
    θ, prof, other_mles, interpolants, confidence_regions = prepare_bivariate_profile_results(M, T, F)
    num_params = number_of_parameters(opt_prob)

    ## Normalise the objective function 
    shifted_opt_prob = ProfileLikelihood.normalise_objective_function(opt_prob, ℓmax, normalise)

    ## Profile each parameter 
    for _n in n
        ## Setup
        grid = grids[_n]
        res = grid.resolutions
        profile_vals, other_mle, cache, sub_cache, fixed_vals = prepare_cache_vectors(_n, mles, res, num_params, normalise, ℓmax)
        restricted_prob = exclude_parameter(shifted_opt_prob, _n)

        ## Evolve outwards 
        layer = 1
        final_layer = res
        for i in 1:res
            layer_iterator = ProfileLikelihood.LayerIterator(layer)
            any_above_threshold = false
            for I in layer_iterator
                get_parameters!(fixed_vals, grid, I)
                fixed_prob = construct_fixed_optimisation_function(restricted_prob, _n, fixed_vals, cache)
                fixed_prob.u0 .= mles[Not(_n[1], _n[2])]
                soln = solve(fixed_prob, alg)
                profile_vals[I] = -soln.objective - ℓmax * !normalise
                other_mles[I] = soln.u
                if !any_above_threshold && profile_vals[I] > threshold
                    any_above_threshold = true
                end
            end
            if !any_above_threshold
                final_layer = layer
                break
            end
            layer += 1
        end

        ## Resize the arrays 
        profile_vals = OffsetArray(profile_vals[-final_layer:final_layer, -final_layer:final_layer], -final_layer:final_layer, -final_layer:final_layer)
        other_mles = OffsetArray(other_mle[-final_layer:final_layer, -final_layer:final_layer], -final_layer:final_layer, -final_layer:final_layer)
        param_1_range = get_range(grid, 1, -final_layer, final_layer)
        param_2_range = get_range(grid, 2, -final_layer, final_layer)

        ## Make the contours 
        c = Contour.contour(param_1_range, param_2_range, profile_vals, threshold)
        all_coords = reduce(vcat, [reduce(hcat, coordinates(xy)) for xy in Contour.lines(c)])
        region_x = all_coords[:, 1]
        region_y = all_coords[:, 2]
        region = ProfileLikelihood.ConfidenceRegion(region_x, region_y, conf_level)

        ## Make the interpolant 
        interpolant = Interpolations.interpolate((param_1_range, param_2_range), profile_vals, Gridded(Linear()))

        ## Fill out the results
    end


end
