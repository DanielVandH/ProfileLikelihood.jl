######################################################
## RegularGrid 
######################################################
"""
    struct RegularGrid{N,B,R,S,T} <: AbstractGrid{N,B,T}

Struct for a grid in which each parameter is regularly spaced. 

# Fields 
- `lower_bounds::B`: Lower bounds for each parameter. 
- `upper_bounds::B`: Upper bounds for each parameter. 
- `resolution::R`: Number of grid points for each parameter. If `R <: Number`, then the same number of grid points is used for each parameter. 
- `step_sizes::S`: Grid spacing for each parameter. 

# Constructor 
You can construct a `RegularGrid` using `RegularGrid(lower_bounds, upper_bounds, resolution)`.
"""
Base.@kwdef struct RegularGrid{N,B,R,S,T} <: AbstractGrid{N,B,T}
    lower_bounds::B
    upper_bounds::B
    resolution::R
    step_sizes::S
end

function RegularGrid(lower_bounds::B, upper_bounds::B, resolution::R) where {B,R}
    N = length(lower_bounds)
    T = number_type(B)
    step_sizes = zeros(T, N)
    for i in 1:N
        step_sizes[i] = compute_step_size(lower_bounds, upper_bounds, resolution, i)
    end
    S = typeof(step_sizes)
    return RegularGrid{N,B,R,S,T}(lower_bounds, upper_bounds, resolution, step_sizes)
end

compute_step_size(lower_bounds::Number, upper_bounds::Number, resolution::Number) = (upper_bounds - lower_bounds) / (resolution - 1)
compute_step_size(lower_bounds, upper_bounds, resolution::Number, i) = compute_step_size(lower_bounds[i], upper_bounds[i], resolution)
compute_step_size(lower_bounds, upper_bounds, resolution, i) = compute_step_size(lower_bounds, upper_bounds, resolution[i], i)

get_step_sizes(grid::RegularGrid) = grid.step_sizes
get_step_sizes(grid::RegularGrid, i) = get_step_sizes(grid)[i]

get_step(grid::RegularGrid, i, j) = (j - 1) * get_step_sizes(grid, i)

get_resolutions(grid::RegularGrid) = grid.resolution
get_resolutions(grid::RegularGrid, i) = get_resolutions(grid::RegularGrid)[i]
get_resolutions(grid::RegularGrid{N,B,R,S,T}, i) where {N,B,R<:Number,S,T} = get_resolutions(grid)

increment_parameter(grid::RegularGrid{N,B,R,S,T}, i, j) where {N,B,R,S,T} = T(get_lower_bounds(grid, i) + get_step(grid, i, j))

@inline function Base.checkbounds(grid::RegularGrid, i, j)
    (1 ≤ i ≤ number_of_parameters(grid)) || Base.throw_boundserror(grid, i)
    (1 ≤ j ≤ get_resolutions(grid, i)) || Base.throw_boundserror(grid, j)
    nothing
end
@inline Base.@propagate_inbounds function Base.getindex(grid::RegularGrid, i::Integer, j::Integer)
    i isa Bool && throw(ArgumentError("Invalid index: $i of type Bool."))
    j isa Bool && throw(ArgumentError("Invalid index: $j of type Bool."))
    @boundscheck Base.checkbounds(grid, i, j)
    return increment_parameter(grid, i, j)
end
get_range(grid::RegularGrid, i) = LinRange(get_lower_bounds(grid, i), get_upper_bounds(grid, i), get_resolutions(grid, i))

function get_parameters!(θ, grid::RegularGrid{N,B,R,S,T}, I) where {N,B,R,S,T}
    for d in 1:N
        θ[d] = grid[d, I[d]]
    end
    return nothing
end
function get_parameters(grid::RegularGrid{N,B,R,S,T}, I) where {N,B,R,S,T}
    θ = zeros(T, N)
    get_parameters!(θ, grid, I)
    return θ
end

######################################################
## FusedRegularGrid 
######################################################
"""
    struct FusedRegularGrid{N,B,R,S,T,C,OR} <: AbstractGrid{N,B,T}

Struct representing the fusing of two grids.

# Fields 
- `positive_grid::RegularGrid{N,B,R,S,T}`

This is the first part of the grid, indexed into by positive integers. 
- `negative_grid::RegularGrid{N,B,R,S,T}`

This is the second part of the grid, indexed into by negative integers.
- `centre::C`

The two grids meet at a common centre, and this is that `centre`. 

- `resolutions::R`

This is the vector of resolutions provided (e.g. if `store_original_resolutions=true` in the constructor below), or the transformed 
version from `get_resolution_tuples`.

# Constructor 
You can construct a `FusedRegularGrid` using the method

    FusedRegularGrid(lower_bounds::B, upper_bounds::B, centre::C, resolutions::R; store_original_resolutions=false) where {B,R,C}

For example, the following code creates `fused` as the fusion of `grid_1` and `grid_2`:

```jula-repl
lb = [2.0, 3.0, 1.0, 5.0]
ub = [15.0, 13.0, 27.0, 10.0]
centre = [7.3, 8.3, 2.3, 7.5]
grid_1 = RegularGrid(centre .+ (ub .- centre) / 173, ub, 173)
grid_2 = RegularGrid(centre .- (centre .- lb) / 173, lb, 173)
fused = ProfileLikelihood.FusedRegularGrid(lb, ub, centre, 173)
```

There are `173` points to the left and right of `centre` in this case. To use a varying 
number of points, use e.g.

```julia-repl 
lb = [2.0, 3.0, 1.0, 5.0, 4.0]
ub = [15.0, 13.0, 27.0, 10.0, 13.0]
centre = [7.3, 8.3, 2.3, 7.5, 10.0]
res = [(2, 11), (23, 25), (19, 21), (50, 51), (17, 99)]
grid_1 = RegularGrid(centre .+ (ub .- centre) ./ [2, 23, 19, 50, 17], ub, [2, 23, 19, 50, 17])
grid_2 = RegularGrid(centre .- (centre .- lb) ./ [11, 25, 21, 51, 99], lb, [11, 25, 21, 51, 99])
fused = ProfileLikelihood.FusedRegularGrid(lb, ub, centre, res) # fused grid_1 and grid_2
```
"""
Base.@kwdef struct FusedRegularGrid{N,B,R,S,T,C,OR} <: AbstractGrid{N,B,T}
    positive_grid::RegularGrid{N,B,R,S,T}
    negative_grid::RegularGrid{N,B,R,S,T}
    centre::C
    resolutions::OR
    function FusedRegularGrid(lower_bounds::B, upper_bounds::B, centre::C, resolutions::R; store_original_resolutions=false) where {B,C,R}
        res = get_resolution_tuples(resolutions, length(centre))
        grid_1 = RegularGrid(centre .+ (upper_bounds .- centre) ./ getindex.(res, 1), upper_bounds, getindex.(res, 1))
        grid_2 = RegularGrid(centre .+ (lower_bounds .- centre) ./ getindex.(res, 2), lower_bounds, getindex.(res, 2))
        stored_res = store_original_resolutions ? resolutions : res 
        N = number_of_parameters(grid_1)
        S = typeof(get_step_sizes(grid_1)) 
        R_type = typeof(stored_res)
        T = number_type(lower_bounds)
        return new{N,B,typeof(getindex.(res,1)),S,T,C,R_type}(grid_1,grid_2,centre,stored_res)
    end
end

function get_resolution_tuples(resolutions, N)
    if typeof(resolutions) <: Number
        return [(resolutions, resolutions) for _ in 1:N]
    elseif typeof(resolutions) <: Base.AbstractVecOrTuple
        res = Vector{NTuple{2,Int64}}(undef, N)
        for i in 1:N
            if typeof(resolutions[i]) <: Number
                res[i] = (resolutions[i], resolutions[i])
            else
                res[i] = (resolutions[i][1], resolutions[i][2])
            end
        end
        return res
    elseif typeof(resolutions) <: Tuple
        res = [(resolutions[1], resolutions[2]) for _ in 1:N]
        return res
    end
    throw("Invalid resolution specified.")
end

get_positive_grid(grid::FusedRegularGrid) = grid.positive_grid
get_negative_grid(grid::FusedRegularGrid) = grid.negative_grid
get_centre(grid::FusedRegularGrid) = grid.centre
get_centre(grid::FusedRegularGrid, i) = get_centre(grid)[i]

@inline get_lower_bounds(grid::FusedRegularGrid) = get_upper_bounds(get_negative_grid(grid))
@inline get_upper_bounds(grid::FusedRegularGrid) = get_upper_bounds(get_positive_grid(grid))

@inline Base.@propagate_inbounds function Base.getindex(grid::FusedRegularGrid, i::Integer, j::Integer)
    if j > 0
        return get_positive_grid(grid)[i, j]
    elseif j < 0
        return get_negative_grid(grid)[i, -j]
    else
        return get_centre(grid, i)
    end
end
function get_range(grid::FusedRegularGrid, i)
    rng = -get_resolutions(get_negative_grid(grid), i):get_resolutions(get_positive_grid(grid), i)
    return OffsetVector([grid[i, j] for j in rng], rng)
end

function get_parameters!(θ, grid::FusedRegularGrid{N,B,R,S,T,C}, I) where {N,B,R,S,T,C}
    for d in 1:N
        θ[d] = grid[d, I[d]]
    end
    return nothing
end
function get_parameters(grid::FusedRegularGrid{N,B,R,S,T,C}, I) where {N,B,R,S,T,C}
    θ = zeros(T, N)
    get_parameters!(θ, grid, I)
    return θ
end

######################################################
## IrregularGrid
######################################################
"""
    struct IrregularGrid{N,B,R,S,T} <: AbstractGrid{N,B,T}

Struct for an irregular grid of parameters.

# Fields 
- `lower_bounds::B`: Lower bounds for each parameter. 
- `upper_bounds::B`: Upper bounds for each parameter. 
- `grid::G`: The set of parameter values, e.g. a matrix where each column is the parameter vector.

# Constructor 
You can construct a `IrregularGrid` using `IrregularGrid(lower_bounds, upper_bounds, grid)`.
"""
Base.@kwdef struct IrregularGrid{N,B,G,T} <: AbstractGrid{N,B,T}
    lower_bounds::B
    upper_bounds::B
    grid::G
end
function IrregularGrid(lower_bounds::B, upper_bounds::B, grid::G) where {B,G}
    N = length(lower_bounds)
    T = number_type(G)
    return IrregularGrid{N,B,G,T}(lower_bounds, upper_bounds, grid)
end

get_grid(grid::IrregularGrid) = grid.grid

get_parameters(grid::AbstractMatrix, i) = @views grid[:, i]
get_parameters(grid::AbstractVector, i) = grid[i]
get_parameters(grid::IrregularGrid, i) = get_parameters(get_grid(grid), i)

@inline Base.@propagate_inbounds Base.getindex(grid::IrregularGrid, i) = get_parameters(grid, i)

number_of_parameter_sets(grid::AbstractMatrix) = size(grid, 2)
number_of_parameter_sets(grid::AbstractVector) = length(grid)
number_of_parameter_sets(grid::IrregularGrid) = number_of_parameter_sets(get_grid(grid))

each_parameter(grid::AbstractMatrix) = axes(grid, 2)
each_parameter(grid::AbstractVector) = eachindex(grid)
each_parameter(grid::IrregularGrid) = each_parameter(get_grid(grid))

function get_parameters!(θ, grid::IrregularGrid{N,B,G,T}, i) where {N,B,G,T}
    θ .= get_parameters(grid, i)
    return nothing
end

######################################################
## GridSearch
######################################################
"""
    struct GridSearch{F,G}

Struct for a `GridSearch`.

# Fields 
- `f::F`: The function to optimise, of the form `f(x, p)`.
- `p::P`: The arguments `p` in the function `f`.
- `grid::G`: The grid, where `G<:AbstractGrid`. See also [`grid_search`](@ref).
"""
Base.@kwdef struct GridSearch{F,G,P}
    f::F
    p::P
    grid::G
    function GridSearch(f, grid::G, p::P=nothing) where {G,P}
        T = number_type(G)
        wrapped_f = FunctionWrappers.FunctionWrapper{T,Tuple{Vector{T},P}}(f)
        return new{typeof(wrapped_f),G,P}(wrapped_f, p, grid)
    end
end

GridSearch(prob::LikelihoodProblem, grid) = GridSearch(get_log_likelihood_function(prob), grid, get_data(prob))

get_grid(prob::GridSearch) = prob.grid
get_function(prob::GridSearch) = prob.f
get_parameters(prob::GridSearch) = prob.p
eval_function(f::FunctionWrappers.FunctionWrapper{T,Tuple{A,P}}, x, p) where {T,A,P} = f(x, p)
eval_function(prob::GridSearch, x) = eval_function(get_function(prob), x, get_parameters(prob))
number_of_parameters(prob::GridSearch) = number_of_parameters(get_grid(prob))


"""
    grid_search(prob; save_vals=Val(false), minimise:=Val(false), parallel=Val(false))

Performs a grid search for the given grid search problem `prob`.

# Arguments 
- `prob::GridSearch{F, G}`: The grid search problem.

# Keyword Arguments 
- `save_vals:=Val(false)`: Whether to return a array with the function values. 
- `minimise:=Val(false)`: Whether to minimise or to maximise the function.
- `parallel:=Val(false)`: Whether to run the grid search with multithreading.

# Outputs 
- `f_opt`: The optimal objective value. 
- `x_argopt`: The parameter that gave `f_opt`.
- `f_res`: If `save_vals==Val(true)`, then this is the array of function values.
"""
@inline function grid_search(prob::P; save_vals::V=Val(false), minimise::M=Val(false), parallel::T=Val(false)) where {P,V,M,T}
    if parallel == Val(false)
        return grid_search_serial(prob, save_vals, minimise)
    else
        return grid_search_parallel(prob, save_vals, minimise)
    end
end

function grid_search_serial(prob, save_vals, minimise)
    f_opt, x_argopt, cur_x, grid_iterator, f_res = setup_grid_search_serial(prob; save_vals, minimise)
    for I in grid_iterator
        f_opt = step_grid_search!(prob, I, cur_x, x_argopt, f_opt, f_res; save_vals, minimise)
    end
    return grid_search_return_values(f_opt, x_argopt, f_res, save_vals)
end
function grid_search_parallel(prob, save_vals, minimise)
    nt = Base.Threads.nthreads()
    prob_copies, f_opt, x_argopt, cur_x, grid_iterator_collect, chunked_grid, f_res = setup_grid_search_parallel(prob; save_vals, minimise)
    Base.Threads.@threads for (cart_idx_range, id) in chunked_grid
        for linear_idx in cart_idx_range
            I = @inbounds grid_iterator_collect[linear_idx]
            @inbounds @views f_opt[id] = step_grid_search!(prob_copies[id], I, cur_x[id], x_argopt[id], f_opt[id], f_res; save_vals, minimise)
        end
    end
    multithreaded_find_extrema_for_grid_search!(nt, f_opt, x_argopt, minimise)
    return @inbounds grid_search_return_values(f_opt[1], x_argopt[1], f_res, save_vals)
end

"""
    grid_search(f, grid::AbstractGrid; save_vals=Val(false), minimise=Val(false), parallel=Val(false))

For a given `grid` and function `f`, performs a grid search. 

# Arguments 
- `f`: The function to optimise. 
- `grid::AbstractGrid`: The grid to use for optimising. 

# Keyword Arguments 
- `save_vals=Val(false)`: Whether to return a array with the function values. 
- `minimise=Val(false)`: Whether to minimise or to maximise the function.
- `parallel=Val(false)`: Whether to run the grid search with multithreading.
"""
grid_search(f, grid::AbstractGrid, p=nothing; save_vals=Val(false), minimise=Val(false), parallel=Val(false)) = grid_search(GridSearch(f, grid, p); save_vals, minimise, parallel)

"""
    grid_search(prob::LikelihoodProblem, grid::AbstractGrid, parallel=Val(false); save_vals=Val(false))

Given a `grid` and a likelihood problem `prob`, maximises it over the grid using a grid search. If 
`save_vals==Val(true)`, then the likelihood function values at each gridpoint are returned. Set 
`parallel=Val(true)` if you want multithreading.
"""
function grid_search(prob::LikelihoodProblem, grid::AbstractGrid; save_vals=Val(false), parallel=Val(false))
    if parallel == Val(false)
        gs = GridSearch(prob, grid)
    else # Don't put the full form below, with (f, g, p), since if f is a closure containing parameters that are aliased with p, we run into issues
        gs = [GridSearch(deepcopy(prob), deepcopy(grid)) for _ in 1:Base.Threads.nthreads()]
    end
    if save_vals == Val(false)
        ℓ_max, θ_mle = grid_search(gs; minimise=Val(false), save_vals, parallel)
        return LikelihoodSolution{number_of_parameters(prob),
            typeof(θ_mle),typeof(prob),typeof(ℓ_max),
            typeof(SciMLBase.ReturnCode.Success),Symbol}(θ_mle, prob, :GridSearch, ℓ_max, SciMLBase.ReturnCode.Success)
    else
        ℓ_max, θ_mle, f_res = grid_search(gs; minimise=Val(false), save_vals, parallel)
        return LikelihoodSolution{number_of_parameters(prob),
            typeof(θ_mle),typeof(prob),typeof(ℓ_max),
            typeof(SciMLBase.ReturnCode.Success),Symbol}(θ_mle, prob, :GridSearch, ℓ_max, SciMLBase.ReturnCode.Success), f_res
    end
end

@inline function step_grid_search!(prob::G, I, cur_x, x_argopt, f_opt, f_res; save_vals::V=Val(false), minimise::M=Val(false)) where {G,V,M}
    get_parameters!(cur_x, get_grid(prob), I)
    f_val = eval_function(prob, cur_x)
    f_opt = update_extremum!(x_argopt, f_val, cur_x, f_opt; minimise)
    if V == Val{true}
        @inbounds f_res[I] = f_val
    end
    return f_opt
end

function prepare_grid(grid::RegularGrid)
    T = number_type(grid)
    N = number_of_parameters(grid)
    return zeros(T, (get_resolutions(grid, i) for i in 1:N)...)
end
function prepare_grid(grid::IrregularGrid)
    T = number_type(grid)
    M = number_of_parameter_sets(grid)
    return zeros(T, M)
end
prepare_grid(prob::GridSearch) = prepare_grid(get_grid(prob))

@inline function setup_grid_search_serial(prob::GridSearch{F,G,P}; save_vals::V=Val(false), minimise::M=Val(false)) where {P,F,N,B,R,S,T,V,M,G<:RegularGrid{N,B,R,S,T}}
    f_opt = get_default_extremum(T, M)
    x_argopt = zeros(T, N)
    cur_x = zeros(T, N)
    if V == Val{true}
        f_res = prepare_grid(prob)
    end
    resolution_generator = (1:get_resolutions(get_grid(prob), d) for d in 1:N)
    grid_iterator = CartesianIndices((tuple(resolution_generator...)))
    if V == Val{true}
        return f_opt, x_argopt, cur_x, grid_iterator, f_res
    else
        return f_opt, x_argopt, cur_x, grid_iterator, T(NaN)
    end
end
@inline function setup_grid_search_serial(prob::GridSearch{F,G,P}; save_vals::V=Val(false), minimise::M=Val(false)) where {P,F,N,B,T,V,M,H,G<:IrregularGrid{N,B,H,T}}
    f_opt = get_default_extremum(T, M)
    x_argopt = zeros(T, N)
    cur_x = zeros(T, N)
    if V == Val{true}
        f_res = prepare_grid(prob)
    end
    grid_iterator = each_parameter(get_grid(prob))
    if V == Val{true}
        return f_opt, x_argopt, cur_x, grid_iterator, f_res
    else
        return f_opt, x_argopt, cur_x, grid_iterator, T(NaN)
    end
end
@inline function setup_grid_search_parallel(prob::Union{Vector{GridSearch{F,G,P}},GridSearch{F,G,P}}; save_vals::V=Val(false), minimise::M=Val(false)) where {P,F,N,B,R,S,T,V,M,G<:RegularGrid{N,B,R,S,T}}
    nt = Base.Threads.nthreads()
    f_opt = [get_default_extremum(T, M) for _ in 1:nt]
    x_argopt = [zeros(T, N) for _ in 1:nt]
    cur_x = [zeros(T, N) for _ in 1:nt]
    if V == Val{true}
        f_res = prepare_grid(prob isa Vector ? prob[1] : prob)
    end
    prob_copies = prob isa Vector ? prob : [deepcopy(prob) for _ in 1:nt]
    resolution_generator = (1:get_resolutions(get_grid(prob isa Vector ? prob[1] : prob), d) for d in 1:N)
    grid_iterator = CartesianIndices((tuple(resolution_generator...)))
    grid_iterator_collect = collect(grid_iterator)
    chunked_grid = chunks(grid_iterator_collect, nt)
    if V == Val{true}
        return prob_copies, f_opt, x_argopt, cur_x, grid_iterator_collect, chunked_grid, f_res
    else
        return prob_copies, f_opt, x_argopt, cur_x, grid_iterator_collect, chunked_grid, T(NaN)
    end
end
@inline function setup_grid_search_parallel(prob::Union{Vector{GridSearch{F,G,P}},GridSearch{F,G,P}}; save_vals::V=Val(false), minimise::M=Val(false)) where {P,F,N,B,T,V,M,H,G<:IrregularGrid{N,B,H,T}}
    nt = Base.Threads.nthreads()
    f_opt = [get_default_extremum(T, M) for _ in 1:nt]
    x_argopt = [zeros(T, N) for _ in 1:nt]
    cur_x = [zeros(T, N) for _ in 1:nt]
    if V == Val{true}
        f_res = prepare_grid(prob isa Vector ? prob[1] : prob)
    end
    prob_copies = prob isa Vector ? prob : [deepcopy(prob) for _ in 1:nt]
    grid_iterator = each_parameter(get_grid(prob isa Vector ? prob[1] : prob))
    grid_iterator_collect = collect(grid_iterator)
    chunked_grid = chunks(grid_iterator_collect, nt)
    if V == Val{true}
        return prob_copies, f_opt, x_argopt, cur_x, grid_iterator_collect, chunked_grid, f_res
    else
        return prob_copies, f_opt, x_argopt, cur_x, grid_iterator_collect, chunked_grid, T(NaN)
    end
end

@inline function multithreaded_find_extrema_for_grid_search!(nt, f_opt, x_argopt, minimise)
    for i in 2:nt
        if minimise == Val(false)
            if @inbounds f_opt[i] ≥ f_opt[1]
                @inbounds f_opt[1] = f_opt[i]
                @inbounds x_argopt[1] .= x_argopt[i]
            end
        else
            if @inbounds f_opt[i] ≤ f_opt[1]
                @inbounds f_opt[1] = f_opt[i]
                @inbounds x_argopt[1] .= x_argopt[i]
            end
        end
    end
    return nothing
end

@inline grid_search_return_values(f_opt, x_argopt, f_res, V::Val{false}) = f_opt, x_argopt
@inline grid_search_return_values(f_opt, x_argopt, f_res, V::Val{true}) = f_opt, x_argopt, f_res
