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

######################################################
## GridSearch
######################################################
"""
    struct GridSearch{F,G}

Struct for a `GridSearch`.

# Fields 
- `f::F`: The function to optimise. 
- `grid::G`: The grid, where `G<:AbstractGrid`. See also [`grid_search`](@ref).
"""
Base.@kwdef struct GridSearch{F,G}
    f::F
    grid::G
    function GridSearch(f, grid::G) where {G}
        T = number_type(G)
        wrapped_f = FunctionWrappers.FunctionWrapper{T,Tuple{Vector{T}}}(f)
        return new{typeof(wrapped_f),G}(wrapped_f, grid)
    end
end

GridSearch(prob::LikelihoodProblem, grid) = GridSearch(θ -> get_log_likelihood_function(prob)(θ, get_data(prob)), grid)

get_grid(prob::GridSearch) = prob.grid
get_function(prob::GridSearch) = prob.f
eval_function(f::FunctionWrappers.FunctionWrapper{T,Tuple{A}}, x) where {T,A} = f(x)
eval_function(prob::GridSearch, x) = eval_function(get_function(prob), x)
number_of_parameters(prob::GridSearch) = number_of_parameters(get_grid(prob))

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

"""
    grid_search(prob::GridSearch{F,G}; save_vals::V=Val(false), minimise::M=Val(false)) where {F,N,B,R,S,T,V,M,G<:RegularGrid{N,B,R,S,T}}
    grid_search(prob::GridSearch{F,G}; save_vals::V=Val(false), minimise::M=Val(false)) where {F,N,B,T,V,M,H,G<:IrregularGrid{N,B,H,T}}

Performs a grid search for the given grid search problem `prob`.

# Arguments 
- `prob::GridSearch{F, G}`: The grid search problem.

# Keyword Arguments 
- `save_vals::V=Val(false)`: Whether to return a matrix with the function values. 
- `minimise::M=Val(false)`: Whether to minimise or to maximise the function.
"""
@generated function grid_search(prob::GridSearch{F,G}; save_vals::V=Val(false), minimise::M=Val(false)) where {F,N,B,R,S,T,V,M,G<:RegularGrid{N,B,R,S,T}}
    quote
        f_opt = get_default_extremum($T, $M)
        x_argopt = zeros($T, $N)
        cur_x = zeros($T, $N)
        if $V == Val{true}
            f_res = prepare_grid(prob)
        end
        Base.Cartesian.@nloops $N i (d -> 1:get_resolutions(get_grid(prob), d)) (d -> cur_x[d] = get_grid(prob)[d, i_d]) begin # [N loops] [i index] [range over LinRanges] [set coordinates]
            f_val = eval_function(prob, cur_x)
            f_opt = update_extremum!(x_argopt, f_val, cur_x, f_opt; minimise)
            if $V == Val{true}
                (Base.Cartesian.@nref $N f_res i) = f_val
            end
        end
        if $V == Val{true}
            return f_opt, x_argopt, f_res
        else
            return f_opt, x_argopt
        end
    end
end
@doc (@doc grid_search(::GridSearch)) function grid_search(prob::GridSearch{F,G}; save_vals::V=Val(false), minimise::M=Val(false)) where {F,N,B,T,V,M,H,G<:IrregularGrid{N,B,H,T}}
    f_opt = get_default_extremum(T, M)
    x_argopt = zeros(T, N)
    cur_x = zeros(T, N)
    if V == Val{true}
        f_res = prepare_grid(prob)
    end
    for i in each_parameter(get_grid(prob))
        cur_x .= get_parameters(get_grid(prob), i)
        f_val = eval_function(prob, cur_x)
        f_opt = update_extremum!(x_argopt, f_val, cur_x, f_opt; minimise)
        if V == Val{true}
            f_res[i] = f_val
        end
    end
    if V == Val{true}
        return f_opt, x_argopt, f_res
    else
        return f_opt, x_argopt
    end
end

"""
    grid_search(f, grid::AbstractGrid; save_vals=Val(false), minimise=Val(false))

For a given `grid` and function `f`, performs a grid search. 

# Arguments 
- `f`: The function to optimise. 
- `grid::AbstractGrid`: The grid to use for optimising. 

# Keyword Arguments 
- `save_vals::V=Val(false)`: Whether to return a matrix with the function values. 
- `minimise::M=Val(false)`: Whether to minimise or to maximise the function.
"""
grid_search(f, grid::AbstractGrid; save_vals=Val(false), minimise=Val(false)) = grid_search(GridSearch(f, grid); save_vals, minimise)

"""
    grid_search(prob::LikelihoodProblem, grid::AbstractGrid; save_vals=Val(false))

Given a `grid` and a likelihood problem `prob`, maximises it over the grid using a grid search. If 
`save_vals==Val(true)`, then the likelihood function values at each gridpoint are returned.
"""
function grid_search(prob::LikelihoodProblem, grid::AbstractGrid; save_vals=Val(false))
    gs = GridSearch(prob, grid)
    if save_vals == Val(false)
        ℓ_max, θ_mle = grid_search(gs; minimise=Val(false), save_vals)
        return LikelihoodSolution(θ_mle, prob, :GridSearch, ℓ_max, SciMLBase.ReturnCode.Success)
    else
        ℓ_max, θ_mle, f_res = grid_search(gs; minimise=Val(false), save_vals)
        return LikelihoodSolution(θ_mle, prob, :GridSearch, ℓ_max, SciMLBase.ReturnCode.Success), f_res
    end
end