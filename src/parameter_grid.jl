"""
    abstract type AbstractGrid{N,B,T}

Abstract type representing a grid of values. `N` is the number of parameters, 
`B` is the type of the bounds, and `T` is the element type of the parameters. 
"""
abstract type AbstractGrid{N,B,T} end

"""
    bounds(grid::AbstractGrid, i)
    bounds(grid::AbstractGrid, i, j)

Extracts the bounds for the `i`th parameter value, and the `j`th bound of this set of bounds if `j` is provided.
"""
@inline bounds(grid::AbstractGrid, i) = grid.bounds[i]
@doc (@doc bounds(::AbstractGrid, ::Any)) @inline bounds(grid::AbstractGrid, i, j) = bounds(grid, i)[j]

########################################################################################
## UniformGrid 
########################################################################################
"""
    struct UniformGrid{N,B,R,S,T} <: AbstractGrid{N,B,T}

Struct representing a grid of values with uniform spacing in each dimension. The type `N` is the 
number of parameters, and `T` is the element type of the parameters.

# Fields 
- `bounds::B`

The bounds for each parameter.
- `resolution::R`

The resolution for each parameter's grid.
- `steps::S`

The stepsize for each parameter.

# Constructor
We provide the constructor

    UniformGrid(bounds, resolution)
"""
Base.@kwdef struct UniformGrid{N,B,R,S,T} <: AbstractGrid{N,B,T}
    bounds::B
    resolution::R
    steps::S
end
function UniformGrid(bounds, resolution)
    finite_bounds(bounds)
    N = length(bounds)
    B = typeof(bounds)
    R = typeof(resolution)
    T = (eltype ∘ eltype)(bounds) # This gets the Tuple type, and then promotes the types, giving us the more general parameter type 
    steps = zeros(T, N)
    for i in 1:N
        if resolution isa Number
            steps[i] = (bounds[i][2] - bounds[i][1]) / (resolution - 1)
        else
            steps[i] = (bounds[i][2] - bounds[i][1]) / (resolution[i] - 1)
        end
    end
    S = typeof(steps)
    return UniformGrid{N,B,R,S,T}(bounds, resolution, steps)
end

"""
    resolution(grid::UniformGrid, i::Integer)
    resolution(grid::UniformGrid{N,B,R,S,T}, ::Integer) where {N,B,R<:Integer,S,T} 

Extract the resolution for the `i`th parameter value from the `grid`.
"""
function resolution end
@inline resolution(grid::UniformGrid, i::Integer) = grid.resolution[i]
@inline resolution(grid::UniformGrid{N,B,R,S,T}, ::Integer) where {N,B,R<:Integer,S,T} = grid.resolution

"""
    steps(grid::UniformGrid, i)

Extracts the stepsize for the `i`th parameter of the grid.
"""
@inline steps(grid::UniformGrid, i) = grid.steps[i]

"""
    checkbounds(grid::UniformGrid{N,B,R,S,T}, i, j) where {N,B,R,S,T}

Check if the values `i` and `j` match the set of indices over the grid.
"""
@inline function Base.checkbounds(grid::UniformGrid{N,B,R,S,T}, i, j) where {N,B,R,S,T}
    (1 ≤ i ≤ N) || Base.throw_boundserror(grid, i)
    (1 ≤ j ≤ resolution(grid, i)) || Base.throw_boundserror(grid, j)
    nothing
end

"""
    getindex(grid::UniformGrid{N,B,R,S,T}, i::Integer, j::Integer) where {N,B,R,S,T}

Extracts the `j`th value for the `i`th parameter in the given `grid`.
"""
@inline Base.@propagate_inbounds function Base.getindex(grid::UniformGrid{N,B,R,S,T}, i::Integer, j::Integer) where {N,B,R,S,T}
    i isa Bool && throw(ArgumentError("Invalid index: $i of type Bool."))
    j isa Bool && throw(ArgumentError("Invalid index: $j of type Bool."))
    @boundscheck Base.checkbounds(grid, i, j)
    step = (j - 1) * steps(grid, i)
    return T(bounds(grid, i, 1) + step)
end

########################################################################################
## LatinGrid 
########################################################################################
"""
    struct LatinGrid{N,M,B,G,T} <: AbstractGrid{N,B,T}

Struct representing a grid of values with defined according to a Latin hypercube 
design. The type `N` is the number of parameters, `M` is the number of 
parameter sets, and `T` is the element type of the parameters.

# Fields 
- `bounds::B`

The bounds for each parameter.
- `grid::G`

The grid of parameter values.

# Constructor
We provide the constructor

    LatinGrid(bounds, m, gens; use_threads = false)

where `m` is the number of parameter sets to generate, and `gens` is the number of generations 
when generating the hypercube design.
"""
Base.@kwdef struct LatinGrid{N,M,B,G,T} <: AbstractGrid{N,B,T}
    bounds::B
    grid::G
end
function LatinGrid(bounds, m, gens; use_threads=false)
    finite_bounds(bounds)
    N = length(bounds)
    T = (eltype ∘ eltype)(bounds) # This gets the Tuple type, and then promotes the types, giving us the more general parameter type 
    latin_grid = collect(get_lhc_params(m, N, gens, bounds; use_threads)')
    B = typeof(bounds)
    G = typeof(latin_grid)
    return LatinGrid{N,m,B,G,T}(bounds, latin_grid)
end

@inline Base.@propagate_inbounds Base.getindex(grid::LatinGrid, I...) = Base.getindex(grid.grid, I...)
