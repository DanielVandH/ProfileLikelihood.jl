"""
    GridSearch{F<:Function,N,B,intype,G<:AbstractGrid{N,B,intype},outtype}

A struct for storing a `grid` and function `f` for performing a grid search to maximise `f`. See also
[`AbstractGrid`](@ref) and its subtypes ([`UniformGrid`](@ref) and [`LatinGrid`](@ref)) for the definitions of `grid`.

# Constructors 
We provide the following constructors:

    GridSearch(f, grid::AbstractGrid{N,B,intype}) where {N,B,intype}
    GridSearch(f, bounds, resolution)
    GridSearch(f, bounds, m, gens; use_threads=false)
    GridSearch(prob::AbstractLikelihoodProblem, bounds = bounds(prob; make_open=true), resolution)
    GridSearch(prob::AbstractLikelihoodProblem, bounds=bounds(prob;make_open=true),m,gens;use_threads=false)
"""
Base.@kwdef struct GridSearch{F<:Function,N,B,intype,G<:AbstractGrid{N,B,intype},outtype}
    f::F
    grid::G
end
function GridSearch(f, grid::AbstractGrid{N,B,intype}) where {N,B,intype}
    bound_mids = [(bounds(grid, i, 1) + bounds(grid, i, 2)) / 2 for i in 1:N]
    outtype = typeof(f(bound_mids))
    return GridSearch{typeof(f),N,B,intype,typeof(grid),outtype}(f, grid)
end
function GridSearch(f, bounds, resolution)
    grid = UniformGrid(bounds, resolution)
    return GridSearch(f, grid)
end
function GridSearch(f, bounds, m, gens; use_threads=false)
    grid = LatinGrid(bounds, m, gens; use_threads)
    return GridSearch(f, grid)
end
function GridSearch(prob::AbstractLikelihoodProblem, bounds=bounds(prob; make_open=true), resolution)
    grid = UniformGrid(bounds, resolution)
    p = data(prob)
    f = @inline θ -> prob.loglik(θ, p)
    return GridSearch(f, grid)
end
function GridSearch(prob::AbstractLikelihoodProblem, bounds=bounds(prob; make_open=true), m, gens; use_threads=false)
    grid = LatinGrid(bounds, m, gens; use_threads)
    p = data(prob)
    f = @inline θ -> prob.loglik(θ, p)
    return GridSearch(f, grid)
end

"""
    prepare_grid(grid::UniformGrid{N,B,R,S,T}) where {N,B,R,S,T}
    prepare_grid(::LatinGrid{N,M,B,G,T}) where {N,M,B,G,T}

Pre-allocates the grid of function values for the grid search.
"""
function prepare_grid end
function prepare_grid(grid::UniformGrid{N,B,R,S,T}) where {N,B,R,S,T}
    zeros(T, [resolution(grid, i) for i in 1:N]...) # can't just use grid.resolution... since resolution can be a scalar
end
function prepare_grid(::LatinGrid{N,M,B,G,T}) where {N,M,B,G,T}
    zeros(T, M)
end

"""
    grid_search(prob::GridSearch{F,N,B,intype,G,outtype}; save_res=Val(false)) where {F<:Function,N,B,R,S,intype,G<:UniformGrid{N,B,R,S,intype},outtype}
    grid_search(prob::GridSearch{F,N,B,intype,G,outtype}; save_res=Val(false)) where {F<:Function,N,M,B,intype,Gr,G<:LatinGrid{N,M,B,Gr,intype},outtype}
    grid_search(f, bounds, resolution; save_res = Val(false), find_max = Val(true))
    grid_search(f, bounds, m, gens; save_res = Val(false), find_max = Val(true), use_threads = false)
    grid_search(prob::AbstractLikelihoodProblem, bounds, resolution; save_res = Val(false))
    grid_search(prob::AbstractLikelihoodProblem, bounds, m, gens; save_res = Val(false), use_threads = false)

Perform a grid search, findnig the maximum of the givne function. See also [`GridSearch`](@ref), [`UniformGrid`](@ref), and 
[`LatinGrid`](@ref).
"""
function grid_search end
@generated function grid_search(prob::GridSearch{F,N,B,intype,G,outtype}; save_res=Val(false)) where {F<:Function,N,B,R,S,intype,G<:UniformGrid{N,B,R,S,intype},outtype} # need to do generated function for type stability
    quote
        f_max = typemin($outtype)
        x_argmax = zeros($intype, $N)
        cur_x = zeros($intype, $N)
        if save_res == Val(true)
            f_res = prepare_grid(prob.grid)
        end
        Base.Cartesian.@nloops $N i (d -> 1:resolution(prob.grid, d)) (d -> cur_x[d] = prob.grid[d, i_d]) begin # [N loops] [i index] [range over LinRanges] [set coordinates]
            f_val = prob.f(cur_x)
            if f_val > f_max
                x_argmax .= cur_x
                f_max = f_val
            end
            if save_res
                (Base.Cartesian.@nref $N f_res i) = f_val
            end
        end
        if save_res == Val(true)
            return f_max, x_argmax, f_res
        else
            return f_max, x_argmax
        end
    end
end
function grid_search(prob::GridSearch{F,N,B,intype,G,outtype}; save_res=Val(false)) where {F<:Function,N,M,B,intype,Gr,G<:LatinGrid{N,M,B,Gr,intype},outtype}
    f_max = typemin(outtype)
    x_argmax = zeros(intype, N)
    if save_res == Val(true)
        f_res = prepare_grid(prob.grid)
    end
    for (i, θ) in enumerate(eachcol(prob.grid))
        f_val = prob.f(θ)
        if f_val > f_max
            x_argmax .= cur_x
            f_max = f_val
        end
        if save_res
            f_res[i] = f_val
        end
    end
    if save_res == Val(true)
        return f_max, x_argmax, f_res
    else
        return f_max, x_argmax
    end
end
function grid_search(f, bounds, resolution; save_res = Val(false), find_max = Val(true))
    if find_max == Val(false)
        g = x -> -f(x)
        gs = GridSearch(g, bounds, resolution)
        res = grid_search(gs; save_res)
        if save_res == Val(true)
            return res[1], res[2], -res[3] 
        else 
            return res 
        end
    end
    gs = GridSearch(f, bounds, resolutions)
    return grid_search(gs; save_res)
end
function grid_search(f, bounds, m, gens; save_res = Val(false), find_max = Val(true), use_threads = false)
    if find_max == Val(false)
        g = x -> -f(x)
        gs = GridSearch(g, bounds, m, gens; use_threads)
        res = grid_search(gs; save_res)
        if save_res == Val(true)
            return res[1], res[2], -res[3] 
        else 
            return res 
        end
    end
    gs = GridSearch(f, bounds, resolutions)
    return grid_search(gs; save_res)
end
function grid_search(prob::AbstractLikelihoodProblem, bounds, resolution; save_res = Val(false))
    gs = GridSearch(prob, bounds, resolution)
    if save_res == Val(true)
        ℓ_max, θ_argmax, f_res = grid_search(gs; save_res)
    else 
        ℓ_max, θ_argmax = grid_search(gs; save_res)
    end
    sol = LikelihoodSolution(θ_argmax, prob, :UniformGridSearch, ℓ_max, :Success, nothing)
    save_res == Val(true) ? (sol, f_res) : sol
end
function grid_search(prob::AbstractLikelihoodProblem, bounds, m, gens; save_res = Val(false), use_threads = false)
    gs = GridSearch(prob, bounds, m, gens; use_threads)
    if save_res == Val(true)
        ℓ_max, θ_argmax, f_res = grid_search(gs; save_res)
    else 
        ℓ_max, θ_argmax = grid_search(gs; save_res)
    end
    sol = LikelihoodSolution(θ_argmax, prob, LatinGridSearch, ℓ_max, :Success, nothing)
    save_res == Val(true) ? (sol, f_res) : sol
end