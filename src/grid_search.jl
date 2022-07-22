"""
    GridSearch{N,F<:Function,B,R,intype,outtype}

Struct for performing a grid search. See also [`grid_search`](@ref).

# Fields 
- `f::F<:Function`

The function `f` to find the maximum of, given as a function `x -> f(x)` where `length(x) = N`, `eltype(x) = intype`, and `typeof(f(x)) = outtype`.

- `bounds::B`

The bounds for the parameter values to search over, given as a vector of length `N` such that `bounds[i] = (lower, upper)` for `x[i]`.

- `resolution::R`

The number of gridpoints to use for each parameter; for `x[i]` this is given by `resolution[i]`.

# Constructors 
We provide the constructors:

    GridSearch(f, bounds, resolution)
    GridSearch(prob::LikelihoodProblem{ST,iip,F,θType,P,B,LC,UC,S,K,D,θ₀Type,ℓ}, bounds=bounds(prob; make_open=true), resolution=20)
"""
Base.@kwdef struct GridSearch{N,F<:Function,B,R,intype,outtype}
    f::F
    bounds::B
    resolution::R
end
function GridSearch(f, bounds, resolution)
    for i in 1:length(bounds)
        lower, upper = bounds[i]
        if isnothing(lower) || isnothing(upper) || isinf(lower) || isinf(upper)
            throw(ArgumentError("All bounds must be finite."))
        end
    end
    N = length(bounds)
    F = typeof(f)
    B = typeof(bounds)
    if resolution isa Number
        resolution = repeat([resolution], N)
    end
    R = typeof(resolution)
    bound_mids = [(bounds[i][1] + bounds[i][2]) / 2 for i in 1:N]
    intype = typeof(first(bound_mids))
    outtype = typeof(f(bound_mids))
    return GridSearch{N,F,B,R,intype,outtype}(f, bounds, resolution)
end
function GridSearch(prob::LikelihoodProblem{ST,iip,F,θType,P,B,LC,UC,S,K,D,θ₀Type,ℓ}, bounds=bounds(prob; make_open=true), resolution=20) where {ST,iip,F,θType,P,B,LC,UC,S,K,D,θ₀Type,ℓ}
    p = data(prob)
    f = θ -> prob.loglik(θ, p)
    return GridSearch(f, bounds, resolution)
end

"""
    grid_search(prob::GridSearch{N,F,B,R,intype,outtype}; save_res = false)
    grid_search(f, bounds, resolution; save_res=false, find_max=true) 
    grid_search(prob::LikelihoodProblem, bounds = bounds(prob; make_open=true), resolution = 20; save_res = false)

Perform a grid search, finding the maximum of the given function.

# Arguments 
- `prob`: The [`GridSearch`](@ref) or [`LikelihoodProblem`](@ref).
- `bounds`: Parameter bounds - see [`GridSearch`](@ref).
- `resolution`: Grid resolution for each parameter - see [`GridSearch`](@ref).

# Keyword Arguments 
- `save_res=false`: Whether to return the evaluated function values, given as a matrix `A[i₁, …, iₙ]`.
- `find_max=true`: Only for the second method, you can specify `find_max=false` to instead find a minimum.

# Output 
- If calling the first or second method, the output is `f_max, x_argmax` (and `f_res` if `save_res`), where `f_max` is the maximum function value and `x_argmax` is the position of the maximum. 
- If calling the third method, the output is a [`LikelihoodSolution`](@ref) (and `f_res` if `save_res`).
"""
function grid_search end
@generated function grid_search(prob::GridSearch{N,F,B,R,intype,outtype}; save_res::Bool=false) where {N,F,B,R,intype,outtype} # had to use generated because of type instabilities. Iterators.product wasn't good because of extra allocations (from type stabilities). generated functions solves it and also lets us use Base.Cartesian
    quote
        f_max = typemin($outtype)
        x_argmax = zeros($intype, $N)
        param_ranges = Vector{LinRange{$intype,Int64}}(undef, $N)
        for i in 1:$N
            param_ranges[i] = LinRange(prob.bounds[i][1], prob.bounds[i][2], prob.resolution[i])
        end
        cur_x = zeros($intype, $N)
        if save_res
            f_res = zeros($outtype, prob.resolution...)
        end
        Base.Cartesian.@nloops $N i (d -> 1:prob.resolution[d]) (d -> cur_x[d] = param_ranges[d][i_d]) begin # [N loops] [i index] [range over LinRanges] [set coordinates]
            f_val = prob.f(cur_x)
            if f_val > f_max
                x_argmax .= cur_x
                f_max = f_val
            end
            if save_res
                (Base.Cartesian.@nref $N f_res i) = f_val
            end
        end
        if save_res
            return f_max, x_argmax, f_res
        else
            return f_max, x_argmax
        end
    end
end
function grid_search(f, bounds, resolution; save_res::Bool=false, find_max::Bool=true)
    if !find_max
        g = x -> -f(x)
        return grid_search(GridSearch(g, bounds, resolution); save_res)
    else
        return grid_search(GridSearch(f, bounds, resolution); save_res)
    end
end
function grid_search(prob::LikelihoodProblem, bounds=bounds(prob; make_open=true), resolution=20; save_res::Bool=false)
    gs = GridSearch(prob, bounds, resolution)
    if save_res
        ℓ_max, θ_argmax, f_res = grid_search(gs; save_res)
    else
        ℓ_max, θ_argmax = grid_search(gs; save_res)
    end
    sol = LikelihoodSolution(θ_argmax, prob, :GridSearch, ℓ_max, :Success, nothing)
    save_res ? (sol, f_res) : sol
end