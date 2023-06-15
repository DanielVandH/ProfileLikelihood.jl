##########################################################
##
## UNIVARIATE
##
##########################################################
"""
    ProfileLikelihoodSolution{I,V,LP,LS,Spl,CT,CF,OM} 

Struct for the normalised profile log-likelihood. See [`profile`](@ref) for a constructor.

# Fields 
- `parameter_values::Dict{I, V}`

This is a dictionary such that `parameter_values[i]` gives the parameter values used for the normalised profile log-likelihood of the `i`th variable.
- `profile_values::Dict{I, V}`

This is a dictionary such that `profile_values[i]` gives the values of the normalised profile log-likelihood function at the corresponding values in `θ[i]`.
- `likelihood_problem::LP`

The original [`LikelihoodProblem`](@ref).
- `likelihood_solution::LS`

The solution to the full problem.
- `splines::Dict{I, Spl}`

This is a dictionary such that `splines[i]` is a spline through the data `(parameter_values[i], profile_values[i])`. This spline can be evaluated at a point `ψ` for the `i`th variable by calling an instance of the struct with arguments `(ψ, i)`. See also `spline_profile`.
- `confidence_intervals::Dict{I,ConfidenceInterval{CT,CF}}`

This is a dictonary such that `confidence_intervals[i]` is a confidence interval for the `i`th parameter.
- `other_mles::OM`

This is a dictionary such that `other_mles[i]` gives the vector for the MLEs of the other parameters not being profiled, for each datum.

# Spline evaluation 

This struct is callable. We define the method 

    (prof::ProfileLikelihoodSolution)(θ, i)

that evaluates the spline through the `i`th profile at the point `θ`.
"""
Base.@kwdef struct ProfileLikelihoodSolution{I,V,LP,LS,Spl,CT,CF,OM}
    parameter_values::Dict{I,V}
    profile_values::Dict{I,V}
    likelihood_problem::LP
    likelihood_solution::LS
    splines::Dict{I,Spl}
    confidence_intervals::Dict{I,ConfidenceInterval{CT,CF}}
    other_mles::OM
end

get_parameter_values(prof::ProfileLikelihoodSolution) = prof.parameter_values
get_parameter_values(prof::ProfileLikelihoodSolution, i) = get_parameter_values(prof)[i]
get_parameter_values(prof::ProfileLikelihoodSolution, sym::Symbol) = get_parameter_values(prof, SciMLBase.sym_to_index(sym, prof))
get_profile_values(prof::ProfileLikelihoodSolution) = prof.profile_values
get_profile_values(prof::ProfileLikelihoodSolution, i) = get_profile_values(prof)[i]
get_profile_values(prof::ProfileLikelihoodSolution, sym::Symbol) = get_profile_values(prof, SciMLBase.sym_to_index(sym, prof))
get_likelihood_problem(prof::ProfileLikelihoodSolution) = prof.likelihood_problem
get_likelihood_solution(prof::ProfileLikelihoodSolution) = prof.likelihood_solution
get_splines(prof::ProfileLikelihoodSolution) = prof.splines
get_splines(prof::ProfileLikelihoodSolution, i) = get_splines(prof)[i]
get_splines(prof::ProfileLikelihoodSolution, sym::Symbol) = get_splines(prof, SciMLBase.sym_to_index(sym, prof))
get_confidence_intervals(prof::ProfileLikelihoodSolution) = prof.confidence_intervals
get_confidence_intervals(prof::ProfileLikelihoodSolution, i) = get_confidence_intervals(prof)[i]
get_confidence_intervals(prof::ProfileLikelihoodSolution, sym::Symbol) = get_confidence_intervals(prof, SciMLBase.sym_to_index(sym, prof))
get_other_mles(prof::ProfileLikelihoodSolution) = prof.other_mles
get_other_mles(prof::ProfileLikelihoodSolution, i) = get_other_mles(prof)[i]
get_other_mles(prof::ProfileLikelihoodSolution, sym::Symbol) = get_other_mles(prof, SciMLBase.sym_to_index(sym, prof))
get_syms(prof::ProfileLikelihoodSolution) = get_syms(get_likelihood_problem(prof))
get_syms(prof::ProfileLikelihoodSolution, i) = get_syms(prof)[i]
profiled_parameters(prof::ProfileLikelihoodSolution) = (sort ∘ collect ∘ keys ∘ get_confidence_intervals)(prof)::Vector{Int}
number_of_profiled_parameters(prof::ProfileLikelihoodSolution) = length(profiled_parameters(prof))

struct ProfileLikelihoodSolutionView{N,PLS}
    parent::PLS
end

Base.getindex(prof::PLS, i::Integer) where {PLS<:ProfileLikelihoodSolution} = ProfileLikelihoodSolutionView{i,PLS}(prof)
SciMLBase.sym_to_index(sym, prof::ProfileLikelihoodSolution) = SciMLBase.sym_to_index(sym, get_syms(prof))
function Base.getindex(prof::ProfileLikelihoodSolution, sym)
    if SciMLBase.issymbollike(sym)
        i = SciMLBase.sym_to_index(sym, prof)
    else
        i = sym
    end
    if i === nothing
        throw(BoundsError(prof, sym))
    else
        prof[i]
    end
end

get_parent(prof::ProfileLikelihoodSolutionView) = prof.parent
get_index(::ProfileLikelihoodSolutionView{N,PLS}) where {N,PLS} = N
get_parameter_values(prof::ProfileLikelihoodSolutionView{N,PLS}) where {N,PLS} = get_parameter_values(get_parent(prof), N)
get_parameter_values(prof::ProfileLikelihoodSolutionView{N,PLS}, i) where {N,PLS} = get_parameter_values(prof)[i]
get_profile_values(prof::ProfileLikelihoodSolutionView{N,PLS}) where {N,PLS} = get_profile_values(get_parent(prof), N)
get_profile_values(prof::ProfileLikelihoodSolutionView{N,PLS}, i) where {N,PLS} = get_profile_values(prof)[i]
get_likelihood_problem(prof::ProfileLikelihoodSolutionView{N,PLS}) where {N,PLS} = get_likelihood_problem(get_parent(prof))
get_likelihood_solution(prof::ProfileLikelihoodSolutionView{N,PLS}) where {N,PLS} = get_likelihood_solution(get_parent(prof))
get_splines(prof::ProfileLikelihoodSolutionView{N,PLS}) where {N,PLS} = get_splines(get_parent(prof), N)
get_confidence_intervals(prof::ProfileLikelihoodSolutionView{N,PLS}) where {N,PLS} = get_confidence_intervals(get_parent(prof), N)
get_confidence_intervals(prof::ProfileLikelihoodSolutionView{N,PLS}, i) where {N,PLS} = get_confidence_intervals(prof)[i]
get_other_mles(prof::ProfileLikelihoodSolutionView{N,PLS}) where {N,PLS} = get_other_mles(get_parent(prof), N)
get_other_mles(prof::ProfileLikelihoodSolutionView{N,PLS}, i) where {N,PLS} = get_other_mles(prof)[i]
get_syms(prof::ProfileLikelihoodSolutionView{N,PLS}) where {N,PLS} = get_syms(get_parent(prof), N)

eval_spline(prof::ProfileLikelihoodSolutionView, θ) = get_splines(prof)(θ)
(prof::ProfileLikelihoodSolutionView)(θ) = eval_spline(prof, θ)
eval_spline(prof::ProfileLikelihoodSolution, θ, i) = prof[i](θ)
(prof::ProfileLikelihoodSolution)(θ, i) = prof[i](θ)

##########################################################
##
## BIVARIATE
##
##########################################################
"""
    BivariateProfileLikelihoodSolution{I,V,LP,LS,Spl,CT,CF,OM} 

Struct for the normalised bivariate profile log-likelihood. See [`bivariate_profile`](@ref) for a constructor. 

# Arguments 
- `parameter_values::Dict{I, G}`

Maps the tuple `(i, j)` to the grid values used for this parameter pair. The result is a `Tuple`, with the first element 
the grid for the `i`th parameter, and the second element the grid for the `j`th parameter. The grids are given as 
`OffsetVector`s, with the `0`th index the MLE, negative indices to the left of the MLE, and positive indices to the 
right of the MLE.

- `profile_values::Dict{I, V}`

Maps the tuple `(i, j)` to the matrix used for this parameter pair. The result is a `OffsetMatrix`, with the `(k, ℓ)` entry 
the profile at `(parameter_values[(i, j)][1][k], parameter_values[(i, j)][2][k])`, and particularly the `(0, 0)` entry is the 
profile at the MLEs.

- `likelihood_problem::LP`

The original likelihood problem. 

- `likelihood_solution::LS`

The original likelihood solution. 

- `interpolants::Dict{I,Spl}`

Maps the tuple `(i, j)` to the interpolant for that parameter pair's profile. This interpolant also uses linear extrapolation. 

- `confidence_regions::Dict{I,ConfidenceRegion{CT,CF}}`

Maps the tuple `(i, j)` to the confidence region for that parameter pair's confidence region. See also [`ConfidenceRegion`](@ref).

- `other_mles::OM`

Maps the tuple `(i, j)` to an `OffsetMatrix` storing the solutions for the nuisance parameters at the corresponding grid values. 

# Interpolant evaluation 

This struct is callable. We define the method 

    (prof::BivariateProfileLikelihoodSolution)(θ, ψ, i, j)

that evaluates the interpolant through the `(i, j)`th profile at the point `(θ, ψ)`.
"""
Base.@kwdef struct BivariateProfileLikelihoodSolution{I,G,V,LP,LS,Spl,CT,CF,OM}
    parameter_values::Dict{I,G}
    profile_values::Dict{I,V}
    likelihood_problem::LP
    likelihood_solution::LS
    interpolants::Dict{I,Spl}
    confidence_regions::Dict{I,ConfidenceRegion{CT,CF}}
    other_mles::OM
end
get_parameter_values(prof::BivariateProfileLikelihoodSolution) = prof.parameter_values
get_parameter_values(prof::BivariateProfileLikelihoodSolution, i, j) = get_parameter_values(prof)[(i, j)]
get_parameter_values(prof::BivariateProfileLikelihoodSolution, i, j, k) = get_parameter_values(prof, i, j)[k]
function get_parameter_values(prof::BivariateProfileLikelihoodSolution, i, j, sym::Symbol)
    idx = SciMLBase.sym_to_index(sym, prof)
    if idx == i
        return get_parameter_values(prof, i, j, 1)
    elseif idx == j
        return get_parameter_values(prof, i, j, 2)
    end
    throw(BoundsError(get_parameter_values(prof, i, j), sym))
end
get_parameter_values(prof::BivariateProfileLikelihoodSolution, sym1::Symbol, sym2::Symbol, k::Integer) = get_parameter_values(prof, SciMLBase.sym_to_index(sym1, prof), SciMLBase.sym_to_index(sym2, prof), k)
get_parameter_values(prof::BivariateProfileLikelihoodSolution, sym1::Symbol, sym2::Symbol, k::Symbol) = get_parameter_values(prof, SciMLBase.sym_to_index(sym1, prof), SciMLBase.sym_to_index(sym2, prof), k)
get_parameter_values(prof::BivariateProfileLikelihoodSolution, sym1::Symbol, sym2::Symbol) = get_parameter_values(prof, SciMLBase.sym_to_index(sym1, prof), SciMLBase.sym_to_index(sym2, prof))
get_parameter_values(prof::BivariateProfileLikelihoodSolution, i, j, k, ℓ) = get_parameter_values(prof, i, j, k)[ℓ]
get_profile_values(prof::BivariateProfileLikelihoodSolution) = prof.profile_values
get_profile_values(prof::BivariateProfileLikelihoodSolution, i, j) = get_profile_values(prof)[(i, j)]
get_profile_values(prof::BivariateProfileLikelihoodSolution, sym1::Symbol, sym2::Symbol) = get_profile_values(prof, SciMLBase.sym_to_index(sym1, prof), SciMLBase.sym_to_index(sym2, prof))
get_profile_values(prof::BivariateProfileLikelihoodSolution, i, j, k, ℓ) = get_profile_values(prof, i, j)[k, ℓ]
get_likelihood_problem(prof::BivariateProfileLikelihoodSolution) = prof.likelihood_problem
get_likelihood_solution(prof::BivariateProfileLikelihoodSolution) = prof.likelihood_solution
get_interpolants(prof::BivariateProfileLikelihoodSolution) = prof.interpolants
get_interpolants(prof::BivariateProfileLikelihoodSolution, i, j) = get_interpolants(prof)[(i, j)]
get_interpolants(prof::BivariateProfileLikelihoodSolution, sym1::Symbol, sym2::Symbol) = get_interpolants(prof, SciMLBase.sym_to_index(sym1, prof), SciMLBase.sym_to_index(sym2, prof))
get_confidence_regions(prof::BivariateProfileLikelihoodSolution) = prof.confidence_regions
get_confidence_regions(prof::BivariateProfileLikelihoodSolution, i, j) = get_confidence_regions(prof)[(i, j)]
get_confidence_regions(prof::BivariateProfileLikelihoodSolution, sym1::Symbol, sym2::Symbol) = get_confidence_regions(prof, SciMLBase.sym_to_index(sym1, prof), SciMLBase.sym_to_index(sym2, prof))
get_other_mles(prof::BivariateProfileLikelihoodSolution) = prof.other_mles
get_other_mles(prof::BivariateProfileLikelihoodSolution, i, j) = get_other_mles(prof)[(i, j)]
get_other_mles(prof::BivariateProfileLikelihoodSolution, sym1::Symbol, sym2::Symbol) = get_other_mles(prof, SciMLBase.sym_to_index(sym1, prof), SciMLBase.sym_to_index(sym2, prof))
get_other_mles(prof::BivariateProfileLikelihoodSolution, i, j, k, ℓ) = get_other_mles(prof, i, j)[k, ℓ]
get_syms(prof::BivariateProfileLikelihoodSolution) = get_syms(get_likelihood_problem(prof))
get_syms(prof::BivariateProfileLikelihoodSolution, i, j) = (get_syms(prof)[i], get_syms(prof)[j])
profiled_parameters(prof::BivariateProfileLikelihoodSolution) = (collect ∘ keys ∘ get_confidence_regions)(prof)::Vector{NTuple{2,Int}}
number_of_profiled_parameters(prof::BivariateProfileLikelihoodSolution) = length(profiled_parameters(prof))
number_of_layers(prof::BivariateProfileLikelihoodSolution, i, j) = length(get_parameter_values(prof, i, j, 1)) ÷ 2
function get_bounding_box(prof::BivariateProfileLikelihoodSolution, i, j)
    CR = get_confidence_regions(prof, i, j)
    x = get_x(CR)
    y = get_y(CR)
    a, b = extrema(x)
    c, d = extrema(y)
    return a, b, c, d
end

struct BivariateProfileLikelihoodSolutionView{N,M,PLS}
    parent::PLS
end

function Base.getindex(prof::PLS, i::Integer, j::Integer) where {PLS<:BivariateProfileLikelihoodSolution}
    if (i, j) ∈ profiled_parameters(prof)
        return BivariateProfileLikelihoodSolutionView{i,j,PLS}(prof)
    else
        throw(BoundsError(prof, (i, j)))
    end
end
SciMLBase.sym_to_index(sym, prof::BivariateProfileLikelihoodSolution) = SciMLBase.sym_to_index(sym, get_syms(prof))
function Base.getindex(prof::BivariateProfileLikelihoodSolution, sym1, sym2)
    if SciMLBase.issymbollike(sym1)
        i = SciMLBase.sym_to_index(sym1, prof)
    else
        i = sym1
    end
    if SciMLBase.issymbollike(sym2)
        j = SciMLBase.sym_to_index(sym2, prof)
    else
        j = sym2
    end
    if i === nothing
        throw(BoundsError(prof, sym1))
    elseif j === nothing
        throw(BoundsError(prof, sym2))
    else
        prof[i, j]
    end
end
function Base.getindex(prof::BivariateProfileLikelihoodSolution, ij::NTuple{2})
    return prof[ij[1], ij[2]]
end

get_parent(prof::BivariateProfileLikelihoodSolutionView) = prof.parent
get_index(::BivariateProfileLikelihoodSolutionView{N,M,PLS}) where {N,M,PLS} = (N, M)
get_parameter_values(prof::BivariateProfileLikelihoodSolutionView{N,M,PLS}) where {N,M,PLS} = get_parameter_values(get_parent(prof), N, M)
get_parameter_values(prof::BivariateProfileLikelihoodSolutionView{N,M,PLS}, i) where {N,M,PLS} = get_parameter_values(get_parent(prof), N, M, i)
get_parameter_values(prof::BivariateProfileLikelihoodSolutionView{N,M,PLS}, i, j) where {N,M,PLS} = get_parameter_values(get_parent(prof), N, M, i, j)
get_profile_values(prof::BivariateProfileLikelihoodSolutionView{N,M,PLS}) where {N,M,PLS} = get_profile_values(get_parent(prof), N, M)
get_profile_values(prof::BivariateProfileLikelihoodSolutionView{N,M,PLS}, i, j) where {N,M,PLS} = get_profile_values(get_parent(prof), N, M, i, j)
get_likelihood_problem(prof::BivariateProfileLikelihoodSolutionView{N,M,PLS}) where {N,M,PLS} = get_likelihood_problem(get_parent(prof))
get_likelihood_solution(prof::BivariateProfileLikelihoodSolutionView{N,M,PLS}) where {N,M,PLS} = get_likelihood_solution(get_parent(prof))
get_interpolants(prof::BivariateProfileLikelihoodSolutionView{N,M,PLS}) where {N,M,PLS} = get_interpolants(get_parent(prof), N, M)
get_confidence_regions(prof::BivariateProfileLikelihoodSolutionView{N,M,PLS}) where {N,M,PLS} = get_confidence_regions(get_parent(prof), N, M)
get_other_mles(prof::BivariateProfileLikelihoodSolutionView{N,M,PLS}) where {N,M,PLS} = get_other_mles(get_parent(prof), N, M)
get_other_mles(prof::BivariateProfileLikelihoodSolutionView{N,M,PLS}, i, j) where {N,M,PLS} = get_other_mles(get_parent(prof), N, M, i, j)
get_syms(prof::BivariateProfileLikelihoodSolutionView{N,M,PLS}) where {N,M,PLS} = get_syms(get_parent(prof), N, M)
number_of_layers(prof::BivariateProfileLikelihoodSolutionView{N,M,PLS}) where {N,M,PLS} = number_of_layers(get_parent(prof), N, M)
get_bounding_box(prof::BivariateProfileLikelihoodSolutionView{N,M,PLS}) where {N,M,PLS} = get_bounding_box(get_parent(prof), N, M)

eval_interpolant(prof::BivariateProfileLikelihoodSolutionView, θ, φ) = get_interpolants(prof)(θ, φ)
(prof::BivariateProfileLikelihoodSolutionView)(θ, φ) = eval_interpolant(prof, θ, φ)
eval_interpolant(prof::BivariateProfileLikelihoodSolution, θ, φ, i, j) = prof[i, j](θ, φ)
(prof::BivariateProfileLikelihoodSolution)(θ, φ, i, j) = prof[i, j](θ, φ)