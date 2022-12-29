# https://stackoverflow.com/a/41847530
number_type(x) = number_type(typeof(x))
number_type(::Type{T}) where {T<:AbstractArray} = number_type(eltype(T))
number_type(::Type{NTuple{N,F}}) where {N,F} = number_type(F)
number_type(::Type{T}) where {T} = T

function get_default_extremum(::Type{T}, minimise::Type{M}) where {M,T}
    f_opt = M == Val{false} ? typemin(T) : typemax(T)
    return f_opt
end

function update_extremum!(new_x, new_f, old_x, old_f; minimise::M) where {M}
    if M == Val{false}
        if new_f > old_f
            new_x .= old_x
            return new_f
        end
        return old_f
    elseif M == Val{true}
        if new_f < old_f
            new_x .= old_x
            return new_f
        end
        return old_f
    end
    return nothing
end

@inline function gaussian_loglikelihood(x::AbstractArray{T}, μ::AbstractArray{T}, σ, n) where {T}
    ℓ = -0.5n * log(2π * σ^2)
    s = zero(T)
    @simd for i ∈ eachindex(x, μ)
        s += (x[i] - μ[i])^2
    end
    ℓ = ℓ - 0.5 / σ^2 * s
    return ℓ
end

get_chisq_threshold(level, d=1) = -0.5chisqinvcdf(d, level)

function subscriptnumber(i::Int)#https://stackoverflow.com/a/64758370
    if i < 0
        c = [Char(0x208B)]
    else
        c = []
    end
    for d in reverse(digits(abs(i)))
        push!(c, Char(0x2080 + d))
    end
    return join(c)
end

function linear_extrapolation(x, x₀, y₀, x₁, y₁)
    y = (y₀ * (x₁ - x) + y₁ * (x - x₀)) / (x₁ - x₀)
    return y
end
function linear_extrapolation!(y, x, x₀, y₀, x₁, y₁)
    for (i, y0, y1) in zip(eachindex(y), y₀, y₁)
        y[i] = linear_extrapolation(x, x₀, y0, x₁, y1)
    end
    return nothing
end

function convert_symbol_tuples(n::NTuple{M,NTuple{2,S where S<:Union{Integer,Symbol}}}, prob) where {M}
    integer_n = ntuple(i -> (SciMLBase.sym_to_index(n[i][1], prob), SciMLBase.sym_to_index(n[i][2], prob)), M)
    return integer_n
end
function convert_symbol_tuples(n::NTuple{2,S where S<:Union{Integer,Symbol}}, prob)
    return convert_symbol_tuples((n,), prob)[1]
end

_Val(V::Val{B}) where {B} = V
_Val(V) = Val(V)
take_val(::Val{B}) where {B} = B
take_val(V) = V

struct VecBSpline{N,Spl}
    splines::Vector{Spl}
    VecBSpline(spl::Vector{S}) where {S} = new{length(spl),S}(spl)
end
(spl::VecBSpline{N,S})(x) where {N,S} = [spl.splines[i](x) for i in 1:N]