"""
sym_names(prob::AbstractLikelihoodProblem)

Return the symbolic names used for the likelihood problem `prob`, or creates them 
if none were provided in the construction of `prob`.
"""
@inline function sym_names(prob::AbstractLikelihoodProblem)
nm = prob.prob.f.syms
!isnothing(nm) && return nm
rt = Vector{String}(undef, num_params(prob))
for i in 1:num_params(prob)
    rt[i] = "θ" * subscriptnumber(i)
end
return rt
end

"""
choose_θ₀(::Nothing, lb, ub, num_params)
choose_θ₀(θ₀::T, args...) where {T<:AbstractVector}

Chooses the initial guess `θ₀` based on the provided lower bounds `lb` and upper bounds 
`ub`, with a number of parameters `num_params`, if no `θ₀` is provided. Otherwise, simply 
returns `θ₀`.
"""
function choose_θ₀ end
@inline function choose_θ₀(::Nothing, lb, ub, num_params)
if isnothing(lb) && isnothing(ub)
    θ₀ = zeros(num_params)
elseif !isnothing(lb) && !isnothing(ub)
    if all(isfinite, lb) && all(isfinite, ub)
        θ₀ = similar(lb)
        for i in eachindex(θ₀)
            θ₀[i] = 0.5(lb[i] + ub[i])
        end
    elseif all(isfinite, lb) && !all(isfinite, ub)
        θ₀ = copy(lb)
    elseif !all(isfinite, lb) && all(isfinite, ub)
        θ₀ = copy(ub)
    else
        θ₀ = zeros(num_params)
    end
else
    error("If any of `lb` or `ub` is provided, both must be provided.")
end
return θ₀
end
@inline choose_θ₀(θ₀::T, args...) where {T<:AbstractVector} = θ₀

"""
negate_loglik(loglik)
negate_loglik(loglik, integrator)

Returns `(θ, p) -> -loglik(θ, p)`, or `(θ, p) -> -loglik(θ, p, integrator)` if an 
`integrator` is provided.
"""
function negate_loglik end
@inline negate_loglik(loglik) = @inline (θ, p) -> -loglik(θ, p)
@inline negate_loglik(loglik, integrator) = @inline (θ, p) -> -loglik(θ, p, integrator)

"""
construct_optimisation_function(negloglik, adtype,
    grad, hess, hv,
    cons, cons_j, cons_h,
    hess_prototype, cons_jac_prototype, cons_hess_prototype,
    syms,
    hess_colorvec, cons_jac_colorvec, cons_hess_colorvec,
    expr, cons_expr, f_kwargs)

Constructs an optimisation function with objective `negloglik`. 
See [`OptimizationFunction`](@ref).
"""
@inline function construct_optimisation_function(negloglik, adtype,
grad, hess, hv,
cons, cons_j, cons_h,
hess_prototype, cons_jac_prototype, cons_hess_prototype,
syms,
hess_colorvec, cons_jac_colorvec, cons_hess_colorvec,
expr, cons_expr, f_kwargs)
optfcall = (
    grad=grad, hess=hess, hv=hv,
    cons=cons, cons_j=cons_j, cons_h=cons_h,
    hess_prototype=hess_prototype, cons_jac_prototype=cons_jac_prototype, cons_hess_prototype=cons_hess_prototype,
    syms=syms,
    hess_colorvec=hess_colorvec, cons_jac_colorvec=cons_jac_colorvec, cons_hess_colorvec=cons_hess_colorvec,
    expr=expr, cons_expr=cons_expr
)
optfcall_cleaned = clean_named_tuple(optfcall)
if !isnothing(optfcall_cleaned)
    if isnothing(f_kwargs)
        f = OptimizationFunction(negloglik, adtype; optfcall_cleaned...)
    else
        f = OptimizationFunction(negloglik, adtype; optfcall_cleaned..., f_kwargs...)
    end
else
    if isnothing(f_kwargs)
        f = OptimizationFunction(negloglik, adtype)
    else
        f = OptimizationFunction(negloglik, adtype; f_kwargs...)
    end
end
return f
end

"""
construct_optimisation_problem(f, θ₀, data,
    lb, ub, 
    lcons, ucons, 
    sense, 
    prob_kwargs)

Constructs an optimisation problem. See [`OptimizationProblem`](@ref).
"""
@inline function construct_optimisation_problem(f, θ₀, data,
lb, ub,
lcons, ucons,
sense,
prob_kwargs)
optprobcall = (
    lb=lb, ub=ub,
    lcons=lcons, ucons=ucons,
    sense=sense
)
optprobcall_cleaned = clean_named_tuple(optprobcall)
if !isnothing(optprobcall_cleaned)
    if isnothing(prob_kwargs)
        prob = OptimizationProblem(f, θ₀, data; optprobcall_cleaned...)
    else
        prob = OptimizationProblem(f, θ₀, data; optprobcall_cleaned..., prob_kwargs...)
    end
else
    if isnothing(prob_kwargs)
        prob = OptimizationProblem(f, θ₀, data)
    else
        prob = OptimizationProblem(f, θ₀, data; prob_kwargs...)
    end
end
return prob
end