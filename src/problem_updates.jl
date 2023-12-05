update_initial_estimate(prob::OptimizationProblem, θ) = remake(prob; u0=θ)
update_initial_estimate(prob::OptimizationProblem, sol::SciMLBase.OptimizationSolution) = update_initial_estimate(prob, sol.u)

# replace obj with a new obj
@inline function replace_objective_function(f::OF, new_f::FF) where {iip,AD,F,G,H,HV,C,CJ,CH,HP,CJP,CHP,S,S2,O,EX,CEX,SYS,LH,LHP,HCV,CJCV,CHCV,LHCV,OF<:OptimizationFunction{iip,AD,F,G,H,HV,C,CJ,CH,HP,CJP,CHP,S,S2,O,EX,CEX,SYS,LH,LHP,HCV,CJCV,CHCV,LHCV},FF}
    # scimlbase needs to add a constructorof method for OptimizationFunction before we can just do @set with accessors.jl. 
    # g = @set f.f = new_f
    # return g
    return OptimizationFunction{iip,AD,FF,G,H,HV,C,CJ,CH,HP,CJP,CHP,S,S2,O,EX,CEX,SYS,LH,LHP,HCV,CJCV,CHCV,LHCV}(
        new_f,
        f.adtype,
        f.grad,
        f.hess,
        f.hv,
        f.cons,
        f.cons_j,
        f.cons_h,
        f.hess_prototype,
        f.cons_jac_prototype,
        f.cons_hess_prototype,
        f.syms,
        f.paramsyms,
        f.observed,
        f.expr,
        f.cons_expr,
        f.sys,
        f.lag_h,
        f.lag_hess_prototype,
        f.hess_colorvec,
        f.cons_jac_colorvec,
        f.cons_hess_colorvec,
        f.lag_hess_colorvec,
    )
end

@inline function replace_objective_function(prob::F, obj::FF) where {F<:OptimizationProblem,FF}
    g = replace_objective_function(prob.f, obj)
    return remake(prob; f=g)
end

# fix the objective function's nth parameter at θₙ
function construct_fixed_optimisation_function(prob::OptimizationProblem, n::Integer, θₙ, cache)
    original_f = prob.f
    new_f = @inline (θ, p) -> begin
        cache2 = get_tmp(cache, θ)
        for i in eachindex(cache2)
            if i < n
                cache2[i] = θ[i]
            elseif i > n
                cache2[i] = θ[i-1]
            else
                cache2[n] = θₙ
            end
        end
        return original_f(cache2, p)
    end
    return replace_objective_function(prob, new_f)
end

# fix the objective function's (n1, n2) parameters at (θn1, θn2)
function construct_fixed_optimisation_function(prob::OptimizationProblem, n::NTuple{2,Integer}, θₙ, cache)
    n₁, n₂ = n
    θₙ₁, θₙ₂ = θₙ
    if n₁ > n₂
        n₁, n₂ = n₂, n₁
        θₙ₁, θₙ₂ = θₙ₂, θₙ₁
    end
    original_f = prob.f
    new_f = @inline (θ, p) -> begin
        cache2 = get_tmp(cache, θ)
        @inbounds for i in eachindex(cache2)
            if i < n₁
                cache2[i] = θ[i]
            elseif i == n₁
                cache2[i] = θₙ₁
            elseif n₁ < i < n₂
                cache2[i] = θ[i-1]
            elseif i == n₂
                cache2[i] = θₙ₂
            elseif i > n₂
                cache2[i] = θ[i-2]
            end
        end
        return original_f(cache2, p)
    end
    return replace_objective_function(prob, new_f)
end

# remove lower bounds, upper bounds, and also remove the nth value from the initial estimate
function exclude_parameter(prob::OptimizationProblem, n::Integer)
    !has_bounds(prob) && return update_initial_estimate(prob, prob.u0[Not(n)])
    lb₋ₙ = get_lower_bounds(prob, Not(n))
    ub₋ₙ = get_upper_bounds(prob, Not(n))
    new_prob = remake(prob; lb=lb₋ₙ, ub=ub₋ₙ, u0=prob.u0[Not(n)])
    return new_prob
end

# remove lower bounds, upper bounds, and also remove the (n1,n2) value from the initial estimate
function exclude_parameter(prob::OptimizationProblem, n::NTuple{2,Integer})
    !has_bounds(prob) && return update_initial_estimate(prob, prob.u0[Not(n[1], n[2])])
    lb₋ₙ = get_lower_bounds(prob, Not(n[1], n[2]))
    ub₋ₙ = get_upper_bounds(prob, Not(n[1], n[2]))
    new_prob = remake(prob; lb=lb₋ₙ, ub=ub₋ₙ, u0=prob.u0[Not(n[1], n[2])])
    return new_prob
end

# replace obj by obj - shift
function shift_objective_function(prob::OptimizationProblem, shift)
    original_f = prob.f
    new_f = @inline (θ, p) -> original_f(θ, p) - shift
    return replace_objective_function(prob, new_f)
end