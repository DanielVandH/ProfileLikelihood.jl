update_initial_estimate(prob::OptimizationProblem, θ) = remake(prob; u0=θ)
update_initial_estimate(prob::OptimizationProblem, sol::SciMLBase.OptimizationSolution) = update_initial_estimate(prob, sol.u)

# replace obj with a new obj
function replace_objective_function(prob::OptimizationProblem{iip,FF,uType,P,B,LC,UC,Sns,K}, obj::F) where {F,iip,AD,G,H,HV,C,CJ,CH,LH,HP,CJP,CHP,LHP,S,S2,O,HCV,CJCV,CHCV,LHCV,EX,CEX,SYS,FF2,FF<:OptimizationFunction{iip,AD,FF2,G,H,HV,C,CJ,CH,LH,HP,CJP,CHP,LHP,S,S2,O,HCV,CJCV,CHCV,LHCV,EX,CEX,SYS},uType,P,B,LC,UC,Sns,K}
    f = OptimizationFunction{iip,AD,F,G,H,HV,C,CJ,CH,LH,HP,CJP,CHP,LHP,S,S2,O,HCV,CJCV,CHCV,LHCV,EX,CEX,SYS
    }(obj,
        prob.f.adtype, prob.f.grad,
        prob.f.hess, prob.f.hv,
        prob.f.cons, prob.f.cons_j, prob.f.cons_h, prob.f.lag_h,
        prob.f.hess_prototype, prob.f.cons_jac_prototype, prob.f.cons_hess_prototype, prob.f.lag_hess_prototype,
        prob.f.syms, prob.f.paramsyms, prob.f.observed,
        prob.f.hess_colorvec, prob.f.cons_jac_colorvec, prob.f.cons_hess_colorvec, prob.f.lag_hess_colorvec,
        prob.f.expr, prob.f.cons_expr, prob.f.sys)
    return remake(prob; f=f)
end

# fix the objective function's nth parameter at θₙ
function construct_fixed_optimisation_function(prob::OptimizationProblem, n, θₙ, cache)
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

# remove lower bounds, upper bounds, and also remove the nth value from the initial estimate
function exclude_parameter(prob::OptimizationProblem, n::Integer)
    !has_bounds(prob) && return update_initial_estimate(prob, prob.u0[Not(n)])
    lb₋ₙ = get_lower_bounds(prob, Not(n))
    ub₋ₙ = get_upper_bounds(prob, Not(n))
    new_prob = remake(prob; lb=lb₋ₙ, ub=ub₋ₙ, u0=prob.u0[Not(n)])
    return new_prob
end

# replace obj by obj - shift
function shift_objective_function(prob::OptimizationProblem, shift)
    original_f = prob.f
    new_f = @inline (θ, p) -> original_f(θ, p) - shift
    return replace_objective_function(prob, new_f)
end