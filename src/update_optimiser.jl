"""
    construct_new_f(prob::OptimizationProblem, i, val, cache::PreallocationTools.DiffCache)
    construct_new_f(prob::OptimizationProblem, i, val, cache::AbstractVector) 

Given `prob`, computes a new objective function which fixes the `i`th variable at the value `val`. 
`cache` is used to store the variables along with this fixed value.
"""
function construct_new_f end
@doc (@doc construct_new_f) @inline function construct_new_f(prob::OptimizationProblem, i, val, cache::PreallocationTools.DiffCache)
    new_f = @inline (θ, p) -> begin
        cache2 = get_tmp(cache, θ)
        cache2[Not(i)] .= θ
        cache2[i] = val
        return prob.f(cache2, p)
    end
    return new_f
end
@doc (@doc construct_new_f) @inline function construct_new_f(prob::OptimizationProblem, i, val, cache::AbstractVector) 
    new_f = @inline (θ, p) -> begin
        cache[Not(i)] .= θ
        cache[i] = val
        return prob.f(cache, p)
    end
    return new_f
end

"""
    update_prob(prob::OptimizationProblem, i::Int) 
    update_prob(prob::OptimizationProblem, u0::AbstractVector) 
    update_prob(prob::OptimizationProblem, i, val, cache) 
    update_prob(prob::OptimizationProblem, i, val, cache, u0)

Given the [`OptimizationProblem`](@ref) `prob`, updates it based on the methods above.

1. Removes the `i`th entry of the lower and upper bounds. 
2. Updates the problem with a new initial guess `u0`.
3. Replaces the objective function with a new one that fixes the `i`th variable at the value `val`. `cache` is used to store the variables along with this fixed value. 
4. Performs method 3 and method 2.

These updates is not done in-place.
"""
function update_prob end 
@inline function update_prob(prob::OptimizationProblem, i::Int) 
    return remake(prob; lb=prob.lb[Not(i)], ub=prob.ub[Not(i)])
end
@inline function update_prob(prob::OptimizationProblem, u0::AbstractVector) 
    return remake(prob; u0=u0)
end
@inline function update_prob(prob::OptimizationProblem{iip,F,θType,P,B,LC,UC,S,K}, i, val, cache) where {iip,F,θType,P,B,LC,UC,S,K}
    new_f = construct_new_f(prob, i, val, cache)
    f = OptimizationFunction{
        iip,
        F.parameters[2],
        typeof(new_f),
        F.parameters[4:end]...
    }(new_f,
        prob.f.adtype, prob.f.grad,
        prob.f.hess, prob.f.hv,
        prob.f.cons, prob.f.cons_j, prob.f.cons_h,
        prob.f.hess_prototype, prob.f.cons_jac_prototype, prob.f.cons_hess_prototype,
        prob.f.syms,
        prob.f.hess_colorvec, prob.f.cons_jac_colorvec, prob.f.cons_hess_colorvec,
        prob.f.expr, prob.f.cons_expr)
    return remake(prob; f=f)
end
@inline function update_prob(prob::OptimizationProblem, i, val, cache, u0)
    prob = update_prob(prob, i, val, cache)
    prob = update_prob(prob, u0)
    return prob
end