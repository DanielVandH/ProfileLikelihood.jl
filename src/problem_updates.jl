update_initial_estimate(prob::OptimizationProblem, θ) = remake(prob; u0=θ)
update_initial_estimate(prob::OptimizationProblem, sol::SciMLBase.OptimizationSolution) = update_initial_estimate(prob, sol.u)

@generated function _to_namedtuple(obj)
    A = (Expr(:(=), n, :(obj.$n)) for n in setdiff(fieldnames(obj), (:f, :adtype, :grad, :hess)))
    Expr(:tuple, A...)
end

# replace obj with a new obj
@inline function replace_objective_function(f::OF, new_f::FF, new_grad!::GG, new_hess!::HH) where {OF, FF, GG, HH}
    return OptimizationFunction(
        new_f,
        f.adtype;
        grad=new_grad!,
        hess=new_hess!,
        _to_namedtuple(f)...
    )
end

@inline function replace_objective_function(prob::F, obj::FF, grad, hess) where {F<:OptimizationProblem, FF}
    g = replace_objective_function(prob.f, obj, grad, hess)
    return remake(prob; f=g)
end

# fix the objective function's nth parameter at θₙ
function construct_fixed_optimisation_function(prob::OptimizationProblem, n::Integer, θₙ, cache)
    original_f = prob.f
    original_grad! = prob.f.grad
    original_hess! = prob.f.hess

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

    # In case a custom gradient is provided the new gradient function the full gradient
    # is computed, but the value for parameter n is dropped
    if isnothing(original_grad!)
        # Default for an OptimizationProblem
        new_grad! = nothing 
    else
        new_grad! = (g, θ, p) -> begin
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
            _g = similar(cache2)           
            original_grad!(_g, cache2, p)
            for i in eachindex(cache2)
                if i < n
                    g[i] = _g[i]
                elseif i > n
                    g[i-1] = _g[i]
                end
            end
            return nothing
        end
    end

    # Likewise, if a custom Hessian is provided the full Hessian is computed, 
    # but the value for parameter n is dropped 
    if isnothing(original_hess!)
        # Default for an OptimizationProblem
        new_hess! = nothing 
    else
        new_hess! = (H, θ, p) -> begin
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
            _H = Matrix{eltype(cache2)}(undef, length(cache2), length(cache2))
            original_hess!(_H, cache2, p)
            for i in eachindex(cache2)
                for j in eachindex(cache2)
                    if i == n || j == n
                        continue
                    else i < n && j < n
                        ix = i < n ? i : i - 1
                        jx = j < n ? j : j - 1
                        H[ix, jx] = _H[i, j]
                    end
                end
            end
            return nothing
        end
    end

    return replace_objective_function(prob, new_f, new_grad!, new_hess!)
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
    original_grad! = prob.f.grad
    original_hess! = prob.f.hess

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

    # In case a custom gradient is provided the new gradient function the full gradient
    # but the value for parameter n₁ and n₂ are dropped
    if isnothing(original_grad!)
        # Default for an OptimizationProblem
        new_grad! = nothing 
    else
        new_grad! = (g, θ, p) -> begin
            cache2 = get_tmp(cache, θ)
            for i in eachindex(cache2)
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
            _g = similar(cache2)           
            original_grad!(_g, cache2, p)
            for i in eachindex(cache2)
                if i == n₁ || i == n₂
                    continue
                else
                    ix = i < n₁ ? i : (i < n₂ ? i - 1 : i - 2)
                    g[ix] = _g[i]
                end
            end
            return nothing
        end
    end

    # Likewise, if a custom Hessian is provided the full Hessian is computed, 
    # but the value for parameter n₁ and n₂ are dropped
    if isnothing(original_hess!)
        # Default for an OptimizationProblem
        new_hess! = nothing 
    else
        new_hess! = (H, θ, p) -> begin
            cache2 = get_tmp(cache, θ)
            for i in eachindex(cache2)
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
            _H = Matrix{eltype(cache2)}(undef, length(cache2), length(cache2))
            original_hess!(_H, cache2, p)
            for i in eachindex(cache2)
                for j in eachindex(cache2)
                    if (i == n₁ || j == n₁) || (i == n₂ || j == n₂)
                        continue
                    else
                        ix = i < n₁ ? i : (i < n₂ ? i - 1 : i - 2)
                        jx = j < n₁ ? j : (j < n₂ ? j - 1 : j - 2)
                        H[ix, jx] = _H[i, j]
                    end
                end
            end
            return nothing
        end
    end

    return replace_objective_function(prob, new_f, new_grad!, new_hess!)
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
    original_f = prob.f.f
    new_f = @inline (θ, p) -> original_f(θ, p) - shift
    return replace_objective_function(prob, new_f, prob.f.grad, prob.f.hess)
end