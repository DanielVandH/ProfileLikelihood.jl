"""
    transform_result(sol::LikelihoodSolution,  F::Vector{Fnc} where Fnc <: Function) 
    transform_result(sol::LikelihoodSolution,  F::Fnc) where {Fnc <: Function}
    transform_result(CI::ConfidenceInterval, F::Fnc) where {Fnc <: Function}
    transform_result(sol::ProfileLikelihoodSolution, F::Vector{Fnc} where {Fnc <: Function})
    transform_result(sol::ProfileLikelihoodSolution, F::Fnc) where {Fnc <: Function}

Methods for transforming results by the given function(s). When transforming confidence intervals, it is 
assumed that `F` is injective.
"""
function transform_result end 
function transform_result(sol::LikelihoodSolution,  F::Vector{Fnc} where Fnc <: Function) 
    length(F) == num_params(sol) || throw(ArgumentError("The vector of transforms F must satisfy length(F) = num_params(sol)."))
    new_θ = similar(mle(sol))
    for (i, f) in pairs(F)
        new_θ[i] = f(mle(sol)[i])
    end
    return remake(sol, θ = new_θ)
end
function transform_result(sol::LikelihoodSolution,  F::Fnc) where {Fnc <: Function}
    LikelihoodSolution(F.(mle(sol)), sol.prob, sol.alg, sol.maximum, sol.retcode, sol.original)
end
function transform_result(CI::ConfidenceInterval, F::Fnc) where {Fnc <: Function}
    ℓ = F(lower(CI))
    u = F(upper(CI))
    lvl = level(CI)
    ℓ, u = extrema([ℓ, u])
    ConfidenceInterval(ℓ, u, lvl)
end
function transform_result(sol::ProfileLikelihoodSolution, F::Vector{Fnc} where {Fnc <: Function})
    ## Transform the parameter values first 
    new_θ = typeof(sol.θ)([])
    for i in keys(sol.θ)
        new_θ[i] = F[i].(sol.θ[i])
    end
    ## Transform the mle 
    new_mle = transform_result(sol.mle, F)
    ## Remake the splines 
    new_spline = typeof(sol.spline)([])
    for i in keys(sol.spline)
        new_spline[i] = Spline1D(new_θ[i], sol.profile[i])
    end
    ## Transform the confidence intervals 
    new_CI = typeof(sol.confidence_intervals)([])
    for i in keys(sol.confidence_intervals)
        new_CI[i] = transform_result(sol.confidence_intervals[i], F[i])
    end
    ## Done 
    return ProfileLikelihoodSolution(new_θ, sol.profile, sol.prob, new_mle, new_spline, new_CI)
end
function transform_result(sol::ProfileLikelihoodSolution, F::Fnc) where {Fnc <: Function}
    return transform_result(sol, repeat([F], num_params(sol)))
end