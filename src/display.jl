function Base.show(io::IO, ::MIME"text/plain", prob::T) where {T<:AbstractLikelihoodProblem}#used MIME to make θ₀ vertical not horizontal
    type_color, no_color = SciMLBase.get_colorizers(io)
    println(io,
        type_color, nameof(typeof(prob)),
        no_color, ". In-place: ",
        type_color, isinplace(prob.prob))
    println(io,
        no_color, "θ₀: ", summary(prob.θ₀))
    #show(io, mime, prob.θ₀)
    sym_param_names = sym_names(prob)
    for i in 1:num_params(prob)
        i < num_params(prob) ? println(io,
            no_color, "     $(sym_param_names[i]): $(prob.θ₀[i])") : print(io,
            no_color, "     $(sym_param_names[i]): $(prob.θ₀[i])")
    end
end
function Base.summary(io::IO, prob::T) where {T<:AbstractLikelihoodProblem}
    type_color, no_color = SciMLBase.get_colorizers(io)
    print(io,
        type_color, nameof(typeof(prob)),
        no_color, ". In-place: ",
        type_color, isinplace(prob.prob),
        no_color)
end
function Base.show(io::IO, mime::MIME"text/plain", sol::T) where {T<:AbstractLikelihoodSolution}
    type_color, no_color = SciMLBase.get_colorizers(io)
    println(io,
        type_color, nameof(typeof(sol)),
        no_color, ". retcode: ",
        type_color, retcode(sol))
    println(io,
        no_color, "Algorithm: ",
        algorithm_name(sol))
    print(io,
        no_color, "Maximum likelihood: ")
    show(io, mime, sol.maximum)
    println(io)
    println(io,
        no_color, "Maximum likelihood estimates: ", summary(sol.θ))
    sym_param_names = sym_names(sol)
    for i in 1:num_params(sol)
        i < num_params(sol) ? println(io,
            no_color, "     $(sym_param_names[i]): $(sol.θ[i])") : print(io,
            no_color, "     $(sym_param_names[i]): $(sol.θ[i])")
    end
end
function Base.summary(io::IO, sol::T) where {T<:AbstractLikelihoodSolution}
    type_color, no_color = SciMLBase.get_colorizers(io)
    print(io,
        type_color, nameof(typeof(sol)),
        no_color, ". retcode: ",
        type_color, retcode(sol),
        no_color)
end
function Base.show(io::IO, ::MIME"text/plain", sol::T) where {T<:ProfileLikelihoodSolution}
    type_color, no_color = SciMLBase.get_colorizers(io)
    println(io,
        type_color, nameof(typeof(sol)),
        no_color, ". MLE retcode: ",
        type_color, retcode(sol))
    println(io,
        no_color, "Algorithm: ",
        algorithm_name(sol))
    println(io,
        no_color, "Confidence intervals: ")
    CIs = confidence_intervals(sol)
    sym_param_names = sym_names(sol)
    for i in 1:num_params(sol)
        i < num_params(sol) ? println(io,
            no_color, "     $(100level(CIs[i]))% CI for $(sym_param_names[i]): ($(lower(CIs[i])), $(upper(CIs[i])))") : print(io,
            no_color, "     $(100level(CIs[i]))% CI for $(sym_param_names[i]): ($(lower(CIs[i])), $(upper(CIs[i])))")
    end
end
function Base.show(io::IO, ::MIME"text/plain", sol::T) where {I,PLS,V,LP,LS,Spl,CT,CF,T<:ProfileLikelihoodSolutionView{I,PLS,V,LP,LS,Spl,CT,CF}}
    type_color, no_color = SciMLBase.get_colorizers(io)
    prof = sol.parent
    param_name = sym_names(prof)[I]
    CIs = confidence_intervals(prof)[I]
    println(io,
        type_color, "Profile likelihood",
        no_color, " for parameter",
        type_color, " $param_name",
        no_color, ". MLE retcode: ",
        type_color, retcode(prof))
    println(io,
        no_color, "MLE: $(mle(prof)[I])")
    print(io,
        no_color, "$(100level(CIs))% CI for $(param_name): ($(lower(CIs)), $(upper(CIs)))")
end