function Base.show(io::IO, ::MIME"text/plain", prob::T) where {T<:AbstractLikelihoodProblem}#used MIME to make θ₀ vertical not horizontal
    type_color, no_color = SciMLBase.get_colorizers(io)
    println(io,
        type_color, nameof(typeof(prob)),
        no_color, ". In-place: ",
        type_color, isinplace(get_problem(prob)))
    println(io,
        no_color, "θ₀: ", summary(prob.θ₀))
    sym_param_names = variable_symbols(prob)
    for i in 1:number_of_parameters(prob)
        i < number_of_parameters(prob) ? println(io,
            no_color, "     $(sym_param_names[i]): $(prob[i])") : print(io,
            no_color, "     $(sym_param_names[i]): $(prob[i])")
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
        type_color, get_retcode(sol))
    print(io,
        no_color, "Maximum likelihood: ")
    show(io, mime, get_maximum(sol))
    println(io)
    println(io,
        no_color, "Maximum likelihood estimates: ", summary(get_mle(sol)))
    sym_param_names = variable_symbols(sol)
    for i in 1:number_of_parameters(sol)
        i < number_of_parameters(sol) ? println(io,
            no_color, "     $(sym_param_names[i]): $(sol[i])") : print(io,
            no_color, "     $(sym_param_names[i]): $(sol[i])")
    end
end

function Base.summary(io::IO, sol::T) where {T<:AbstractLikelihoodSolution}
    type_color, no_color = SciMLBase.get_colorizers(io)
    print(io,
        type_color, nameof(typeof(sol)),
        no_color, ". retcode: ",
        type_color, get_retcode(sol),
        no_color)
end

function Base.show(io::IO, ::MIME"text/plain", prof::T) where {T<:ProfileLikelihoodSolution}
    type_color, no_color = SciMLBase.get_colorizers(io)
    println(io,
        type_color, nameof(typeof(prof)),
        no_color, ". MLE retcode: ",
        type_color, get_retcode(get_likelihood_solution(prof)))
    println(io,
        no_color, "Confidence intervals: ")
    CIs = get_confidence_intervals(prof)
    sym_param_names = variable_symbols(prof)
    for i in profiled_parameters(prof)
        i ≠ last(profiled_parameters(prof)) ? println(io,
            no_color, "     $(100get_level(CIs[i]))% CI for $(sym_param_names[i]): ($(get_lower(CIs[i])), $(get_upper(CIs[i])))") : print(io,
            no_color, "     $(100get_level(CIs[i]))% CI for $(sym_param_names[i]): ($(get_lower(CIs[i])), $(get_upper(CIs[i])))")
    end
end

function Base.show(io::IO, ::MIME"text/plain", prof::ProfileLikelihoodSolutionView{N,PLS}) where {N,PLS}
    type_color, no_color = SciMLBase.get_colorizers(io)
    param_name = variable_symbols(prof)[N]
    CIs = get_confidence_intervals(prof)
    println(io,
        type_color, "Profile likelihood",
        no_color, " for parameter",
        type_color, " $param_name",
        no_color, ". MLE retcode: ",
        type_color, get_retcode(get_likelihood_solution(prof)))
    println(io,
        no_color, "MLE: $(get_likelihood_solution(prof)[N])")
    print(io,
        no_color, "$(100get_level(CIs))% CI for $(param_name): ($(get_lower(CIs)), $(get_upper(CIs)))")
end

function Base.show(io::IO, ::MIME"text/plain", prof::T) where {T<:BivariateProfileLikelihoodSolution}
    type_color, no_color = SciMLBase.get_colorizers(io)
    println(io,
        type_color, nameof(typeof(prof)),
        no_color, ". MLE retcode: ",
        type_color, get_retcode(get_likelihood_solution(prof)))
    println(io,
        no_color, "Profile info: ")
    pairs = profiled_parameters(prof)
    sym_param_names = variable_symbols(prof)
    for (i, j) in pairs
        a, b, c, d = get_bounding_box(prof, i, j)
        num_layers = number_of_layers(prof, i, j)
        region_level = 100get_level(get_confidence_regions(prof, i, j))
        (i, j) ≠ last(pairs) ? println(io,
            no_color, "     ($(sym_param_names[i]), $(sym_param_names[j])): $num_layers layers. Bbox for $(region_level)% CR: [$a, $b] × [$c, $d]") : print(io,
            no_color, "     ($(sym_param_names[i]), $(sym_param_names[j])): $num_layers layers. Bbox for $(region_level)% CR: [$a, $b] × [$c, $d]")
    end
end
function Base.show(io::IO, ::MIME"text/plain", prof::BivariateProfileLikelihoodSolutionView{N,M,PLS}) where {N,M,PLS}
    type_color, no_color = SciMLBase.get_colorizers(io)
    param_name = get_syms(prof)
    CR = get_confidence_regions(prof)
    println(io,
        type_color, "Bivariate profile likelihood",
        no_color, " for parameters",
        type_color, " $param_name",
        no_color, ". MLE retcode: ",
        type_color, get_retcode(get_likelihood_solution(prof)))
    println(io,
        no_color, "MLEs: ($(get_likelihood_solution(prof)[N]), $(get_likelihood_solution(prof)[M]))")
    a, b, c, d = get_bounding_box(prof)
    print(io,
        no_color, "$(100get_level(CR))% bounding box for $param_name: [$a, $b] × [$c, $d]")
end

