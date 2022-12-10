using SciMLBase
using InvertedIndices
using FunctionWrappers
using PreallocationTools
using StatsFuns
using SimpleNonlinearSolve
using Interpolations

export LikelihoodProblem
export mle
export GridSearch
export grid_search
export RegularGrid
export IrregularGrid
export profile

######################################################
## Utilities 
######################################################


######################################################
## Problem Updates
######################################################


######################################################
## AbstractLikelihoodProblem
######################################################

######################################################
## LikelihoodProblem
######################################################


######################################################
## LikelihoodSolution 
######################################################

######################################################
## ConfidenceInterval
######################################################

######################################################
## ProfileLikelihoodSolution 
######################################################


######################################################
## MLE
######################################################


######################################################
## GridSearch
######################################################


######################################################
## ProfileLikelihood
######################################################

######################################################
## Plotting
######################################################
const ALPHABET = join('a':'z')

function choose_grid_layout(num_plots, cols, rows)
    if isnothing(cols) && isnothing(rows)
        cols = ceil(Int64, sqrt(num_plots))
        rows = ceil(Int64, num_plots / cols) 
    elseif isnothing(cols)
        cols = ceil(Int64, num_plots / rows)
    elseif isnothing(rows)
        rows = ceil(Int64, num_plots / cols)
    end
    plot_positions = vec([(j, i) for i in 1:cols, j in 1:rows])
    return rows, cols, plot_positions
end

@doc (@doc plot_profiles) function plot_profile!(prof::ProfileLikelihoodSolutionView, fig, ℓ, k, i, j, 
spline, true_vals, mle_val=nothing, shade_ci=true, param_name = L"\theta_{%$ℓ}"; axis_kwargs=nothing)
    lower_ci, upper_ci = get_confidence_intervals(prof)
    θ_vals = get_parameter_values(prof)
    ℓ_vals = get_profile_values(prof)
    conf_level = get_level(get_confidence_intervals(prof))
    threshold = get_chisq_threshold(conf_level)
    formatted_conf_level = parse(Float64, Printf.format(Printf.Format("%.2g"), 100conf_level))
    formatted_lower_ci = parse(Float64, Printf.format(Printf.Format("%.3g"), lower_ci))
    formatted_upper_ci = parse(Float64, Printf.format(Printf.Format("%.3g"), upper_ci)) # This is what @sprintf is doing, but we need to do this so that we can extract the returned value to inteprolate into LaTeXStrings
    if axis_kwargs !== nothing
        ax = CairoMakie.Axis(fig[i, j],
            xlabel=param_name,
            ylabel=L"$\ell_^*($%$(param_name)$) - \ell^*$",
            title=L"(%$(ALPHABET[ℓ])): $%$formatted_conf_level$% CI: $(%$formatted_lower_ci, %$formatted_upper_ci)$",
            titlealign=:left; axis_kwargs...)
    else
        ax = CairoMakie.Axis(fig[i, j],
            xlabel=param_name,
            ylabel=L"$\ell_p^*($%$(param_name)$) - \ell^*$",
            title=L"(%$(ALPHABET[ℓ])): $%$formatted_conf_level$% CI: $(%$formatted_lower_ci, %$formatted_upper_ci)$",
            titlealign=:left)
    end
    CairoMakie.ylims!(ax, threshold - 1, 0.1)
    if !spline
        CairoMakie.lines!(ax, θ_vals, ℓ_vals)
        CairoMakie.hlines!(ax, [threshold], color=:red, linetype=:dashed)
        CI_range = lower_ci .< θ_vals .< upper_ci
        shade_ci && CairoMakie.band!(ax, θ_vals[CI_range], ℓ_vals[CI_range], repeat([threshold], count(CI_range)), color=(:blue, 0.35))
    else
        val_range = extrema(θ_vals)
        Δθ₁ = (val_range[2] - val_range[1]) / max(length(θ_vals), 1000)
        data_vals = val_range[1]:Δθ₁:val_range[2]
        CairoMakie.lines!(ax, data_vals, sol(data_vals, k))
        CairoMakie.hlines!(ax, [threshold], color=:red, linetype=:dashed)
        Δθ₂ = (upper_ci - lower_ci) / max(length(θ_vals), 1000)
        if Δθ₂ ≠ 0.0
            ci_vals = lower_ci:Δθ₂:upper_ci
            shade_ci && CairoMakie.band!(ax, ci_vals, sol(ci_vals, k), repeat([threshold], length(ci_vals)), color=(:blue, 0.35))
        end
    end
    if !isnothing(true_vals)
        CairoMakie.vlines!(ax, [true_vals], color=:black, linetype=:dashed)
    end
    if !isnothing(mle_val)
        CairoMakie.vlines!(ax, [mle_val], color=:red, linetype=:dashed)
    end
    return nothing
end

function plot_profiles(prof::ProfileLikelihoodSolution, vars = profiled_parameters(prof); 
    ncol=nothing, 
    nrow=nothing,
    true_vals=repeat([nothing], number_of_profiled_parameters(prof)), 
    spline=true, 
    show_mles=true, 
    shade_ci=true, 
    fig_kwargs=nothing, 
    axis_kwargs=nothing,
    latex_names = [L"\theta_{%$i}" for i in vars]) 
    num_plots = number_of_profiled_parameters(prof)
    _, _, plot_positions = choose_grid_layout(num_plots, ncol, nrow)
    if fig_kwargs !== nothing
        fig = CairoMakie.Figure(; fig_kwargs...)
    else
        fig = CairoMakie.Figure()
    end
    for (ℓ, k) in pairs(vars)
        i, j = plot_positions[ℓ]
        if axis_kwargs !== nothing
            plot_profile!(prof[k], fig, ℓ, k, i, j, spline, true_vals[k], show_mles ? get_likelihood_solution(prof)[k] : nothing, shade_ci, latex_names[k]; axis_kwargs)
        else
            plot_profile!(prof[k], fig, ℓ, k, i, j, spline, true_vals[k], show_mles ? get_likelihood_solution(prof)[k] : nothing, shade_ci, latex_names[k])
        end
    end
    return fig
end

######################################################
## Display
######################################################
function Base.show(io::IO, ::MIME"text/plain", prob::T) where {T<:AbstractLikelihoodProblem}#used MIME to make θ₀ vertical not horizontal
    type_color, no_color = SciMLBase.get_colorizers(io)
    println(io,
        type_color, nameof(typeof(prob)),
        no_color, ". In-place: ",
        type_color, isinplace(get_problem(prob)))
    println(io,
        no_color, "θ₀: ", summary(prob.θ₀))
    sym_param_names = get_syms(prob)
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
    sym_param_names = get_syms(sol)
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
    sym_param_names = get_syms(prof)
    for i in profiled_parameters(prof)
        i ≠ last(profiled_parameters(prof)) ? println(io,
            no_color, "     $(100get_level(CIs[i]))% CI for $(sym_param_names[i]): ($(get_lower(CIs[i])), $(get_upper(CIs[i])))") : print(io,
            no_color, "     $(100get_level(CIs[i]))% CI for $(sym_param_names[i]): ($(get_lower(CIs[i])), $(get_upper(CIs[i])))")
    end
end
function Base.show(io::IO, ::MIME"text/plain", prof::ProfileLikelihoodSolutionView{N,PLS})  where {N,PLS}
    type_color, no_color = SciMLBase.get_colorizers(io)
    param_name = get_syms(prof)
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