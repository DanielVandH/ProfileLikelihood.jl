module ProfileLikelihoodMakieExt

using ProfileLikelihood
@static if isdefined(Base, :get_extension)
    using Makie
else
    using ..Makie
end
using Printf

const ALPHABET = join('a':'z')

function ProfileLikelihood.plot_profiles(prof::ProfileLikelihood.ProfileLikelihoodSolution, vars=ProfileLikelihood.profiled_parameters(prof);
    ncol=nothing,
    nrow=nothing,
    true_vals=Dict(vars .=> nothing),
    spline=true,
    show_mles=true,
    shade_ci=true,
    fig_kwargs=nothing,
    axis_kwargs=nothing,
    show_points=false,
    markersize=9,
    latex_names=Dict(vars .=> [L"\theta_{%$i}" for i in ProfileLikelihood.variable_index.((prof,), vars)]))
    num_plots = vars isa Symbol ? 1 : length(vars)
    _, _, plot_positions = ProfileLikelihood.choose_grid_layout(num_plots, ncol, nrow)
    if fig_kwargs !== nothing
        fig = Makie.Figure(; fig_kwargs...)
    else
        fig = Makie.Figure()
    end
    itr = vars isa Symbol ? [(1, vars)] : pairs(vars)
    for (ℓ, k) in itr
        i, j = plot_positions[ℓ]
        if axis_kwargs !== nothing
            ProfileLikelihood.plot_profiles!(prof[k], fig, ℓ, k, i, j, spline, true_vals[k], show_mles ? ProfileLikelihood.get_likelihood_solution(prof)[k] : nothing, shade_ci, latex_names[k], show_points, markersize; axis_kwargs)
        else
            ProfileLikelihood.plot_profiles!(prof[k], fig, ℓ, k, i, j, spline, true_vals[k], show_mles ? ProfileLikelihood.get_likelihood_solution(prof)[k] : nothing, shade_ci, latex_names[k], show_points, markersize)
        end
    end
    return fig
end

function ProfileLikelihood.plot_profiles(prof::ProfileLikelihood.BivariateProfileLikelihoodSolution, vars=ProfileLikelihood.profiled_parameters(prof);
    ncol=nothing,
    nrow=nothing,
    true_vals=Dict(1:ProfileLikelihood.number_of_parameters(ProfileLikelihood.get_likelihood_problem(prof)) .=> nothing),
    show_mles=true,
    fig_kwargs=nothing,
    axis_kwargs=nothing,
    interpolation=false,
    smooth_confidence_boundary=false,
    close_contour=true,
    latex_names=Dict(1:ProfileLikelihood.number_of_parameters(ProfileLikelihood.get_likelihood_problem(prof)) .=> ProfileLikelihood.variable_symbols(prof)),
    xlim_tuples=nothing,
    ylim_tuples=nothing)
    vars = ProfileLikelihood.convert_symbol_tuples(vars, prof)
    num_plots = (vars isa NTuple{2,Symbol} || vars isa NTuple{2,Int}) ? 1 : length(vars)
    nr, nc, plot_positions = ProfileLikelihood.choose_grid_layout(num_plots, ncol, nrow)
    if fig_kwargs !== nothing
        fig = Makie.Figure(; fig_kwargs...)
    else
        fig = Makie.Figure()
    end
    itr = (vars isa NTuple{2,Symbol} || vars isa NTuple{2,Int}) ? [(1, Tuple(vars))] : pairs(vars)
    for (ℓ, (k, r)) in itr
        i, j = plot_positions[ℓ]
        if axis_kwargs !== nothing
            ProfileLikelihood.plot_profiles!(prof[k, r], fig, ℓ, (k, r), i, j, (true_vals[k], true_vals[r]), interpolation, smooth_confidence_boundary,
                show_mles ? (ProfileLikelihood.get_likelihood_solution(prof)[k], ProfileLikelihood.get_likelihood_solution(prof)[r]) : nothing, (latex_names[k], latex_names[r]),
                close_contour, isnothing(xlim_tuples) ? nothing : xlim_tuples[ℓ], isnothing(ylim_tuples) ? nothing : ylim_tuples[ℓ];
                axis_kwargs)
        else
            ProfileLikelihood.plot_profiles!(prof[k, r], fig, ℓ, (k, r), i, j, (true_vals[k], true_vals[r]), interpolation, smooth_confidence_boundary,
                show_mles ? (ProfileLikelihood.get_likelihood_solution(prof)[k], ProfileLikelihood.get_likelihood_solution(prof)[r]) : nothing, (latex_names[k], latex_names[r]),
                close_contour, isnothing(xlim_tuples) ? nothing : xlim_tuples[ℓ], isnothing(ylim_tuples) ? nothing : ylim_tuples[ℓ])
        end
    end
    Makie.Colorbar(fig[1:nr, nc+1], colorrange=(-16, 0), colormap=:viridis, label=L"Normalised profile $ $", ticks=(-16:4:0))
    return fig
end

function ProfileLikelihood.choose_grid_layout(num_plots, cols, rows)
    if isnothing(cols) && isnothing(rows)
        cols = ceil(Int, sqrt(num_plots))
        rows = ceil(Int, num_plots / cols)
    elseif isnothing(cols)
        cols = ceil(Int, num_plots / rows)
    elseif isnothing(rows)
        rows = ceil(Int, num_plots / cols)
    end
    plot_positions = vec([(j, i) for i in 1:cols, j in 1:rows])
    return rows, cols, plot_positions
end

function ProfileLikelihood.plot_profiles!(prof::ProfileLikelihood.ProfileLikelihoodSolutionView, fig, ℓ, k, i, j,
    spline, true_vals, mle_val=nothing, shade_ci=true, param_name=L"\theta_{%$ℓ}",
    show_points=false, markersize=9; axis_kwargs=nothing)
    lower_ci, upper_ci = ProfileLikelihood.get_confidence_intervals(prof)
    θ_vals = ProfileLikelihood.get_parameter_values(prof)
    ℓ_vals = ProfileLikelihood.get_profile_values(prof)
    conf_level = ProfileLikelihood.get_level(get_confidence_intervals(prof))
    threshold = ProfileLikelihood.get_chisq_threshold(conf_level)
    formatted_conf_level = parse(Float64, Printf.format(Printf.Format("%.2g"), 100conf_level))
    formatted_lower_ci = parse(Float64, Printf.format(Printf.Format("%.3g"), lower_ci))
    formatted_upper_ci = parse(Float64, Printf.format(Printf.Format("%.3g"), upper_ci)) # This is what @sprintf is doing, but we need to do this so that we can extract the returned value to inteprolate into LaTeXStrings
    if axis_kwargs !== nothing
        ax = Makie.Axis(fig[i, j],
            xlabel=param_name,
            ylabel=L"$\ell_p^*($%$(param_name)$) - \ell^*$",
            title=L"(%$(ALPHABET[ℓ])): $%$formatted_conf_level$% CI: $(%$formatted_lower_ci, %$formatted_upper_ci)$",
            titlealign=:left; axis_kwargs...)
    else
        ax = Makie.Axis(fig[i, j],
            xlabel=param_name,
            ylabel=L"$\ell_p^*($%$(param_name)$) - \ell^*$",
            title=L"(%$(ALPHABET[ℓ])): $%$formatted_conf_level$% CI: $(%$formatted_lower_ci, %$formatted_upper_ci)$",
            titlealign=:left)
    end
    Makie.ylims!(ax, threshold - 1, 0.1)
    if !spline
        Makie.lines!(ax, θ_vals, ℓ_vals)
        Makie.hlines!(ax, [threshold], color=:red, linestyle=:dash)
        CI_range = lower_ci .< θ_vals .< upper_ci
        shade_ci && Makie.band!(ax, θ_vals[CI_range], ℓ_vals[CI_range], repeat([threshold], count(CI_range)), color=(:blue, 0.35))
    else
        val_range = extrema(θ_vals)
        Δθ₁ = (val_range[2] - val_range[1]) / max(length(θ_vals), 1000)
        data_vals = val_range[1]:Δθ₁:val_range[2]
        Makie.lines!(ax, data_vals, prof(data_vals))
        Makie.hlines!(ax, [threshold], color=:red, linestyle=:dash)
        Δθ₂ = (upper_ci - lower_ci) / max(length(θ_vals), 1000)
        if Δθ₂ ≠ 0.0
            ci_vals = lower_ci:Δθ₂:upper_ci
            shade_ci && Makie.band!(ax, ci_vals, prof(ci_vals), repeat([threshold], length(ci_vals)), color=(:blue, 0.35))
        end
    end
    if !isnothing(true_vals)
        Makie.vlines!(ax, [true_vals], color=:black, linestyle=:dash)
    end
    if !isnothing(mle_val)
        Makie.vlines!(ax, [mle_val], color=:red, linestyle=:dash)
    end
    if show_points
        Makie.scatter!(ax, θ_vals, ℓ_vals, color=:black, markersize=markersize)
    end
    return nothing
end
function ProfileLikelihood.plot_profiles!(prof::ProfileLikelihood.BivariateProfileLikelihoodSolutionView, fig, ℓ, (k, r), i, j,
    true_vals, interpolation=false, smooth_confidence_boundary=false, mle_val=nothing, (name_1, name_2)=(L"\psi", L"\varphi"), close_contour=true,
    xlim_tuple=nothing, ylim_tuple=nothing; axis_kwargs=nothing)
    if !interpolation
        grid_1 = ProfileLikelihood.get_parameter_values(prof, 1).parent
        grid_2 = ProfileLikelihood.get_parameter_values(prof, 2).parent
        prof_vals = ProfileLikelihood.get_profile_values(prof).parent |> deepcopy
    else
        if isnothing(xlim_tuple) && isnothing(ylim_tuple)
            grid_1 = ProfileLikelihood.get_parameter_values(prof, 1).parent
            grid_2 = ProfileLikelihood.get_parameter_values(prof, 2).parent
            grid_1 = LinRange(extrema(grid_1)..., 12length(grid_1))
            grid_2 = LinRange(extrema(grid_2)..., 12length(grid_2))
            prof_vals = [prof(x, y) for x in grid_1, y in grid_2]
        else
            grid_1 = ProfileLikelihood.get_parameter_values(prof, 1).parent
            grid_2 = ProfileLikelihood.get_parameter_values(prof, 2).parent
            grid_1 = LinRange(xlim_tuple[1], xlim_tuple[2], 12length(grid_1))
            grid_2 = LinRange(ylim_tuple[1], ylim_tuple[2], 12length(grid_2))
            prof_vals = [prof(x, y) for x in grid_1, y in grid_2]
        end
    end
    if axis_kwargs !== nothing
        ax = Makie.Axis(fig[i, j],
            xlabel=L"%$(name_1)",
            ylabel=L"%$(name_2)",
            title=name_1 isa Symbol ? L"(%$(ALPHABET[ℓ])): $(%$(name_1), %$(name_2))$" : L"(%$(ALPHABET[ℓ])): (%$(name_1), %$(name_2))",
            titlealign=:left; axis_kwargs...)
    else
        ax = Makie.Axis(fig[i, j],
            xlabel=L"%$(name_1)",
            ylabel=L"%$(name_2)",
            title=name_1 isa Symbol ? L"(%$(ALPHABET[ℓ])): $(%$(name_1), %$(name_2))$" : L"(%$(ALPHABET[ℓ])): (%$(name_1), %$(name_2))",
            titlealign=:left)
    end
    conf = ProfileLikelihood.get_confidence_regions(prof)
    conf_x = conf.x
    conf_y = conf.y
    prof_vals[isnan.(prof_vals)] .= minimum(prof_vals[.!isnan.(prof_vals)])
    Makie.heatmap!(ax, grid_1, grid_2, prof_vals, colorrange=(-12, 0))
    if !smooth_confidence_boundary
        if close_contour
            Makie.lines!(ax, [conf_x..., conf_x[begin]], [conf_y..., conf_y[begin]], color=:red, linewidth=3)
        else
            Makie.lines!(ax, conf_x, conf_y, color=:red, linewidth=3)
        end
    else
        A = [conf_x conf_y]
        itp = Interpolations.scale(interpolate(A, (BSpline(Cubic(Natural(OnGrid()))), NoInterp())), LinRange(0, 1, length(conf_x)), 1:2)
        finer_t = LinRange(0, 1, 6length(conf_x))
        xs, ys = [itp(t, 1) for t in finer_t], [itp(t, 2) for t in finer_t]
        if close_contour
            Makie.lines!(ax, [xs..., xs[begin]], [ys..., ys[begin]], color=:red, linewidth=3)
        else
            Makie.lines!(ax, xs, ys, color=:red, linewidth=3)
        end
    end
    if !isnothing(true_vals) && (!isnothing(true_vals[1]) && !isnothing(true_vals[2]))
        Makie.scatter!(ax, [true_vals[1]], [true_vals[2]], color=:black, markersize=12)
    end
    if !isnothing(mle_val)
        Makie.scatter!(ax, [mle_val[1]], [mle_val[2]], color=:red, markersize=12)
    end
    !isnothing(xlim_tuple) && Makie.xlims!(ax, xlim_tuple[1], xlim_tuple[2])
    !isnothing(ylim_tuple) && Makie.ylims!(ax, ylim_tuple[1], ylim_tuple[2])
    return nothing
end

end