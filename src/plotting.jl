"""
    choose_grid_layout(num_plots, cols, rows)

Given a number of plots `num_plots`, decides the number of rows and columns 
to fit them in, or returns the provided number of `cols` / `rows` if they are not 
`nothing`. Also returns a vector `plot_positions` whose `k`th entry is a tuple `(row, col)`
giving the position for the `k`th plot.
"""
function choose_grid_layout(num_plots, cols, rows)
    if isnothing(cols) && isnothing(rows)
        cols = ceil(Int64, sqrt(num_plots))
        rows = ceil(Int64, num_plots / cols) # cowplot:::plot_grid 
    elseif isnothing(cols)
        cols = ceil(Int64, num_plots / rows)
    elseif isnothing(rows)
        rows = ceil(Int64, num_plots / cols)
    end
    plot_positions = vec([(j, i) for i in 1:cols, j in 1:rows])
    return rows, cols, plot_positions
end

"""
    plot_profile!(sol::ProfileLikelihoodSolution, fig, k, i, j, spline, true_vals, mle_val=nothing, shade_ci=true; axis_kwargs) 
    plot_profiles(sol::ProfileLikelihoodSolution; <keyword arguments>)

Plots the normalised profile log-likelihoods corresponding to the [`ProfileLikelihoodSolution`](@ref) `sol`.

# Arguments 
- `sol::ProfileLikelihoodSolution`: The [`ProfileLikelihoodSolution`](@ref).
- `axis_kwargs...`: Additional arguments for the `Makie` axes.
- `fig`: An existing figure to put the plot into.
- `k`: The index of the variable to plot for.
- `(i, j)`: The axis position to put the plot into.
- `spline`: If `spline`, the curve plotting is a spline through the actual data. If `!spline`, then the actual data is plotted.
- `true_vals`: If the true values for the parameters are known, then a vector of such values can be provided. A black vertical line will be placed at these values for the respective plots.
- `mle_val`: If this is not `nothing`, then a red vertical line is placed at the MLE for the parameter.

# Keyword Arguments 
- `ncol=nothing, nrow=nothing`: Number of columns and rows to use in the plot. See also [`choose_grid_layout`](@ref).
- `true_vals=repeat([nothing], num_params(sol))`: The true values, if any.
- `spline=true`: Whether to plot the spline through the data.
- `show_mles=true`: Whether to plot a line at the MLEs.
- `shade_ci=true`: Whether to shade the area under the curve between the confidence interval.
- `axis_kwargs...`: Additional arguments for the `Makie` axes. Should be a `NamedTuple` (when provided to `plot_profiles`).
- `fig_kwargs...`: Additional keyword arguments for the `Makie` `Figure` objects. Should be a `NamedTuple`.

# Output 
The `Figure` with the plots.
"""
function plot_profiles end
@doc (@doc plot_profiles) function plot_profile!(sol::ProfileLikelihoodSolution, fig, k, i, j, spline, true_vals, mle_val=nothing, shade_ci=true; axis_kwargs=nothing)
    param_name = names(sol)[k]
    lower_ci, upper_ci = confidence_intervals(sol)[k]
    θ_vals = sol.θ[k]
    ℓ_vals = sol.profile[k]
    conf_level = level(confidence_intervals(sol)[k])
    threshold = -0.5quantile(Chisq(1), conf_level)
    formatted_conf_level = parse(Float64, Printf.format(Printf.Format("%.2g"), 100conf_level))
    formatted_lower_ci = parse(Float64, Printf.format(Printf.Format("%.3g"), lower_ci))
    formatted_upper_ci = parse(Float64, Printf.format(Printf.Format("%.3g"), upper_ci)) # This is what @sprintf is doing, but we need to do this so that we can extract the returned value to inteprolate into LaTeXStrings
    if axis_kwargs !== nothing
        ax = CairoMakie.Axis(fig[i, j],
            xlabel=param_name,
            ylabel=L"$\ell_p^*($%$(param_name)$) - \ell^*$",
            title=L"$%$formatted_conf_level$% CI: $(%$formatted_lower_ci, %$formatted_upper_ci)$",
            titlealign=:left; axis_kwargs...)
    else
        ax = CairoMakie.Axis(fig[i, j],
            xlabel=param_name,
            ylabel=L"$\ell_p^*($%$(param_name)$) - \ell^*$",
            title=L"$%$formatted_conf_level$% CI: $(%$formatted_lower_ci, %$formatted_upper_ci)$",
            titlealign=:left; axis_kwargs...)
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
        ci_vals = lower_ci:Δθ₂:upper_ci
        shade_ci && CairoMakie.band!(ax, ci_vals, sol(ci_vals, k), repeat([threshold], length(ci_vals)), color=(:blue, 0.35))
    end
    if !isnothing(true_vals)
        CairoMakie.vlines!(ax, [true_vals], color=:black, linetype=:dashed)
    end
    if !isnothing(mle_val)
        CairoMakie.vlines!(ax, [mle_val], color=:red, linetype=:dashed)
    end
    return nothing
end
function plot_profiles(sol::ProfileLikelihoodSolution; ncol=nothing, nrow=nothing,
    true_vals=repeat([nothing], num_params(sol)), spline=true, show_mles=true, shade_ci=true, fig_kwargs=nothing, axis_kwargs=nothing) where {T<:ProfileLikelihoodSolution}
    num_plots = num_params(sol)
    _, _, plot_positions = choose_grid_layout(num_plots, ncol, nrow)
    if fig_kwargs !== nothing
        fig = CairoMakie.Figure(; fig_kwargs...)
    else
        fig = CairoMakie.Figure()
    end
    for k in 1:num_plots
        i, j = plot_positions[k]
        if axis_kwargs !== nothing
            plot_profile!(sol, fig, k, i, j, spline, true_vals[k], show_mles ? mle(sol)[k] : nothing, shade_ci; axis_kwargs)
        else
            plot_profile!(sol, fig, k, i, j, spline, true_vals[k], show_mles ? mle(sol)[k] : nothing, shade_ci)
        end
    end
    return fig
end