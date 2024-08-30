module ProfileLikelihood

using SciMLBase
using InvertedIndices
using FunctionWrappers
using PreallocationTools
using StatsFuns
using SimpleNonlinearSolve
using Interpolations
using Printf
using OffsetArrays
using PolygonInbounds
using Contour
using ChunkSplitters
using SymbolicIndexingInterface

include("utils.jl")
include("problem_updates.jl")
include("abstract_type_definitions.jl")
include("likelihood_problem.jl")
include("likelihood_solution.jl")
include("confidence_interval.jl")
include("profile_likelihood_solution.jl")
include("mle.jl")
include("grid_search.jl")
include("profile_likelihood.jl")
include("display.jl")
include("prediction_intervals.jl")
include("bivariate.jl")

export LikelihoodProblem
export mle
export GridSearch
export grid_search
export RegularGrid
export IrregularGrid
export profile
export construct_profile_ranges
export get_confidence_intervals
export get_confidence_regions
export get_prediction_intervals
export gaussian_loglikelihood
export update_initial_estimate
export construct_integrator
export get_mle
export get_maximum
export get_lower_bounds
export get_upper_bounds
export get_x
export get_y
export get_parameter_values
export get_profile_values
export get_range
export replace_profile!
export construct_profile_grids
export bivariate_profile
export refine_profile!

"""
    plot_profiles(prof::ProfileLikelihoodSolution, vars = profiled_parameters(prof); 
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
        latex_names = Dict(vars .=> [L"\theta_{i}" for i in SciMLBase.sym_to_index.(vars, Ref(prof))])) 

Plot results from a profile likelihood solution `prof`. To use this function you you need to have done `using CairoMakie` (or any other Makie backend).

# Arguments 
- `prof::ProfileLikelihoodSolution`: The profile likelihood solution from [`profile`](@ref).
- `vars = profiled_parameters(prof)`: The parameters to plot.

# Keyword Arguments 
- `ncol=nothing`: The number of columns to use. If `nothing`, chosen automatically via `choose_grid_layout`.
- `nrow=nothing`: The number of rows to use. If `nothing`, chosen automatically via `choose_grid_layout`
- `true_vals=Dict(vars .=> nothing)`: A dictionary mapping parameter indices to their true values, if they exist. If `nothing`, nothing is plotted, otherwise a black line is plotted at the true value for the profile. 
- `spline=true`: Whether the curve plotted should come from a spline through the results, or if the data itself should be plotted. 
- `show_mles=true`: Whether to put a red line at the MLEs. 
- `shade_ci=true`: Whether to shade the area under the profile between the confidence interval.
- `fig_kwargs=nothing`: Extra keyword arguments for `Figure` (see the Makie docs).
- `axis_kwargs=nothing`: Extra keyword arguments for `Axis` (see the Makie docs).
- `show_points=false`: Whether to show the profile data. 
- `markersize=9`: The marker size used for `show_points`.
- `latex_names = Dict(vars .=> [L"\theta_{i}" for i in SciMLBase.sym_to_index.(vars, Ref(prof))]))`: LaTeX names to use for the parameters. Defaults to `θᵢ`, where `i` is the index of the parameter. 

# Output 
The `Figure()` is returned.

---

    plot_profiles(prof::BivariateProfileLikelihoodSolution, vars = profiled_parameters(prof); 
        ncol=nothing,
        nrow=nothing,
        true_vals=Dict(1:number_of_parameters(get_likelihood_problem(prof)) .=> nothing),
        show_mles=true,
        fig_kwargs=nothing,
        axis_kwargs=nothing,
        interpolation=false,
        smooth_confidence_boundary=false,
        close_contour=true,
        latex_names=Dict(1:number_of_parameters(get_likelihood_problem(prof)) .=> get_syms(prof)))
     
Plot results from a bivariate profile likelihood solution `prof`. To use this function you you need to have done `using CairoMakie` (or any other Makie backend).

# Arguments 
- `prof::ProfileLikelihoodSolution`: The profile likelihood solution from [`profile`](@ref).
- `vars = profiled_parameters(prof)`: The parameters to plot.

# Keyword Arguments 
- `ncol=nothing`: The number of columns to use. If `nothing`, chosen automatically via `choose_grid_layout`.
- `nrow=nothing`: The number of rows to use. If `nothing`, chosen automatically via `choose_grid_layout`
- `true_vals=Dict(1:number_of_parameters(get_likelihood_problem(prof)) .=> nothing)`: A dictionary mapping parameter indices to their true values, if they exist. If `nothing`, nothing is plotted, otherwise a black dot is plotted at the true value on the bivariate profile's plot.
- `show_mles=true`: Whether to put a red dot at the MLEs. 
- `fig_kwargs=nothing`: Extra keyword arguments for `Figure` (see the Makie docs).
- `axis_kwargs=nothing`: Extra keyword arguments for `Axis` (see the Makie docs).
- `interpolation=false`: Whether to plot the profile using the interpolant (`true`), or to use the data from `prof` directly (`false`).
- `smooth_confidence_boundary=false`: Whether to smooth the confidence region boundary when plotting (`true`) or not (`false`). The smoothing is done with a spline.
- `close_contour=true`: Whether to connect the last part of the confidence region boundary to the beginning (`true`) or not (`false`).
- `latex_names=Dict(1:number_of_parameters(get_likelihood_problem(prof)) .=> get_syms(prof)))`: LaTeX names to use for the parameters. Defaults to the `syms` names.
- `xlim_tuples=nothing`: `xlims` to use for each plot. `nothing` if the `xlims` should be set automatically.
- `ylim_tuples=nothing`: `ylims` to use for each plot. `nothing` if the `ylims` should be set automatically.

# Output 
The `Figure()` is returned.
"""
function plot_profiles end
function plot_profiles! end
export plot_profiles
export plot_profiles!
function choose_grid_layout end
function _get_confidence_regions_delaunay! end
function _get_confidence_regions_contour! end
SciMLBase.sym_to_index(vars::Integer, prof::ProfileLikelihoodSolution) = vars

if !isdefined(Base, :get_extension)
    using Requires
end

@static if !isdefined(Base, :get_extension)
    function __init__()
        @require Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a" include("../ext/ProfileLikelihoodMakieExt.jl")
        @require DelaunayTriangulation = "927a84f5-c5f4-47a5-9785-b46e178433df" include("../ext/ProfileLikelihoodDelaunayTriangulationExt.jl")
    end
end



end # module ProfileLikelihood