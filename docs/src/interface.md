# Interface

The interface for defining a likelihood problem builds on top of [Optimization.jl](https://github.com/SciML/Optimization.jl). Below we list the three main structs that we use, with `LikelihoodProblem` the most important one and the only one that needs to be directly defined. Examples of how we use these structs are given later, and much extra functionality is given in the tests. Complete docstrings are given in the sidebar.

## LikelihoodProblem: Defining the likelihood problem

The `LikelihoodProblem` is the definition of a likelihood function, and provides the following constructor:

```julia
LikelihoodProblem(loglik::Function, θ₀;
    syms=eachindex(θ₀), data=SciMLBase.NullParameters(),
    f_kwargs=nothing, prob_kwargs=nothing)
```

Here, `loglik` is a function for the log-likelihood, taking the form `ℓ(θ, p)`. The second argument, `θ₀`, is the initial estimate for the parameter values. You can provide symbolic names for the parameters via `syms`, so that e.g. `prob[:α]` (where `prob` is a `LikelihoodProblem` with `:α ∈ syms`) returns the initial estimate for `:α`. The argument `p` in the likelihood function can be used to pass data or other parameters into the argument, and the keyword argument `data` can be used for this. Lastly, `f_kwargs` and `prob_kwargs` are additional keyword arguments for the `OptimizationFunction` and `OptimizationProblem`, respectively; see the [Optimization.jl](https://github.com/SciML/Optimization.jl) documentation for more detail here.

We also provide a simple interface for defining a log-likelihood that requires the solution of a differential equation:

```julia 
LikelihoodProblem(loglik::Function, θ₀,
    ode_function, u₀, tspan;
    syms=eachindex(θ₀), data=SciMLBase.NullParameters(),
    ode_parameters=SciMLBase.NullParameters(), ode_alg,
    ode_kwargs=nothing, f_kwargs=nothing, prob_kwargs=nothing)
```

Importantly, `loglik` in this case is now a function of the form `ℓ(θ, p, integrator)`, where `integrator` is the same integrator as in the integrator interface from DifferentialEquations.jl; see the documentation at DifferentialEquations.jl for more detail on using the integrator. Furthermore, `ode_function` is the function for the ODE, `u₀` its initial condition, and `tspan` its time span. Additionally, the parameters for the `ode_function` (e.g. the `p` in `ode_function(du, u, p, t)` or `ode_function(u, p, t)`) can be passed using the keyword argument `ode_parameters`. The algorithm used to solve the differential equation is passed with `ode_alg`, and lastly any additional keyword arguments for solving the problem are to be passed through `ode_kwargs`. 

The full docstrings for the three methods available are given in the sidebar.

## LikelihoodSolution: Obtaining an MLE

The MLEs for a given `LikelihoodProblem` are found using the function `mle`, e.g. `mle(prob, Optim.LBFGS())` will optimise the likelihood function using the LBFGS algorithm from Optim.jl (see also `?mle`). This function returns a `LikelihoodSolution`, defined by:

```julia
struct LikelihoodSolution{N,Θ,P,M,R,A} <: AbstractLikelihoodSolution{N,P}
    mle::Θ
    problem::P
    optimiser::A
    maximum::M
    retcode::R
end
```

If `sol isa LikelihoodSolution`, then you can use the `syms` from your original problem to access a specific MLE, e.g. `sol[:α]` would return the MLE for the paramter `:α`.

If you want to use multiple optimisers, i.e. a sequence of optimisers $(O_1, O_2, \ldots)$, in which $O_j$'s initial estimate starts from the solution from the optimiser $O_{j-1}$, you can also provide a `Tuple` into the algorithm argument, e.g. `mle(prob, (Optim.LBFGS(), NLopt.LN_NELDERMEAD))`.

The full docstring for `mle` is given in the docstring section in the sidebar, along with the docstring for `LikelihoodSolution`.

## ProfileLikelihoodsolution: Profiling the parameters 

The results for a profile likelihood, obtained from `profile(prob, sol)` (see also `?profile`), are stored in a `ProfileLikelihoodSolution` struct:

```julia
struct ProfileLikelihoodSolution{I,V,LP,LS,Spl,CT,CF,OM}
    parameter_values::Dict{I,V}
    profile_values::Dict{I,V}
    likelihood_problem::LP
    likelihood_solution::LS
    splines::Dict{I,Spl}
    confidence_intervals::Dict{I,ConfidenceInterval{CT,CF}}
    other_mles::OM
end
```

Here, the parameter values used for each parameter are given in `parameter_values`, with parameter indices (or symbols) mapped to these values. Similarly, the values of the profile log-likelihood are stored in `profile_values`. We use a spline (see Interpolations.jl) to make the profile log-likelihood a continuous function, and these splines are given by `splines`. Next, the computed confidence intervals are given in `confidence_intervals`, with a confidence interval represented by a `ConfidenceInterval` struct. Lastly, since computing the profile log-likelihood function requires an optimisation problem with one variable fixed and the others free, we obtain for each profile log-likelihood value a set of optimised parameters -- these parameters are given in `other_mles`.

If `prof` is a `ProfileLikelihoodSolution`, then you can also call it as e.g. `prof(0.5, 1)` to evaluate the profile log-likelihood function of the first parameter at the point `0.5`. Alternatively, `prof(0.7, :α)` does the same but for the parameter `:α` at the point `0.7`. You can also index `prof` at a specific index (or symbol) to see the results only for that parameter, e.g. `prof[1]` or `prof[:α]`; this returns a `ProfileLikelihoodSolutionView`.

The full docstring for `profile` and related functions are given in the sidebar.

## Propagating uncertainty: Prediction intervals 

The confidence intervals obtained from profiling can be used to obtain approximate prediction intervals via *profile-wise profile likelihoods*, as defined e.g. in [Simpson and Maclaren (2022)](https://doi.org/10.1101/2022.12.14.520367), for a prediction function $\boldsymbol q(\boldsymbol\theta)$. These intervals can be based on varying a single parameter, or by taking the union of individual prediction intervals. The main function for this is `get_prediction_intervals`. Rather than explain in full detail here, please refer to the second example below (the logistic ODE example), where we reproduce the first case study of [Simpson and Maclaren (2022)](https://doi.org/10.1101/2022.12.14.520367).

The interface we use in `get_prediction_intervals` is not too refined currently, and is most subject to change. It works for now, but I will probably make it be more generally about predictions of vector quantities, assuming a function that returns a tuple of quantities, rather than having to deal with the case of scalar vs vector vs whatever else quantities. Ideally the interface should more easily support multithreading, and the code is not the cleanest to read either. Suggestions for this interface are especially welcome.

The full docstring for `get_prediction_intervals` is given in the sidebar.

## Plotting 

We provide a function `plot_profiles` that can be useful for plotting profile likelihoods. It requires that you have done 

```julia
using CairoMakie
using LaTeXString 
```

(else the function does not exist, thanks to Requires.jl). A full description of this function is given in the corresponding docstring in the sidebar.

## GridSearch

it can sometimes be useful to evaluate the likelihood function over many points prior to optimising it, e.g. to find a good initial estimate or to just obtain data at many points for the purpose of visualisation. We provide functions for this, based on either a `RegularGrid` or an `IrregularGrid`.

A `RegularGrid` is a grid in which the grid for each parameter is uniformly spaced, so that the values for all parameter values to try fall on a lattice. An `IrregularGrid` allows for the parameters to take on whatever values you want, with the requirement that the parameter values to evaluate at are provided as a matrix with each column a different parameter set.

The function `grid_search`, after having defined a grid, can then be used for performing the grid search. The main method of interest is:

```julia
grid_search(prob::LikelihoodProblem, grid::AbstractGrid; save_vals=Val(false), parallel=Val(false))
```

Here, `grid` could be either a `RegularGrid` or an `IrregularGrid`. You can set `save_vals=Val(true)` if you want an array with all the likelihood function values, `Val(false)` otherwise. To enable multithreading, allowing for the evaluation of the function across different points via multiple threads, set `parallel=Val(true)`, otherwise leave it as `Val(false)`. The result of this grid search, if `save_vals=Val(true)`, will be `(sol, f_vals)`, where `sol` is a likelihood solution giving the parameters that gave to the highest likelihood, and `f_res` is the array of likelihoods at the corresponding parameters. If `save_vals=Val(false)`, only `sol` is returned.

More example is given in the examples, and complete docstrings are provided in the sidebar.