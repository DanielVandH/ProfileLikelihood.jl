# Docstrings 

Here we give some of the main docstrings. 

## LikelihoodProblem 

```@docs 
ProfileLikelihood.AbstractLikelihoodProblem 
LikelihoodProblem
```

## LikelihoodSolution 

```@docs 
ProfileLikelihood.AbstractLikelihoodSolution
ProfileLikelihood.LikelihoodSolution 
mle
```

## ProfileLikelihoodSolution

```@docs 
ProfileLikelihood.ProfileLikelihoodSolution 
ProfileLikelihood.ConfidenceInterval
profile 
replace_profile!
refine_profile!
ProfileLikelihood.set_next_initial_estimate!(::Any, ::Any, ::Any, ::Any, ::Any)
ProfileLikelihood.get_confidence_intervals!
ProfileLikelihood.reach_min_steps!
```

## BivariateProfileLikelihoodSolution 

```@docs 
ProfileLikelihood.BivariateProfileLikelihoodSolution 
ProfileLikelihood.ConfidenceRegion 
bivariate_profile
ProfileLikelihood.set_next_initial_estimate!(::Any, ::Any, ::CartesianIndex, ::Any, ::Any, ::Any, ::Any, ::Val{M}) where M
```

## Prediction intervals 

```@docs 
get_prediction_intervals 
```

## Plotting 

```@docs 
plot_profiles 
```

## GridSearch

### Grid definitions 

```@docs 
ProfileLikelihood.AbstractGrid 
ProfileLikelihood.RegularGrid 
ProfileLikelihood.FusedRegularGrid
ProfileLikelihood.IrregularGrid 
```

### Performing a grid search 

```@docs 
ProfileLikelihood.GridSearch
grid_search 
```