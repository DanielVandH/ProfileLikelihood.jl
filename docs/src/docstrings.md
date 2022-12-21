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
ProfileLikelihood.set_next_initial_estimate!
```

## Prediction intervals 

```@docs 
get_prediction_intervals 
eval_prediction_function 
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
ProfileLikelihood.IrregularGrid 
```

### Performing a grid search 

```@docs 
ProfileLikelihood.GridSearch
grid_search 
```