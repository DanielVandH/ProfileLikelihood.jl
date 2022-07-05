prob, loglikk, θ, dat = MultipleLinearRegression()
optprob = prob.prob 
optprobnothing = remake(optprob; lb=nothing,ub=nothing)
@test !ProfileLikelihood.finite_bounds(optprobnothing)
@test bounds(optprobnothing, 1) == (nothing, nothing)
@test bounds(optprobnothing) == [(nothing, nothing) for _ in 1:5]
optprobnothing = remake(optprob; lb=[1.0,Inf],ub=[1.0,1.0])
@test !ProfileLikelihood.finite_bounds(optprobnothing)
optprobnothing = remake(optprob; lb=[1.0,1.0],ub=[1.0,1.0])
@test ProfileLikelihood.finite_bounds(optprobnothing)

@test ProfileLikelihood.num_params(optprob) == 5