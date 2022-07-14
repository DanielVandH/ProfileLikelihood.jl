prob, loglikk, θ, dat = MultipleLinearRegression()
σ, β = θ
sol = mle(prob)

sol_remake = remake(sol; θ = [1,2,3,4,5])
@test sol_remake.θ == [1,2,3,4,5]
@test sol_remake.prob == prob 
@test sol_remake.alg == sol.alg 
@test sol_remake.maximum == sol.maximum 
@test sol_remake.retcode == sol.retcode 
@test sol_remake.original == sol.original

sol2 = transform_result(sol, [exp, exp, exp, exp, exp])
@test sol2.θ == exp.(mle(sol))
@test sol2.prob == prob 
@test sol2.alg == sol.alg 
@test sol2.maximum == sol.maximum 
@test sol2.retcode == sol.retcode 
@test sol2.original == sol.original

sol3 = transform_result(sol, sin)
@test sol3.θ == sin.(mle(sol))
@test sol3.prob == prob 
@test sol3.alg == sol.alg 
@test sol3.maximum == sol.maximum 
@test sol3.retcode == sol.retcode 
@test sol3.original == sol.original

@test_throws ArgumentError transform_result(sol, [sqrt, sin])
@test_throws MethodError transform_result(sol, 1)

sol4 = transform_result(sol, [identity, x -> x, sin ∘ cos, sqrt, x -> exp(x)])
@test sol4.θ == [sol.θ[1], sol.θ[2], sin(cos(sol.θ[3])), sqrt(sol.θ[4]), exp(sol.θ[5])]
@test sol4.prob == prob 
@test sol4.alg == sol.alg 
@test sol4.maximum == sol.maximum 
@test sol4.retcode == sol.retcode 
@test sol4.original == sol.original

resolution = 1000
param_ranges = ProfileLikelihood.construct_profile_ranges(prob, sol, resolution; param_bounds)
sol = mle(prob)
prof = profile(prob, sol; param_ranges)
CIs = confidence_intervals(prof)
for CI in values(CIs) 
    CI2 = transform_result(CI, exp)
    @test CI2[1] == exp(CI[1])
    @test CI2[2] == exp(CI[2])
    @test CI2.level == CI.level

    CI3 = transform_result(CI, x -> -x)
    @test CI3[1] == -CI[2] 
    @test CI3[2] == -CI[1]
    @test CI3.level == CI.level

    CI4 = transform_result(CI, identity)
    @test CI4[1] == CI[1] 
    @test CI4[2] == CI[2] 
    @test CI4.level == CI.level 
end

F = [abs, cos, x -> sin(x), exp, exp]
prof2 = transform_result(prof, F)
for i in 1:ProfileLikelihood.num_params(prof)
    @test prof2.θ[i] == F[i].(prof.θ[i])
    @test prof2.spline[i](prof2.θ[i]) ≈ prof.profile[i]
    @test prof2(prof2.θ[i], i) ≈ prof(prof.θ[i], i)
    @test bounds(prof2.confidence_intervals[i])[1] == bounds(transform_result(prof.confidence_intervals[i], F[i]))[1]
    @test bounds(prof2.confidence_intervals[i])[2] == bounds(transform_result(prof.confidence_intervals[i], F[i]))[2]
end
@test prof2.profile == prof.profile 
@test prof2.prob == prof.prob 
@test prof2.mle.θ == transform_result(sol, F).θ
@test prof2.mle.prob == transform_result(sol, F).prob

F = exp 
prof3 = transform_result(prof, F)
for i in 1:ProfileLikelihood.num_params(prof)
    @test prof3.θ[i] == F.(prof.θ[i])
    @test prof3.spline[i](prof3.θ[i]) ≈ prof.profile[i]
    @test prof3(prof3.θ[i], i) ≈ prof(prof.θ[i], i)
    @test bounds(prof3.confidence_intervals[i])[1] == bounds(transform_result(prof.confidence_intervals[i], F))[1]
    @test bounds(prof3.confidence_intervals[i])[2] == bounds(transform_result(prof.confidence_intervals[i], F))[2]
end
@test prof3.profile == prof.profile 
@test prof3.prob == prof.prob 
@test prof3.mle.θ == transform_result(sol, F).θ
@test prof3.mle.prob == transform_result(sol, F).prob

