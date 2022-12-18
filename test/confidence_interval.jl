######################################################
## ConfidenceInterval 
######################################################
CI = PL.ConfidenceInterval(0.1, 0.2, 0.95)
@test PL.get_lower(CI) == CI.lower == 0.1
@test PL.get_upper(CI) == CI.upper == 0.2
@test PL.get_level(CI) == CI.level == 0.95
@test PL.get_bounds(CI) == (CI.lower, CI.upper) == (0.1, 0.2)
@test CI[1] == CI.lower == 0.1
@test CI[2] == CI.upper == 0.2
@test CI[begin] == CI[1] == 0.1
@test CI[end] == CI[2] == 0.2
@test length(CI) == 0.1
a, b = CI
@test a == PL.get_lower(CI)
@test b == PL.get_upper(CI)
@test_throws BoundsError a, b, c = CI
@test 0.17 ∈ CI
@test 0.24 ∉ CI
@test 0.0 ∉ CI