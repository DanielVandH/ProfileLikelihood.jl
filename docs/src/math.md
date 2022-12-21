# Mathematical and Implementation Details

We now give some of the mathematical and implementation details used in this package, namely for computing the profile likelihood function and for computing prediction intervals.

## Computing the profile likelihood function 

Let us start by giving a mathematical description of the method that we use for computing the profile log-likelihood function. Suppose that we have a parameter vector $\boldsymbol\theta$ that we partition as $\boldsymbol \theta = (\psi, \boldsymbol \omega)$ ($\psi$ is a scalar in this description, since we only support univariate profiles currently). We suppose that we have a likelihood function $\mathcal L(\boldsymbol \theta) \equiv \mathcal L(\psi, \boldsymbol \omega)$ so that the normalised profile log-likelihood function for $\psi$ is defined as 

```math
\hat\ell_p(\psi) = \sup_{\boldsymbol \omega \in \Omega \mid \psi} \left[\ell(\psi, \boldsymbol\omega) - \ell^*\right],
```

where $\Omega$ is the parameter space for $\boldsymbol \omega$, $\ell(\psi,\boldsymbol\omega) = \mathcal L(\psi, \boldsymbol \omega)$, and $\ell^* = \ell(\hat{\boldsymbol \theta})$, where $\boldsymbol \theta$ are the MLEs for $\boldsymbol \theta$. This definition of $\hat\ell_p(\psi)$ induces a function $\boldsymbol\omega^*(\psi)$ depending on $\psi$ that gives the values of $\boldsymbol \omega$ leading to the supremum above, i.e. 

```math
\ell(\psi, \boldsymbol\omega^{\star}(\psi)) = \sup_{\boldsymbol \omega \in \Omega \mid \psi} \left[\ell(\psi, \boldsymbol\omega) - \ell^{\star}\right]. 
``` 

To compute $\hat\ell_p(\psi)$, then, requires a way to efficiently compute the $\omega^*(\psi)$, and requires knowing where to stop computing. Where we stop computing the profile likelihood is simply when $\hat\ell_p(\psi) < -\chi_{1,1-\alpha}^2/2$, where $\alpha$ is the significance level (e.g. $\alpha=0.05$, in which case $\chi_{1,1-0.05}^2/2 \approx 1.92$). This motivates a iterative algorithm, where we start at the MLE and then step left and right.

We describe how we evaluate the function to the right of the MLE -- the case of going to the left is identical. First, we define $\psi_1 = \hat\psi$, where $\hat\psi$ is the MLE for $\psi$. This defines $\boldsymbol{\omega}\_{1} = \boldsymbol{\omega}^{\star}(\psi\_{1})$, which in this case just gives the MLE $\hat{\boldsymbol\theta} = (\hat\psi, \boldsymbol\omega_1)$ by definition. The value of the normalised profile log-likelihood here is simply $\hat\ell_1 = \hat\ell(\psi_1) = 0$. Then, defining some step size $\Delta\psi$, we define $\psi_2 = \psi_1 + \Delta \psi$, and in general $\psi_{j+1} = \psi_j + \Delta \psi$, we need to estimate $\boldsymbol\omega_2 = \boldsymbol \omega^*(\psi\_2)$. We do this by starting an optimiser at the initial estimate $\boldsymbol\omega_2 = \boldsymbol\omega_1$ and then using this initial estimate to produce a refined value of $\boldsymbol\omega_2$ that we take as its true value. In particular, each $\boldsymbol\omega_j$ comes from starting the optimiser at the previous $\boldsymbol\omega_{j-1}$, and the value for $\hat\ell_j = \hat\ell(\psi_j)$ comes from the value of the likelihood at $(\psi_j, \boldsymbol\omega_j)$. The same holds when going to the left except with $\hat\ell_{j+1} = \psi_j - \Delta\psi$, and then rearranging the indices $j$ when combining the results to the left and to the right.  At each step, we check if $\hat\ell_j < -\chi_{1,1-\alpha}^2/2$ and, if so, we terminate. 

Once we have terminated the algorithm, we need to obtain the confidence intervals. To do this, we fit a spline to the data $(\psi_j, \hat\ell_j)$, and use a bisection algorithm over the two intervals $(\min_j \psi_j, \hat\psi)$ and $(\hat\psi, \max_j\psi_j)$, to find where $\hat\ell_j = -\chi_{1-\alpha}^2/2$. This leads to two solutions $(L, U)$ that we take together to give the confidence interval for $\psi$. 

This is all done for each parameter.

Note that a better method for initialising the optimisation for $\boldsymbol\omega_j$ may be to use e.g. linear interpolation for the previous two values, $\boldsymbol\omega_{j-1}$ and $\boldsymbol\omega_{j-2}$ (with special care for the bounds of the parameters). We provide support for this, letting $\boldsymbol\omega_j = [\boldsymbol\omega_{j-2}(\psi_{j-1} - \psi_j) + \boldsymbol\omega_{j-1}(\psi_j - \psi_{j-2})] / (\psi_{j-1} - \psi_{j-2})$. See the `next_initial_estimate_method` option in `?profile`.

## Computing prediction intervals

Our method for computing prediction intervals follows [Simpson and Maclaren (2022)](https://doi.org/10.1101/2022.12.14.520367), as does our description that follows. This method is nice as it provides a means for sensitivity analysis, enabling the attribution of components of uncertainty in some prediction function $q(\psi, \boldsymbol \omega)$ (with $\psi$ the interest parameter and $\boldsymbol\omega$ the nuisance parameters as above) to individual parameters. The resulting intervals are called *profile-wise intervals*, with the predictions themselves called *parameter-based, profile-wise predictions* or *profile-wise predictions*.

The idea is to take a set of profile likelihoods and the confidence intervals obtained from each, and then pushing those into a prediction function that we then use to obtain prediction intervals, making heavy use of the transformation invariance property of MLEs.

So, let us start with some prediction function $q(\psi, \boldsymbol \omega)$, and recall that the profile likelihood function for $\psi$ induces a function $\boldsymbol\omega^{\star}(\psi)$. The profile-wise likelihood for $q$, given the set of values $(\psi, \boldsymbol\omega^{\star}(\psi))$, is defined by 

```math
\hat\ell_p\left(q\left(\psi, \boldsymbol\omega^{\star}(\psi)\right) = q\right) = \sup_{\psi \mid  q\left(\psi, \boldsymbol\omega^{\star}(\psi)\right) = q} \hat\ell_p(\psi). 
```

Note that if $q(\psi, \boldsymbol\omega^{\star}(\psi))$ is injective, there is only one such $\psi$ such that $q\left(\psi, \boldsymbol\omega^{\star}(\psi)\right) = q'$ for any given $q'$ in which case the profile-wise likelihood for $q$ (based on $\psi$) is simply $\hat\ell_p(\psi)$. This definition is intuitive, recalling that the profile likelihood comes from a definition like the above except with the likelihood function on the right, so profile-wise likelihoods come from profile likelihoods.  Using this definition, and using the transformation invariance property of the MLE, confidence sets for $\psi$ directly translate into confidence sets for $q$, in particular to find a $100(1-\alpha)\%$ prediction interval for $q$ we need only evaluate $q$ for $\psi$ inside its confidence interval.

Let us now describe the extra details involved in obtaining these prediction intervals, in particular what we are doing in the `get_prediction_intervals` function. For this, we imagine that $q$ is scalar valued, but the description below can be easily extended to the vector case (just apply the idea to each component -- see the logistic ODE example). We also only explain this for a single parameter $\psi$, but we describe how we use the results for each parameter to obtain a more conservative interval.

The first step is to evaluate the family of curves. If we suppose that the confidence interval for $\psi$ is $(\psi_L, \psi_U)$, we define $\psi_j = \psi_L + (j-1)(\psi_U - \psi_L)/(n_\psi - 1)$, $j=1,\ldots,n_\psi$ -- this is a set of $n_\psi$ equally spaced points between the interval limits. For each $\psi_j$ we need to then compute $\boldsymbol\omega^{\star}(\psi_j)$. Rather than re-optimise, we use the data from our profile likelihoods, where we have stored values for $(\psi, \boldsymbol\omega^{\star}(\psi))$ to define a continuous function $\boldsymbol\omega^{\star}(\psi)$ via linear interpolation. Using this linear interpolant we can thus compute $\boldsymbol\omega^{\star}(\psi_j)$ for each gridpoint $\psi_j$. We can therefore compute $\boldsymbol\theta_j = (\psi_j, \boldsymbol\omega^{\star}(\psi_j))$ so that we can evaluate the prediction function at each $\psi_j$, $q_j = q(\boldsymbol\theta_j)$. 

We now have a sample ${q_1, \ldots, q_{n_\psi}}$. If we let $q_L = min_{j=1}^{n_\psi} q_j$ and $q_U = max_{j=1}^{n_\psi} q_j$, then our prediction interval is $(q_L, q_U)$. To be more specific, this is the profile-wise interval for $q$ given the basis $(\psi, \boldsymbol\omega^{\star}(\psi))$.

We have now described how prediction intervals are obtained based on a single parameter. Suppose we do this for a collection of parameters $\{\psi^1, \ldots, \psi^d\}$ (e.g. if $\boldsymbol\theta = (D, \lambda, K)$, then we might have computed profiles for $\psi^1=D$, $\psi^2=\lambda$, and $\psi^3=K$), giving $d$ different intervals for each $\psi^i$, say ${(q_L^i, q_U^i)}\_{i=1}^d$. We can take the union of these intervals to get a more conservative interval for the prediction, giving the new interval $(min_{i=1}^d q_L^i, max_{i=1}^d q_U^i)$.
