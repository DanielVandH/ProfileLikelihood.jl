# Mathematical and Implementation Details

We now give some of the mathematical and implementation details used in this package, namely for computing the profile likelihood function and for computing prediction intervals.

## Computing the profile likelihood function 

Let us start by giving a mathematical description of the method that we use for computing the profile log-likelihood function. Suppose that we have a parameter vector $\boldsymbol\theta$ that we partition as $\boldsymbol \theta = (\boldsymbol\psi, \boldsymbol \omega)$ - $\boldsymbol\psi$ is either a scalar, $\psi$, or a 2-vector, $(\psi, \varphi)$. We suppose that we have a likelihood function $\mathcal L(\boldsymbol \theta) \equiv \mathcal L(\boldsymbol\psi, \boldsymbol \omega)$ so that the normalised profile log-likelihood function for $\boldsymbol\psi$ is defined as 

```math
\hat\ell_p(\boldsymbol\psi) = \sup_{\boldsymbol \omega \in \Omega \mid \boldsymbol\psi} \left[\ell(\boldsymbol\psi, \boldsymbol\omega) - \ell^*\right],
```

where $\Omega$ is the parameter space for $\boldsymbol \omega$, $\ell(\boldsymbol\psi,\boldsymbol\omega) = \log \mathcal L(\boldsymbol\psi, \boldsymbol \omega)$, and $\ell^* = \ell(\hat{\boldsymbol \theta})$, where $\boldsymbol \theta$ are the MLEs for $\boldsymbol \theta$. This definition of $\hat\ell_p(\boldsymbol\psi)$ induces a function $\boldsymbol\omega^*(\boldsymbol\psi)$ depending on $\boldsymbol\psi$ that gives the values of $\boldsymbol \omega$ leading to the supremum above, i.e. 

```math
\ell(\boldsymbol\psi, \boldsymbol\omega^{\star}(\psi)) = \sup_{\boldsymbol \omega \in \Omega \mid \boldsymbol\psi} \left[\ell(\boldsymbol\psi, \boldsymbol\omega) - \ell^{\star}\right]. 
``` 

To compute $\hat\ell_p(\boldsymbol\psi)$, then, requires a way to efficiently compute the $\omega^*(\psi)$, and requires knowing where to stop computing. Where we stop computing the profile likelihood is simply when $\hat\ell_p(\psi) < -\chi_{k,1-\alpha}^2/2$, where $\alpha$ is the significance level and $k=1$ if $\boldsymbol\psi = \psi$ and $k=2$ if $\boldsymbol\psi = (\psi,\varphi)$. This motivates a iterative algorithm, where we start at the MLE and expand outwards.

### Univariate profile likelihoods 

We first describe our implementation for a univariate profile, in which case $\boldsymbol\psi = \psi$. The basic summary of this procedure is that we simply step to the left and to the right of the MLE, continuing until we reach the threshold.

We describe how we evaluate the function to the right of the MLE -- the case of going to the left is identical. First, we define $\psi_1 = \hat\psi$, where $\hat\psi$ is the MLE for $\psi$. This defines $\boldsymbol{\omega}_{1} = \boldsymbol{\omega}^{\star}(\psi_{1})$, which in this case just gives the MLE $\hat{\boldsymbol\theta} = (\hat\psi, \boldsymbol\omega_1)$ by definition. The value of the normalised profile log-likelihood here is simply $\hat\ell_1 = \hat\ell(\psi_1) = 0$. Then, defining some step size $\Delta\psi$, we define $\psi_2 = \psi_1 + \Delta \psi$, and in general $\psi_{j+1} = \psi_j + \Delta \psi$, we need to estimate $\boldsymbol\omega_2 = \boldsymbol \omega^*(\psi_2)$. We do this by starting an optimiser at the initial estimate $\boldsymbol\omega_2 = \boldsymbol\omega_1$ and then using this initial estimate to produce a refined value of $\boldsymbol\omega_2$ that we take as its true value. In particular, each $\boldsymbol\omega_j$ comes from starting the optimiser at the previous $\boldsymbol\omega_{j-1}$, and the value for $\hat\ell_j = \hat\ell(\psi_j)$ comes from the value of the likelihood at $(\psi_j, \boldsymbol\omega_j)$. The same holds when going to the left except with $\psi_{j+1} = \psi_j - \Delta\psi$, and then rearranging the indices $j$ when combining the results to the left and to the right.  At each step, we check if $\hat\ell_j < -\chi_{1,1-\alpha}^2/2$ and, if so, we terminate. 

Once we have terminated the algorithm, we need to obtain the confidence intervals. To do this, we fit a spline to the data $(\psi_j, \hat\ell_j)$, and use a bisection algorithm over the two intervals $(\min_j \psi_j, \hat\psi)$ and $(\hat\psi, \max_j\psi_j)$, to find where $\hat\ell_j = -\chi_{1-\alpha}^2/2$. This leads to two solutions $(L, U)$ that we take together to give the confidence interval for $\psi$. 

This is all done for each parameter.

Note that a better method for initialising the optimisation for $\boldsymbol\omega_j$ may be to use e.g. linear interpolation for the previous two values, $\boldsymbol\omega_{j-1}$ and $\boldsymbol\omega_{j-2}$ (with special care for the bounds of the parameters). We provide support for this, letting $\boldsymbol\omega_j = [\boldsymbol\omega_{j-2}(\psi_{j-1} - \psi_j) + \boldsymbol\omega_{j-1}(\psi_j - \psi_{j-2})] / (\psi_{j-1} - \psi_{j-2})$. See the `next_initial_estimate_method` option in `?profile`.

### Bivariate profile likelihoods 

Now we describe the implementation for a bivariate profile. In this case, there are many possibilities as instead of only having to think about going to the left and to the right, we could think about how we step away from the MLE in each direction, and how we want to stop iterating. This package currently has a basic implementation that we describe below, where we simply step out from the MLE in *layers*. In what follows, we let $\boldsymbol\psi = (\psi, \varphi)$.

To start, we suppose that we are on some square grid with integer coordinates $\{(i, j) : i, j = -N, -N+1, \ldots, 0, \ldots, N-1, N\}$, and we suppose that $(0, 0)$ refers to the MLE. We call the coordinate $(0, 0)$ the zeroth layer, denoted $L_0 = \{(0, 0)\}$. The $j$th layer, $L_j$, is defined to wrap around $L_{j-1}$. For example, $L_1$ wraps around $\{(0, 0)\}$ so that $L_1 = \{(-1,-1),(0,-1),(1,-1),(1,0),(1,1),(0,1),(-1,1),(-1,0)\}$. Note that $|L_j| = 8j$. The idea here is that we can solve the required optimisation problems at each $(i, j) \in L_j$, accelerating the solutions by making use of information at $L_{j-1}$ to define initial estimates for each optimisation problem. Layers are implemented via `ProfileLikelihood.LayerIterator`. 

We also need to prescribe the parameter values that are defined at each coordinate. We suppose that we have bounds $\psi_L \leq \psi \leq \psi_U$ and $\varphi_L \leq \varphi \leq \varphi_U$ for $\psi$ and $\varphi$, and that we have MLEs $\hat\psi$ and $\hat\varphi$ for $\psi$ and $\varphi$, respectively. We then define $\Delta\psi_L = (\hat\psi - \psi_L)/(N-1)$, $\Delta\psi_R = (\psi_U - \hat\psi)/(N-1)$, $\Delta\varphi_L = (\hat\varphi-\varphi_L)/(N-1)$, and $\Delta\varphi_R = (\varphi_U - \hat\varphi)/(N-1)$. With these definitions, we let

```math 
\psi_j = \begin{cases} \psi_0 + j\Delta\psi_R & j > 0, \\ \hat\psi & j = 0, \\ \psi_0 - |j|\Delta\psi_L & j < 0, \end{cases} \qquad \varphi_j = \begin{cases} \varphi_0 + j\Delta\varphi_R & j>0, \\ \hat\varphi & j = 0, \\ \varphi_0 - |j|\Delta\varphi_L & j < 0. \end{cases} 
```

We thus associate the parameter values $(\psi_i, \varphi_j)$ with the coordinate $(i, j)$. We will similarly associate $\hat\ell_p(\psi_i, \varphi_j) = \hat\ell(\psi_i, \varphi_j, \boldsymbol\omega^*(\psi_i, \varphi_j))$ with the coordinate $(i, j)$, and lastly $\boldsymbol\omega_{ij} = \boldsymbol\omega^*(\psi_i, \varphi_j)$. 

So, the procedure is as follows: First, we start at the layer $L_1$ and compute $\hat\ell_{p, ij} \equiv \hat\ell_p(\psi_i, \varphi_j)$ for each $(i, j) \in L_1$, starting each initial estimate $\boldsymbol\omega_{ij}$ at $\boldsymbol\omega_{00}$, which is just the MLE. For each $(i, j)$, the parameter values we use are $(\psi_i, \varphi_j)$. Once we have evaluated at each $(i, j) \in L_1$, we can move into $L_2$, making use of the same procedure. In this case, though, there are a few choices that we could make for choosing an initial value for $\boldsymbol\omega_{ij}$, $(i, j) \in L_2$. As defined in `set_next_initial_estimate!`, we currently have three options: 

- `:mle`: The simplest choice is to simply start $\boldsymbol\omega_{ij}$ at $\boldsymbol\omega_{00}$.
- `:nearest`: An alternative choice is to simply start $\boldsymbol\omega_{ij}$ at $\boldsymbol\omega_{i'j'}$, where $(i', j') \in L_1$ is the nearest coordinate to $(i, j) \in L_2$. For example, if $(i, j) = (-2, 0)$ then $(i', j') = (-1, 0)$, and if $(i, j) = (2, 2)$ then $(i', j') = (1, 1)$.
- `:interp`: An alternative choice, which is currently the slowest (but could be made faster in the future, it just needs some work), is to maintain a linear interpolant across the data from $L_0$ and $L_1$ (i.e., all the previous layers, so $L_j$ uses data from $L_0, L_1, \ldots, L_{j-1}$), and use extrapolation to obtain a new value for $\boldsymbol\omega_{ij}$ from the linear interpolant evaluated at $(\psi_i, \varphi_j)$. 

This procedure allows us to easily solve our optimisation problems for any given $L_j$. To decide how to terminate, we will simply terminate when we find that $\hat\ell_{p, ij} < -\chi_{2, 1-\alpha}^2/2$ for all $(i, j)$ in some layer, i.e. we do not terminate if any $\hat\ell_{p, ij} > \chi_{2, 1-\alpha}^2/2$. This choice means that we stop once we have found a box, i.e. a layer, that bounds the confidence region.

Now having a box that bounds the confidence region (assuming it to be simply connected), we need to find the boundary of the confidence region. Note that we define the confidence region as $\mathcal C = \{(\psi_i, \varphi_j) : \hat\ell_p(\psi_i, \varphi_j) < -\chi_{2,1-\alpha}^2/2\}$, and we are trying to now find the boundary $\partial\mathcal C$. We currently provide two methods for this, as defined in `get_confidence_regions`:

- `:contour`: Here we use Contours.jl, defining a contour at level $-\chi_{2,1-\alpha}^2/2$, to find the contour.
- `:delaunay`: This method uses DelaunayTriangulation.jl, defining a Delaunay triangulation over the entire grid of $(\psi_i, \varphi_j)$ in the bounding box, making use of triangulation contouring to find the contours. In particular, we take a set of triangles $\mathcal T$ that triangulate the bounding box, and then we iterate over all edges $\mathcal E$ in the triangulation. Assuming that $\hat\ell_p$ is linear over each triangle, we can assume that $\hat\ell_p$ increases linearly over an edge. Thus, if the value of the profile at one vertex of an edge is below $-\chi_{2,1-\alpha}^2/2$ and the other is above $-\chi_{2,1-\alpha}^2/2$, then there must be a point where $\hat\ell_p = -\chi_{2,1-\alpha}^2/2$ on this edge (making use of our linearity assumption). This thus defines a point on the boundary of $\partial\mathcal C$. We do this for each edge, giving us a complete boundary.

These procedures give us the complete solution.

## Computing prediction intervals

Our method for computing prediction intervals follows [Simpson and Maclaren (2022)](https://doi.org/10.1101/2022.12.14.520367), as does our description that follows. This method is nice as it provides a means for sensitivity analysis, enabling the attribution of components of uncertainty in some prediction function $q(\boldsymbol\psi, \boldsymbol \omega)$ (with $\boldsymbol\psi$ the interest parameter and $\boldsymbol\omega$ the nuisance parameters as above) to individual parameters (or pairs, in the case of a bivariate profile). The resulting intervals are called *profile-wise intervals*, with the predictions themselves called *parameter-based, profile-wise predictions* or *profile-wise predictions*.

The idea is to take a set of profile likelihoods and the confidence intervals obtained from each, and then pushing those into a prediction function that we then use to obtain prediction intervals, making heavy use of the transformation invariance property of MLEs.

So, let us start with some prediction function $q(\boldsymbol\psi, \boldsymbol \omega)$, and recall that the profile likelihood function for $\boldsymbol\psi$ induces a function $\boldsymbol\omega^{\star}(\boldsymbol\psi)$. The profile-wise likelihood for $q$, given the set of values $(\boldsymbol\psi, \boldsymbol\omega^{\star}(\boldsymbol\psi))$, is defined by 

```math
\hat\ell_p\left(q\left(\boldsymbol\psi, \boldsymbol\omega^{\star}(\boldsymbol\psi)\right) = q\right) = \sup_{\boldsymbol\psi \mid  q\left(\boldsymbol\psi, \boldsymbol\omega^{\star}(\boldsymbol\psi)\right) = q} \hat\ell_p(\boldsymbol\psi). 
```

Note that if $q(\boldsymbol\psi, \boldsymbol\omega^{\star}(\boldsymbol\psi))$ is injective, there is only one such $\boldsymbol\psi$ such that $q\left(\boldsymbol\psi, \boldsymbol\omega^{\star}(\psi)\right) = q'$ for any given $q'$ in which case the profile-wise likelihood for $q$ (based on $\boldsymbol\psi$) is simply $\hat\ell_p(\boldsymbol\psi)$. This definition is intuitive, recalling that the profile likelihood comes from a definition like the above except with the likelihood function on the right, so profile-wise likelihoods come from profile likelihoods.  Using this definition, and using the transformation invariance property of the MLE, confidence sets for $\psi$ directly translate into confidence sets for $q$, in particular to find a $100(1-\alpha)\%$ prediction interval for $q$ we need only evaluate $q$ for $\psi$ inside its confidence interval.

Let us now describe the extra details involved in obtaining these prediction intervals, in particular what we are doing in the `get_prediction_intervals` function. For this, we imagine that $q$ is scalar valued, but the description below can be easily extended to the vector case (just apply the idea to each component -- see the logistic ODE example). We also only explain this for a single parameter $\boldsymbol\psi$, but we describe how we use the results for each parameter to obtain a more conservative interval.

The first step is to evaluate the family of curves. Here we describe the evaluation for a scalar parameter of interest, $\boldsymbol\psi = \psi$, but note that the case of a bivariate parameter of interest is similar (just use a confidence region instead of a confidence interval). If we suppose that the confidence interval for $\psi$ is $(\psi_L, \psi_U)$, we define $\psi_j = \psi_L + (j-1)(\psi_U - \psi_L)/(n_\psi - 1)$, $j=1,\ldots,n_\psi$ -- this is a set of $n_\psi$ equally spaced points between the interval limits. For each $\psi_j$ we need to then compute $\boldsymbol\omega^{\star}(\psi_j)$. Rather than re-optimise, we use the data from our profile likelihoods, where we have stored values for $(\psi, \boldsymbol\omega^{\star}(\psi))$ to define a continuous function $\boldsymbol\omega^{\star}(\psi)$ via linear interpolation. Using this linear interpolant we can thus compute $\boldsymbol\omega^{\star}(\psi_j)$ for each gridpoint $\psi_j$. We can therefore compute $\boldsymbol\theta_j = (\psi_j, \boldsymbol\omega^{\star}(\psi_j))$ so that we can evaluate the prediction function at each $\psi_j$, $q_j = q(\boldsymbol\theta_j)$. 

We now have a sample ${q_1, \ldots, q_{n_\psi}}$. If we let $q_L = min_{j=1}^{n_\psi} q_j$ and $q_U = max_{j=1}^{n_\psi} q_j$, then our prediction interval is $(q_L, q_U)$. To be more specific, this is the profile-wise interval for $q$ given the basis $(\boldsymbol\psi, \boldsymbol\omega^{\star}(\boldsymbol\psi))$.

We have now described how prediction intervals are obtained based on a single parameter. Suppose we do this for a collection of parameters $\{\boldsymbol\psi^1, \ldots, \boldsymbol\psi^d\}$ (e.g. if $\boldsymbol\theta = (D, \lambda, K)$, then we might have computed profiles for $\psi^1=D$, $\psi^2=\lambda$, and $\psi^3=K$), giving $d$ different intervals for each $\boldsymbol\psi^i$, say ${(q_L^i, q_U^i)}_{i=1}^d$. We can take the union of these intervals to get a more conservative interval for the prediction, giving the new interval $(min_{i=1}^d q_L^i, max_{i=1}^d q_U^i)$.
