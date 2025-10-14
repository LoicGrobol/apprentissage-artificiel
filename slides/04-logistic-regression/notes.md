Some notes on linear and logistic regression
============================================

## Linear Models

### Modelling

Assume that we have are real-valued quantities $x_1, …, x_n$ and $y$ and we would like to find
a function $M$ such that:

$$y ≈ M(x_1, …, x_n)$$

In that case $M$ is a *model* of $y$ given the predictors $x_1, …, x_n$

There are many reasons why we would want to do this :

- If $M$ is cheap to compute and the $x_i$ are cheap to measure directly, but $y$ is not, this would
  give us a cheap way to estimate $y$.
  - A physical system example: it is much easier to read the height of mercury in an old-school
    thermometer and, knowing the characteristic of the thermoter, to deduce the ambient temperature
    than it is to observe and measure the movement of molecules in air.
- If $M$ has a simple expression and we want to understand the relationship between factors of
  interest $x_i$ and a phenomenon quantified by $y$.
  - For instance if we want to disentangle the influence of different factors (smoking, air
    pollution…) in the occurence of lung cancer, we can start by looking for simple quantitative
    approximations correlating their prevalences.
- If the $x_i$ either controllable or known in advance, it can help predict values of $y$ that can't
  be directly observed.
  - For instance if one of the $x_i$ is *time*, having such a model can let you know what the future
    values of $y$ (that you can of course not observe *right now*) should be.
- Sometimes, if $y$ is categorical (discrete, unordered), we'll say that $M$ is a *classifier* and
  if not, that $M$ is a *regression*.
- …

In practice, for most of the situations we encounter :

- We don't have exact measurements of any of the $x_i$ or $y$, and they are in any case only
  approximations of reality.
- Even if we did, finding a perfectly accurate relationship between them is either impossible (if
  your model of reality is inherentyl stochastic) or intractable. We will thus never have $y =
  M(x_1, …, x_n)$, but only approximations.

To take these limitations into account, we generally assume a slightly different form of our
problem :

- We look at $y$ and the $x_i$ as *random variables*, which need not be fully correlated (that is,
  there is not necessarily a deterministic relation between them). This accounts for both sources of
  imprecision: inherent stochasticity of the phenomenon we are modelling and imprecision of
  measurements).
- For any given model $M$, e define another random variable: $ε_M$, the *residual of $M$*, i.e. its
  **prediction error**:

$$ε_M = y - M(x_1, …, x_n)$$

and therefore, we have

$$y_M = M(x_1, …, x_n) + ε_f$$

Now this looks like a cheap trick: for any choice of $f$, and any values of the variable, this last
equality will always be true. But it will allow us to *reason* about what we are doing by giving a
name to the incertitudes that we have to take into account.

Other notations:

- When it is unambiguous, we write $ŷ=M(x_1, …, x_n)$ the *prediction* of our model.
- In general, we have access to a finite sample of our random variables, that is $\mathcal{D} =
  \{(X_1, y_1), …, (X_n, y_n)\}$, where $X_i = (x_{i, 1}, …, x_{i, n})$ is a vector of values from
  $(x_1, …, x_n)$.

**A word of caution** modelling and statisticss can be complex, with many conceptuals tricks (for
instance the disctinction made between concepts of *residuals*, and *errors*), inconsistency, and
general lack of expliciteness, even in textbooks (often because it's not the main concerns for
applications). I encourage you to think long and hard about what it is that you are actually doing,
not only in terms of statistical implementation, but also in terms of empistemology and philosophy
of science.

Note that this doesn't solve the intractability problems : there is an unstructured uncountable
infinity of functions $ℝ^n→ℝ$, so finding one that has (e.g.) minimal average redidual is not
feasible in general. What we do instead is look instead only in smaller and more easily managed
families of well-behaved functions.

### Linear modelling

The most simple and the best-behaved of all functions are constants, but that's of course not very
useful for most modelling problems. The next best-behaved ones are *linear* functions, that is
functions of the form

$$
\left|\begin{array}{rrl}
  M :  & ℝ^n & ⟶ ℝ \\ 
       & x_1, …, x_n & ⟼ ŷ = \sum_{i=1}^n α_ix_i = α_1x_1 + … + α_nx_n + β
 \end{array}\right.
$$

Where $β$ and the $α_i$ are real. Or in matrix form, with $α=(α_1 ⋯ α_n)$ and
$X=\begin{pmatrix}x_1\\\vdots\\x_n\end{pmatrix}$:

$$ŷ = αX + β$$

Simple!

However, for all their simplicity, with the appropriate tricks, many modelling problems can either
be reduced to or approximated by linear modelling. Most notably, there are ways to use linear
modelling thechnique to solve *polynomial* modelling problems and approximate elliptic modelling
problems (which is in fact where we get the least squares methods from).

- [Legendre and Gauß history of least squares](https://www.jstor.org/stable/2240811)


## Logistic modelling

While linear models are appropriate for many situations, one thing they don't do well is fit data
where $y$ has a bounded range: nonconstant linear functions are always unbounded, so if the variable
you are trying to predict is not, they don't really make sense.

Note that *in practice* this is not necssarily an issue: if your predictors (the $x_i$) are bounded,
$ŷ$ will be bounded too. So if you don't want to apply your model to new inputs far out of their
domain, it might be OK and plenty of works do actually just that. On the other hand, if you can, it
is better to use a more appropriate family of models such as *logistic* ones.

- <https://papers.tinbergen.nl/02119.pdf>

- Also called log-linear.
- Used for modelling a variable $y∈]0,1[$, typically a (conditional) Bernoulli probability
  distribution, can of course be shifted and scaled for other intervals.

## Classifiers

- A binary classifier between two classes $C^{+}$ and $C^{-}$ can be obtained from any scalar
  function $f: E ⟶ I⊂ℝ$: choose a threshold $τ∈I$, then sort any given sample $X$ in $C^{-}$ if
  $f(X) ≤ τ$ and sort it in $C^{+}$ otherwise.
- That works for both linear and logistic models, but the logistic model has the advantage that the
  score is in $]0, 1[$
  - Easy to interpret as a *likelihood* of being $C^{+}$ if you set $τ=0.5$.
  - They play nicer with stochastic gradient descent if you don't care about calibration: logistic
    classifiers learned with SGD tend to not be properly *calibrated*: they tend to give very high
    confidence to wrong results. There are ways to make them better at it but no silver bullet that
    I know of.
