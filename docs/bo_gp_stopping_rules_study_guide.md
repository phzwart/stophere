# Bayesian Optimization, Gaussian Processes, and Stopping Rules

This guide connects the main ideas needed for the project **When is Bayesian optimization done?** It is designed to answer four practical questions:

1. What is Bayesian optimization trying to find?
2. What does the Gaussian process learn?
3. How do acquisition and stopping rules use the GP posterior?
4. How can hyperparameter misspecification cause a confident but incorrect stop?

The central project question is:

> For a given objective landscape and acquisition strategy, which stopping rule saves evaluations while maintaining an acceptable probability of returning a good solution?

## 1. The optimization problem

Suppose an experiment accepts an input \(x\) and returns an outcome \(y\):

$$
y=f(x)+\epsilon,
$$

where:

- \(x\) is a controllable input, such as reaction temperature, material composition, or a model hyperparameter.
- \(f(x)\) is the unknown objective function.
- \(\epsilon\) represents measurement noise.
- Each evaluation of \(f\) may require money, laboratory time, or substantial computation.

For maximization, the target is

$$
x^*=\arg\max_{x\in\mathcal X} f(x).
$$

Bayesian optimization does **not** need to reconstruct the entire objective perfectly. It needs enough information to recommend a high-value input using a limited evaluation budget.

## 2. Five objects that must remain separate

| Object | Question answered | Typical output |
| --- | --- | --- |
| Objective function | What does the real experiment return at \(x\)? | Observation \(y\) |
| GP surrogate | What functions remain plausible after seeing the data? | Posterior mean and uncertainty |
| Acquisition function | Which input should be evaluated next? | \(x_{t+1}\) |
| Stopping rule | Is another evaluation worth its cost? | Continue or stop |
| Recommendation rule | Which input should be returned after stopping? | \(\hat x\) |

These objects may use the same data but solve different problems. In particular, a stopping rule and a recommendation rule are not interchangeable. A budget can force the algorithm to stop, while the recommendation rule can still return the best observed point or the maximizer of the posterior mean.

## 3. The sequential Bayesian optimization loop

At iteration \(t\), the available dataset is

$$
\mathcal D_t=\{(x_1,y_1),\ldots,(x_t,y_t)\}.
$$

One BO iteration follows this sequence:

1. Fit or update a GP using \(\mathcal D_t\).
2. Compute the posterior mean \(\mu_t(x)\) and posterior standard deviation \(\sigma_t(x)\).
3. Construct an acquisition function \(\alpha_t(x)\).
4. Solve

   $$
   x_{t+1}=\arg\max_{x\in\mathcal X}\alpha_t(x).
   $$

5. Check the stopping rule at the point defined by the implementation.
6. If the rule continues, evaluate the real objective at \(x_{t+1}\).
7. Add the new observation to the dataset and repeat.
8. When the rule stops, apply the declared recommendation rule to obtain \(\hat x\).

The exact timing of the stopping check matters. A study must state whether the rule observes the posterior before or after an evaluation and whether it can inspect the proposed next point.

## 4. What a Gaussian process represents

A Gaussian process places a probability distribution over functions:

$$
f\sim\mathcal{GP}(m(x),k(x,x')).
$$

The mean function \(m(x)\) expresses the prior central tendency. The kernel \(k(x,x')\) expresses how strongly the function values at two inputs should be related.

After conditioning on observations, the prediction at a test input \(x_*\) is Gaussian:

$$
f(x_*)\mid\mathcal D_t
\sim
\mathcal N\left(\mu_t(x_*),\sigma_t^2(x_*)\right).
$$

- \(\mu_t(x_*)\) is the current prediction.
- \(\sigma_t(x_*)\) represents uncertainty under the fitted model.

The uncertainty is conditional on the model assumptions. A narrow interval means that the fitted GP is confident. It does not prove that the GP assumptions are correct.

## 5. A numerical GP example

Assume two noisily observed points:

$$
X=\begin{bmatrix}0.25\\0.75\end{bmatrix},
\qquad
y=\begin{bmatrix}0.80\\0.20\end{bmatrix}.
$$

Use an RBF kernel with unit output scale:

$$
k(x,x')=\exp\left(-\frac{(x-x')^2}{2\ell^2}\right),
$$

and set

$$
\ell=0.25,
\qquad
\sigma_n=0.05.
$$

### 5.1 Covariance between the observations

The two points are separated by \(0.50\), so

$$
k(0.25,0.75)
=\exp\left(-\frac{0.50^2}{2(0.25)^2}\right)
=e^{-2}
\approx0.1353.
$$

After adding observation noise,

$$
C=K+\sigma_n^2I
=
\begin{bmatrix}
1.0025 & 0.1353\\
0.1353 & 1.0025
\end{bmatrix}.
$$

### 5.2 Prediction at \(x_*=0.50\)

The test input lies \(0.25\) away from both observations:

$$
k_*=\begin{bmatrix}0.6065\\0.6065\end{bmatrix}.
$$

For a zero mean GP, the posterior mean is

$$
\mu(x_*)=k_*^\top C^{-1}y\approx0.533.
$$

The posterior variance of the latent function is

$$
\sigma^2(x_*)
=k(x_*,x_*)-k_*^\top C^{-1}k_*
\approx0.353,
$$

so

$$
\sigma(x_*)\approx0.594.
$$

The GP predicts a value between the two observations but remains uncertain because \(x=0.50\) has not been evaluated.

## 6. Acquisition functions and the next evaluation

An acquisition function converts the posterior into a decision score. It is cheap to evaluate because it uses the surrogate rather than the real experiment.

### 6.1 Upper confidence bound

For maximization, a common form is

$$
\alpha_{\mathrm{UCB}}(x)=\mu(x)+\kappa\sigma(x),
$$

where \(\kappa\) controls the exploration bonus.

Suppose \(\kappa=2\):

| Candidate | \(\mu(x)\) | \(\sigma(x)\) | UCB |
| --- | ---: | ---: | ---: |
| A | 0.72 | 0.06 | 0.84 |
| B | 0.65 | 0.20 | 1.05 |

Candidate A has the higher predicted mean, but candidate B receives the larger acquisition value because its uncertainty creates more upside.

### 6.2 Other acquisition strategies in the project

- **Expected Improvement:** expected gain above the current incumbent.
- **Log Expected Improvement:** a numerically stable formulation for optimizing very small EI values.
- **Probability of Improvement:** posterior probability of beating a threshold.
- **UCB:** posterior mean plus an uncertainty bonus.
- **Max-value Entropy Search:** expected information gained about the unknown maximum value.

Each acquisition changes the trajectory. A fair stopping-rule comparison should either control the trajectory or report results separately by acquisition strategy.

## 7. GP hyperparameters and their downstream effects

### 7.1 Kernel family

The kernel family defines the allowed function shapes.

- An RBF kernel assumes very smooth functions.
- A Matérn kernel allows rougher functions.
- A periodic kernel encodes repeating structure.
- A linear kernel encodes a linear relationship.

A smooth kernel can hide a narrow or rough feature. A highly flexible model can interpret noise as structure.

### 7.2 Length scale

For the RBF kernel,

$$
k(d)=\sigma_f^2\exp\left(-\frac{d^2}{2\ell^2}\right),
$$

where \(d=|x-x'|\).

At distance \(d=0.20\):

| Length scale | Correlation |
| ---: | ---: |
| 0.10 | 0.135 |
| 0.20 | 0.607 |
| 0.50 | 0.923 |

A small length scale causes information to decay quickly. The posterior can vary rapidly, and uncertainty returns between observations. This can produce more exploration and later stopping.

A large length scale spreads information across the domain. The model becomes smoother and can reduce uncertainty in regions that were not directly evaluated. If the real objective has a narrow peak, the chain of failure can be

$$
\text{overestimated }\ell
\Rightarrow
\text{over-smoothed posterior}
\Rightarrow
\text{underestimated uncertainty}
\Rightarrow
\text{small acquisition or regret estimate}
\Rightarrow
\text{premature stop}.
$$

### 7.3 Output scale

The output scale \(\sigma_f^2\) describes the expected vertical variation.

- Too small: the model considers large peaks implausible.
- Too large: predictive intervals can remain unnecessarily wide.

Standardizing the observed outputs makes this parameter easier to fit and makes thresholds more comparable across functions.

### 7.4 Observation noise

The noise variance \(\sigma_n^2\) controls how strongly the model trusts individual observations.

- Too small: the GP may chase random measurement variation and become overconfident between observations.
- Too large: the GP can smooth away real structure and remain uncertain for too long.

Repeated measurements can help distinguish observation noise from short-scale objective structure.

### 7.5 Mean function

Far from data, the GP returns toward its prior mean. A poor mean specification can make unseen regions appear systematically attractive or weak. Standardizing \(y\) and using a fitted constant mean are common starting choices.

### 7.6 ARD length scales and input normalization

Automatic relevance determination assigns a length scale to each input dimension:

$$
\ell_1,\ell_2,\ldots,\ell_d.
$$

A smaller fitted length scale indicates faster variation along that normalized dimension. The inputs must be scaled before comparing ARD length scales. Otherwise, units such as degrees and percentages dominate their numerical interpretation.

### 7.7 Parameters outside the GP

Several important values are not GP hyperparameters:

- \(\kappa\) or \(\beta_t\) controls exploration in UCB.
- \(\xi\) changes the improvement target in EI or PI variants.
- \(\epsilon\) defines acceptable optimization error.
- \(\delta\) defines acceptable failure probability.
- Jitter stabilizes matrix computations. It should not represent scientific measurement noise.

## 8. Fitting the GP

The kernel and likelihood parameters are often fitted by maximizing the log marginal likelihood:

$$
\log p(y\mid X,\theta)
=
-\frac12(y-m)^\top C_\theta^{-1}(y-m)
-\frac12\log|C_\theta|
-\frac n2\log(2\pi),
$$

where

$$
C_\theta=K_\theta+\sigma_n^2I.
$$

The quadratic term rewards fit to the observations. The log determinant penalizes covariance structures that explain the data only by assigning excessive volume or flexibility. The balance acts as an Occam tradeoff.

A practical fitting workflow is:

1. Normalize each input dimension.
2. Standardize the observed outputs.
3. Optimize positive parameters in log space.
4. Use multiple initializations when the likelihood surface is multimodal.
5. Record whether fitted parameters hit their bounds.
6. Refit or warm-start after receiving a new observation.
7. Track hyperparameter paths across the BO trajectory.

With few observations, different combinations of length scale, output scale, and noise may explain the same data. A single maximum-likelihood estimate can therefore understate hyperparameter uncertainty.

## 9. Residuals, deviations, and calibration

For an observed value \(y_i\), define the residual

$$
r_i=y_i-\mu_i.
$$

A standardized residual is

$$
z_i=\frac{y_i-\mu_i}{\sqrt{\sigma_i^2+\sigma_n^2}}.
$$

Useful warning signs include:

- Many large \(|z_i|\): uncertainty may be underestimated.
- Residuals with systematic shape: the kernel or mean may be misspecified.
- Very wide intervals everywhere: the fitted noise may be too large or the length scale too small.
- Narrow intervals with repeated prediction errors: the model is confidently wrong.
- Hyperparameters that change sharply after each observation: the data do not yet identify the model reliably.

Coverage should be tested on benchmark functions where the truth is known. Good in-sample fit alone does not establish posterior calibration.

## 10. Stopping rules

### 10.1 Fixed budget

Stop after \(T_{\max}\) evaluations:

$$
t\ge T_{\max}.
$$

This rule is transparent but does not use the posterior.

### 10.2 Acquisition threshold

For example,

$$
\max_x \operatorname{EI}_t(x)<\tau.
$$

This rule is easy to implement, but the threshold depends on output scaling, model fit, acquisition numerics, and noise. Requiring the condition for several consecutive iterations can reduce single-iteration triggers.

### 10.3 Probability-aware regret rule

A stronger target is

$$
\Pr(r_t\le\epsilon\mid\mathcal D_t)\ge1-\delta,
$$

where \(r_t\) is simple regret. The interpretation is direct: with posterior probability at least \(1-\delta\), the final recommendation lies within \(\epsilon\) of the optimum. The guarantee remains conditional on the model and numerical approximation.

### 10.4 Decision-focused stopping

A decision-focused rule asks whether another observation has enough expected value to change or improve the final recommendation. This aligns the stopping calculation with the actual output of the optimization rather than requiring an accurate model everywhere.

### 10.5 Why no recent improvement is insufficient

A flat best-so-far curve can mean several different things:

- the optimum has been found;
- the acquisition is exploring;
- the acquisition optimizer failed;
- the surrogate is overconfident;
- a narrow, better region remains unseen.

Therefore, lack of recent improvement alone does not establish completion.

## 11. Recommendation rules and evaluation metrics

Common recommendations include:

- the best observed input;
- the input with the largest posterior mean;
- a robust recommendation that accounts for noise;
- the input minimizing posterior expected simple regret.

For a known benchmark optimum \(x^*\), maximization simple regret is

$$
r(\hat x)=f(x^*)-f(\hat x).
$$

The main study should report more than average regret:

- final simple regret;
- number of objective evaluations;
- probability of returning an \(\epsilon\)-good recommendation;
- premature-stop rate;
- non-trigger rate;
- posterior coverage and calibration;
- runtime and numerical failures.

## 12. Minimal BoTorch structure

The code should preserve the conceptual separation:

```python
for step in range(max_steps):
    model = fit_gp(train_X, train_Y)
    acquisition = build_acquisition(model, train_Y)
    x_next, acq_value = optimize_acquisition(acquisition)

    state = make_stopping_state(
        model=model,
        acquisition=acquisition,
        candidate=x_next,
        acquisition_value=acq_value,
        train_X=train_X,
        train_Y=train_Y,
    )

    if stopping_rule(state):
        break

    y_next = objective(x_next)
    train_X = append(train_X, x_next)
    train_Y = append(train_Y, y_next)

x_hat = recommendation_rule(model, train_X, train_Y)
```

This structure allows the study to change the acquisition, stopping observer, or recommendation rule without silently changing the other components.

## 13. Current project plan

### Phase 1

- Complete the ten-paper annotated reading list.
- Maintain a checked BO primer and this detailed study guide.
- Run the one-dimensional BoTorch example from a clean environment.
- Use the length-scale sensitivity example to demonstrate confident misspecification.
- Record unresolved questions for discussion.

### Next implementation steps

1. Reproduce the current examples with fixed versions and seeds.
2. Build benchmark functions with known optima, including hidden narrow peaks.
3. Implement acquisition strategies behind one consistent interface.
4. Implement stopping methods as observers of a shared trajectory where possible.
5. Run a small diagnostic matrix before launching the complete experiment.
6. Freeze configurations and record all seeds.
7. Report both evaluation savings and reliability failures.

## 14. Ten-paper map

| Paper | Role in this project |
| --- | --- |
| [Frazier, 2018](https://arxiv.org/abs/1807.02811) | Main conceptual and mathematical introduction to BO |
| [Shahriari et al., 2016](https://ieeexplore.ieee.org/document/7352306) | Broader review of surrogate models and acquisition strategies |
| [Jones et al., 1998](https://doi.org/10.1023/A:1008306431147) | Efficient Global Optimization and expected improvement |
| [Srinivas et al., 2010](https://arxiv.org/abs/0912.3995) | GP-UCB and regret analysis |
| [Wang and Jegelka, 2017](https://proceedings.mlr.press/v70/wang17e.html) | Max-value entropy search |
| [Balandat et al., 2020](https://arxiv.org/abs/1910.06403) | BoTorch architecture and differentiable acquisition optimization |
| [Ament et al., 2023](https://arxiv.org/abs/2310.20708) | LogEI and numerical stability |
| [Makarova et al., 2022](https://proceedings.mlr.press/v188/makarova22a.html) | Automated stopping based on remaining opportunity |
| [Ishibashi et al., 2023](https://proceedings.mlr.press/v206/ishibashi23a.html) | Stopping criteria connected to simple regret |
| [Wilson, 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/b204de7078301292a8876a762eed3dcb-Abstract-Conference.html) | Expected simple regret and decision-focused stopping |

## 15. Questions for the next research discussion

1. Which reliability statement should the first pilot support?
2. What values of \(\epsilon\) and \(\delta\) have scientific meaning for the intended experiments?
3. Should the first study evaluate stopping rules on recorded shared trajectories or allow each rule to change the trajectory?
4. Which recommendation rule should define simple regret in noisy settings?
5. How should matched and deliberately misspecified GP cases be separated in the results?
6. How many consecutive triggers should a threshold rule require?
7. Should hyperparameters use point estimates, priors, or marginalization in the first pilot?

