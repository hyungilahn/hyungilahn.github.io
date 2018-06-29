---
layout: post
title: Deep Probabilistic Programming Combines Bayesian Inference and Deep Learning
tags: Deep_Probabilistic_Programming
comments: true
---

*Deep probabilistic programming* is a method of implementing "Bayesian" probabilistic modeling on "differentiable" deep learning frameworks. This provides a style of language to define complex (composite, hierarchical) models with multiple components and incorporate probabilistic uncertainty about latent variables or model parameters into predictions.

Deep probabilistic programming can be characterized by approximate Bayesian inference [^1] (calculating the approximate posterior probability distribution of latent variables or model parameters by incorporating the information from the observed data) on differentiable programming (parametric optimization by deep learning).


[^1]: Learning the probability distributions of model parameters or latent variables is called *inference* in probabilistic modeling. Bayesian inference often involves approximating posterior distributions, since the exact calculations are often intractable. In parameterized models, variational inference (VI) approximate the posterior distributions through optimization (differential programming).

#### Why Deep and Differentiable Programming?

Deep learning frameworks (e.g., PyTorch, TensorFlow, MxNet) enable defining a target model in a *deep* and composite network structure assembling the building blocks of heterogenous component models (= parametric linear or non-linear functions including neural nets) that run in data-dependent, procedural and conditional manner. Each component may be based on different sets of feature variables. Also, they provide a tool to estimate the model parameters in terms of *differentiable* optimization like stochastic gradient decent (SGD) and back-propagation algorithms.


Since Bayesian modeling is based on a probabilistic model of the generative process relating the observed data with the uncertain latent variables (= generating parameters), it is very desirable to have the representational power of a deep and composite network model to sufficiently describe the potentially complex generative processes with multiple input variables. In addition, the exact calculation of the posterior distribution of latent variables requires doing the integral calculation to obtain the evidence of the observed data with the assumed prior distribtuion, so this is intractable in most problems. Thus, we need approximate Bayesian inference techniques, such as variational inference (VI). Thankfully, VI transforms the approximate posterior inference problems into the optimization problems searching for the best hyperparameters of approximate posterior distribution (often assumed to be Gaussian distributions). We will discuss it in detail below.

<!---
RNN time-sequence modeling + Bayesian
--->

#### Why Bayesian Inference?

Bayesian probabilistic modeling provides a unified scheme on how to update the uncertain information (or infer the posterior distributions) about modeling parameters or latent variables using observed data. It also assumes the specification of generative processes (or model functions describing how outputs are produced from inputs) and prior distributions of modeling parameters or latent variables. This specification allows for easy incorporation of prior knowledge into the model form and associated parameter uncertainty.

Although the initial choice of compared models and associated prior distributions may depend on our domain knowledge about the underlying problems, bayesian reasoning provides an objective scheme to compare different models and priors.

Bayesian modeling allows us to build more robust and less overfitted models under uncertainty and predict probabilistic estimates about target variables in the model. By incorporating the known form of a physics-based equation describing the potental causal relationships of variables in the underlying phenomenon into our modeling and set the priors on the model parameters, we can build a heuristically reasonable, more generalizable and updatable model with insufficient data.

It sounds all good and simple, but a key difficulty in Bayesian probabilistic modeling arises from calculating the posterior distributions for a given complicated model structure and prior. It is very often intractable to compute the "exact" posterior distribution, but the variational inference (VI), one of the important methods in deep probabilistic programming, present a commonly-applicable approach to compute the "approximate" bayesian posterior.

Since VI transforms Bayesian posterior inference problems (i.e., learning uncertain modeling parameters or latent variables) into optimization problems, the SGD optimization in underlying deep learning frameworks can solve the posterior inference problems.


#### Deep Probabilistic Programming = Bayesian Differentiable Programming

Compared to non-probabilistic deep learning frameworks [^2] that aim to make "deterministic" models for point-estimate predictions, *deep probabilistic programming* frameworks enables us to 1) specify "probabilistic" models involving the uncertain distributions of model parameters or latent variables, 2) provide approximate Bayesian inferences (e.g., variational inferences) using the powerful stochastic gradient decent algorithms of the original deep learning. Thus, deep probabilistic programming naturally integrates the benefits of bayesian modeling and deep learning.

 [^2]: Non-probabilistic deep learning models do not assume the uncertainty in model parameters (e.g., weights in NN).





#### *Application*: Bayesian Regression with a Parametric Function (e.g., knowledge-based known function form, NNs)

In Bayesian modeling we posit that the model parameters (a.k.a. latent or hidden variables) generating the observed data are *uncertain* in our knowledge. Thus, our information about the true values of *generating* variables is described by a probability. That is, we use a probability to denote our uncertainty about the hidden variables selected to describe the generating process.

Suppose we have a dataset $$D = ({\mathbf{X}, \mathbf{y}}) = \{(\mathbf{x}_i,y_i) \mid i=1,2,...,N\}$$ where each data point has feature input vector $$\mathbf{x}_i\in\mathbb{R}^K$$ and observed output variable $$y_i\in\mathbb{R}.$$ ​The goal of Bayesian regression is to fit a function to the data: $$y = f(\mathbf{x}; \mathbf{w} ) + \epsilon$$ assuming that $$\mathbf{w}$$ is the *uncertain* latent variables described by a probability distribution.

<!---
Let's take the example of a Bayesian parametric regression such as
$$y = f(\mathbf{x}; \mathbf{w}) + \epsilon $$ where $$\mathbf{x}$$ and $$y$$ are the given input vector and the observed output variable (scalar), and $$\mathbf{w}$$ is the *uncertain* latent variables (vector) or the generating parameters described by a probability distribution.
--->

There are important modeling assumptions here.
- $$f(\mathbf{x}; \mathbf{w})$$ is an assumed generating function we specify with unexplained error $$\epsilon \sim \mathrm{Normal}(0, \sigma_y^2)$$.
- $$f(\mathbf{x}; \mathbf{w})$$ is a deterministic function for any sampled value of $$\mathbf{w} \sim \mathrm{Normal} (\mathbf{0}, \sigma_w^2 \mathbf{I})$$. The function $$f$$ may be any known form of an equation or a neural network involving the model parameters $$\mathbf{w}$$.
- The level of $$\sigma_y^2$$ is assumed to be fixed as a constant value and also related to how accurately we may specify our function $$f(\mathbf{x}; \mathbf{w})$$.

Whereas a non-Bayesian (deterministic) approach views $$\mathbf{w}$$ as a fixed variable to be estimated, a Bayesian (probabilistic) approach regards $$\mathbf{w}$$ as an uncertain variable whose *probability distribution* is to be estimated to explain the observed data. Maximum likelihood (ML) or maximum a posteriori (MAP) estimations are well-known non-Bayesian approaches determining a fixed $$\mathbf{w}$$.

Now let's represent the above Bayesian regression in terms of probability distributions.

In the Bayesian perspective the complete generative process should be always described in the joint probability distribution of *all observed and latent variables*.

Since $$\mathbf{x}$$ is given and $$\sigma_y^2$$ and $$\sigma_w^2$$ are known and  fixed, the joint distribution for the complete generative process is $$p(y,\mathbf{w} \mid \mathbf{x}, \sigma_y^2, \sigma_w^2, \mathcal{H})$$ where $$\mathcal{H}$$ denotes the hypothesis space of model form $$y = f(\mathbf{x} ; \mathbf{w}) + \epsilon$$.

Factorizing $$p(y,\mathbf{w} \mid \mathbf{x}, \sigma_y^2, \sigma_w^2, \mathcal{H}) = p(y \mid \mathbf{w},
\mathbf{x}, \sigma_y^2, \mathcal{H}) p(\mathbf{w} \mid \sigma_w^2)$$, we represent the complete generative process in the combination of the likelihood and the prior distributions. It is important to note that the exact forms of the likelihood and the prior distributions are part of our modeling assumptions.

- The *likelihood* $$p(y \mid \mathbf{w}, \mathbf{x}, \sigma_y^2, \mathcal{H})$$ is our assumed probability model to describe a generating process of the observed variable $$y$$ from a sample of latent variables $$\mathbf{w}$$. Assuming that the likelihood is normally distributed with $$f(\mathbf{x}; \mathbf{w})$$ as $$\mu_y$$ (= the expected value of $$y$$) and $$\sigma_y^2$$ as the Gaussian noise level,
$$
\begin{aligned}
y \sim p(y \mid \mathbf{w}, \mathbf{x}, \sigma_y^2,\mathcal{H}) = \mathrm{Normal}(\mu_y, \sigma_y^2 ) = \mathrm{Normal}(f(\mathbf{x}; \mathbf{w}), \sigma_y^2 ).
\end{aligned}
$$

Note that a known deterministic physical model $$\mu_y = f(\mathbf{x}; \mathbf{w})$$ can be easily incorporated into the likelihood.

- The *prior* $$p(\mathbf{w} \mid \sigma_w^2)$$ is our assumed probability model to represent the uncertain information of latent variables $$\mathbf{w}$$ (= model parameters) before we consider the observed data $$y$$.
$$\mathbf{w} \sim p(\mathbf{w} \mid \sigma_w^2) = \mathrm{Normal} (\mathbf{0},\sigma_w^2 \mathbf{I})$$.


Bayesian posterior inference is to update our probabilistic information about $$\mathbf{w}$$ after observing the data $$y$$ and considering the likelihood.  Mathematically the Bayes' rule (a.k.a. inverse probability) provides this update rule and the posterior distribution representing the uncertainty about $$\mathbf{w}$$ is computed as

$$
\begin{aligned}
p(\mathbf{w}| \mathbf{y}, \mathbf{X}, \sigma_y^2,\sigma_w^2,\mathcal{H})
= \frac{ p(\mathbf{y} \mid \mathbf{w},
\mathbf{X}, \sigma_y^2, \mathcal{H}) p(\mathbf{w} \mid \sigma_w^2)} { p(\mathbf{y} \mid \mathbf{X}, \sigma_y^2,\sigma_w^2, \mathcal{H}) }
= \frac{ p(\mathbf{y} \mid \mathbf{w},
\mathbf{X}, \sigma_y^2, \mathcal{H}) p(\mathbf{w} \mid \sigma_w^2)} { \int p(\mathbf{y} \mid \mathbf{w},
\mathbf{X}, \sigma_y^2, \mathcal{H}) p(\mathbf{w} \mid \sigma_w^2)  \,\mathrm{d} \mathbf{w} }.
\end{aligned}
$$



$$\mathrm{Posterior}$$ = $$\frac{\mathrm{Likelihood} \times \mathrm{Prior} }{\mathrm{Evidence}}$$.


The *evidence* $$p(\mathbf{y} \mid \mathbf{X}, \sigma_y^2,\sigma_w^2,\mathcal{H})$$ is the normalizing constant of the posterior distribution and should be calculated to obtain the exact posterior distribution [^3]. It is the probability that the model observes the data with the assumed likelihood and the prior distributions. Note that it is also the marginalized probability of *likelihood $$\times$$ prior* over the uncertain parameters $$\mathbf{w}$$.
Since this is intractable to be exactly calculated, we involve an approximate Bayesian inference such as variational inference (VI) or Markov chain Monte Carlo (MCMC).

[^3]: Compared to the Bayesian approach requiring the estimation of the evidence and the *posterior distribution*, the maximum a posteriori (MAP) estimation simply considers the point parameter estimate $$\mathbf{w_\mathrm{MP}}$$ maximizing the numerator part of the posterior (= Likelihood $$\times$$ Prior). $$\mathbf{w_\mathrm{MP}}= \arg \displaystyle\max_{\mathbf{w}} \log \big(p(\mathbf{y} \mid \mathbf{W}, \mathbf{X}, \sigma_y^2, \mathcal{H}) p(\mathbf{w} \mid \sigma_w^2)\big)$$

<!---
$$p(y \mid \mathbf{x}, \sigma_y^2)
= \int p(y, \mathbf{w} |
\mathbf{x}, \sigma_y^2) \,\mathrm{d} \mathbf{w}
= \int p(y \mid \mathbf{w},
\mathbf{x}, \sigma_y^2) p(\mathbf{w} \mid \sigma_w^2)  \,\mathrm{d} \mathbf{w}$$.
--->

$$p(\mathbf{y} \mid \mathbf{X}, \sigma_y^2,\sigma_w^2,\mathcal{H})
= \int p(\mathbf{y}, \mathbf{w} |
\mathbf{X}, \sigma_y^2,\sigma_w^2,\mathcal{H}) \,\mathrm{d} \mathbf{w}
= \int p(\mathbf{y} \mid \mathbf{w},
\mathbf{X}, \sigma_y^2,\mathcal{H}) p(\mathbf{w} \mid \sigma_w^2)  \,\mathrm{d} \mathbf{w}$$

Importantly, the Bayesian prediction of $$y'$$ for a new $$\mathbf{x}'$$ is essentially the marginalization of *likelihood $$\times$$ posterior* over the uncertain parameters $$\mathbf{w}$$, incorporating the *posterior* uncertainty about parameters $$\mathbf{w}$$ into predictions.
<!---
$$p(y' \mid \mathbf{x}', y, \mathbf{x}, \sigma_y^2, \sigma_w^2)
 = \int p(y' \mid \mathbf{w},
\mathbf{x}', \sigma_y^2) p(\mathbf{w} \mid y, \mathbf{x}, \sigma_y^2, \sigma_w^2)  \,\mathrm{d} \mathbf{w}$$.
--->
$$p(y' \mid \mathbf{x}', \mathbf{y}, \mathbf{X}, \sigma_y^2, \sigma_w^2,\mathcal{H})
 = \int p(y' \mid \mathbf{w},
\mathbf{x}', \sigma_y^2,\mathcal{H}) p(\mathbf{w} \mid \mathbf{y}, \mathbf{X}, \sigma_y^2, \sigma_w^2,\mathcal{H})  \,\mathrm{d} \mathbf{w}$$


### Variational Inference

For simpler notations, we define the *generating parameters* $$\theta = (\sigma_y^2, \sigma_w^2) = (1/\beta, 1/\alpha)$$. Note that $$\beta = 1/\sigma_y^2$$ posits the likelihood variance (or unexplained error level) and $$\alpha = 1/\sigma_w^2$$ posits the prior variance (or model parameter uncertainty).  Now we can denote
the prior $$p(\mathbf{w} \mid \sigma_w^2)$$ by $$p(\mathbf{w} \mid
\alpha)$$ and the posteror $$p(\mathbf{w} \mid \mathbf{y}, \mathbf{X}, \sigma_y^2, \sigma_w^2, \mathcal{H})$$ by $$p(\mathbf{w} \mid D, \theta,  \mathcal{H})$$.

The prior distribution: $$p(\mathbf{w} \mid \alpha) = p(\mathbf{w} \mid \sigma_w^2)$$

The posterior distribution: $$p(\mathbf{w} \mid D, \theta,  \mathcal{H})$$ = $$p(\mathbf{w} \mid \mathbf{y}, \mathbf{X}, \sigma_y^2, \sigma_w^2, \mathcal{H})$$

The aim of variational inference is to approximate the posterior $$p(\mathbf{w} \mid D, \theta, \mathcal{H})$$ by a simpler probability distribution $$q(\mathbf{w} \mid \lambda) = \mathrm{Normal} (\mathbf{\lambda}_{\mu},
\mathrm{diag}( {\lambda_\sigma^{-1}} ))$$ where $$\lambda$$ is called the *variational parameters*.

$$
\begin{aligned}
  \lambda^* &= \arg\min_\lambda \text{KL}(
  q(\mathbf{w}|\lambda)
  \;\|\;
  p(\mathbf{w}|D,\theta, \mathcal{H})
  )
\end{aligned}
$$

$$
\begin{aligned}
  \text{KL}(
  q(\mathbf{w}|\lambda)
  \;\|\;
  p(\mathbf{w}|D,\theta, \mathcal{H})
  ) &=
  \mathbb{E}_{q(\mathbf{w}|\lambda)}
  \big[
  \log q(\mathbf{w}|\lambda)-
  \log p(\mathbf{w}|D,\theta, \mathcal{H})
  \big]\\ &=
  \mathbb{E}_{q(\mathbf{w}|\lambda)}
  \big[
  \log q(\mathbf{w}|\lambda)-
  \log \frac{p(\mathbf{w},D|\theta, \mathcal{H})}{p(D|\theta, \mathcal{H})}
  \big]\\ &=
  \mathbb{E}_{q(\mathbf{w}|\lambda)}
  \big[
  \log q(\mathbf{w}|\lambda)-
  \log p(\mathbf{w},D|\theta, \mathcal{H})
  \big] + \log p(D|\theta, \mathcal{H})\\ &
  \geq 0
\end{aligned}
$$

Therefore,

$$
\begin{aligned}
  \log p(D|\theta, \mathcal{H}) &=
  \mathbb{E}_{q(\mathbf{w}|\lambda)}
  \big[
  \log p(\mathbf{w},D|\theta, \mathcal{H}) - \log q(\mathbf{w}|\lambda)
  \big] +
  \text{KL}(
    q(\mathbf{w}|\lambda)
    \;\|\;
    p(\mathbf{w}|D,\theta, \mathcal{H})
    ) \\ &
\geq \mathbb{E}_{q(\mathbf{w}|\lambda)}
\big[
\log p(\mathbf{w},D|\theta, \mathcal{H}) - \log q(\mathbf{w}|\lambda)
\big]
\end{aligned}
$$

Note that the log evidence $$\log p(D|\theta, \mathcal{H})$$ is lower-bounded because $$\text{KL}(
  q(\mathbf{w}|\lambda)
  \;\|\;
  p(\mathbf{w}|D,\theta, \mathcal{H})
)$$ is always greater than 0. The lower bound $$\mathbb{E}_{q(\mathbf{w}|\lambda)}
\big[
\log p(\mathbf{w},D|\theta, \mathcal{H}) - \log q(\mathbf{w}|\lambda)
\big]$$ is called the *Evidence Lower Bound (ELBO)* or the *variational free energy*. Since $$\log p(D|\theta, \mathcal{H})$$ is fixed for a given $$\theta$$ (prior variance $$\sigma_w^2$$ and likelihood variance $$\sigma_y^2$$) and the  hypothesis space $$\mathcal{H}$$ of the model function form $$f$$, the maximization of ELBO by adjusting the *variational parameters* $$\lambda$$ (or $$q(\mathbf{w}|\lambda)$$) minimizes $$\text{KL}(
  q(\mathbf{w}|\lambda)
  \;\|\;
  p(\mathbf{w}|D,\theta, \mathcal{H})
)$$, which makes the approximate posterior distribution $$q(\mathbf{w}|\lambda)$$ closer to the true posterior distribution $$p(\mathbf{w}|D,\theta, \mathcal{H})$$. Also, note that when ELBO is maximized up to the log evidence $$\log p(D|\theta, \mathcal{H})$$, $$\text{KL}(q||p)$$ becomes 0, making the approxmiate posterior $$q(\mathbf{w}|\lambda)$$ the same as the exact posterior $$p(\mathbf{w}|D,\theta, \mathcal{H})$$.

The existence of ELBO transforms the posterior inference problem into the optmization problem, which can be solved using the minus ELBO as the loss function of SGD algorithm in DL frameworks. In addition, ELBO can be represented in a different way to clarify how the approximate posterior distribution of parameters $$q(\mathbf{w} \mid \lambda)$$ shaped by $$\lambda$$ reckons the balance between the likelihood and the prior.

$$
\begin{aligned}
\text{ELBO}(\lambda) & = \mathbb{E}_{q(\mathbf{w}|\lambda)}
\big[
\log p(\mathbf{w},D|\theta, \mathcal{H}) - \log q(\mathbf{w}|\lambda)
\big] \\& =
\mathbb{E}_{q(\mathbf{w}|\lambda)}
\big[
\log p(D|\mathbf{w},\theta, \mathcal{H}) p(\mathbf{w}|\theta) - \log q(\mathbf{w}|\lambda)
\big] \\& =
\mathbb{E}_{q(\mathbf{w}|\lambda)}
\big[
\log p(D|\mathbf{w},\beta, \mathcal{H}) p(\mathbf{w}|\alpha) - \log q(\mathbf{w}|\lambda)
\big] \\& =
\mathbb{E}_{q(\mathbf{w}|\lambda)}
\big[
\log p(D|\mathbf{w},\beta, \mathcal{H}) - \log \frac{q(\mathbf{w}|\lambda)}{ p(\mathbf{w}|\alpha)}
\big] \\& =
\mathbb{E}_{q(\mathbf{w}|\lambda)}
\big[
\log p(D|\mathbf{w},\beta, \mathcal{H})] - \text{KL}(q(\mathbf{w}|\lambda) \;\|\; p(\mathbf{w}|\alpha))
\end{aligned}
$$

The first term $$\mathbb{E}_{q(\mathbf{w}|\lambda)}
\big[
\log p(D|\mathbf{w},\beta, \mathcal{H})]$$ motivates the distribution $$q(\mathbf{w}|\lambda)$$ to concentrate on the $$\mathbf{w}$$ values with which the model have higher likelihood $$p(D|\mathbf{w},\beta, \mathcal{H})$$, whereas the second term the minus $$ \text{KL}(q(\mathbf{w}|\lambda) \;\|\; p(\mathbf{w}|\alpha))$$ inspires the approximate prior $$q(\mathbf{w}|\lambda)$$ to be less deviated from the given prior distribution $$p(\mathbf{w}|\alpha)$$.

### Prediction Errors as Probabilistic Interpretations

In Bayesian approach we make our assumptions on both likelihood and prior distributions. The likelihood distribution posits the model function form $$f(\mathbf{x};\mathbf{w})$$ and the unexplained error level $$\sigma_y^2 = 1/\beta$$. The prior distribution has an assumed variance $$\sigma_w^2 = 1/\alpha$$.

Importantly, the Bayesian prediction of $$y'$$ for a new $$\mathbf{x}'$$ is essentially the marginalization of *likelihood $$\times$$ posterior* over the uncertain parameters $$\mathbf{w}$$, incorporating the *posterior* uncertainty about parameters $$\mathbf{w}$$ into predictions.

$$
\begin{aligned}
p(y' \mid \mathbf{x}', D, \theta,\mathcal{H}) &=
\int p(y' \mid \mathbf{w},
\mathbf{x}', \sigma_y^2,\mathcal{H}) \; q(\mathbf{w} \mid \lambda)  \,\mathrm{d} \mathbf{w} \\ &=
\int \frac{1}{\sqrt{(2\pi)} \sigma_y}\exp(-\frac{1}{2 \sigma_y^2} \sum_{i=1}^N (y'_i - f(\mathbf{x}'_i; \mathbf{w}) )^2) \\ & \frac{1}{\sqrt{(2\pi)^K} \sigma_{q(\mathbf{\mathbf{w} \mid \lambda)}} } \exp\big(- \frac{(\mathbf{w} - \mu_{q(\mathbf{\mathbf{w} \mid \lambda)} })^2}{2 {\sigma_{q(\mathbf{\mathbf{w} \mid \lambda)}}}^2  } \big) \,\mathrm{d} \mathbf{w} \\ &=
\int \frac{1}{Z_D(\beta)}\exp(-\beta E_D) \frac{1}{Z_W(\alpha)}\exp(-\alpha E_W)\,\mathrm{d} \mathbf{w}
\end{aligned}
$$

First, $$\beta E_D = \frac{1}{2 \sigma_y^2} \sum_{i=1}^N (y'_i - f(\mathbf{x}'_i; \mathbf{w})^2 ) = \frac{1}{2\sigma_y^2} \sum_{i=1}^N \epsilon_i^2  = \frac{1}{2\sigma_y^2} \sum_{i=1}^N (n_i^2 + e_i^2) = \frac{N}{2}(\frac{\sigma_n^2 + \sigma_{\mathrm{bias}}^2}{\sigma_y^2})$$

Note that the unexplained error $$\epsilon_i = y'_i - f(\mathbf{x}'_i;\mathbf{w})$$ = $$t_i + n_i - f(\mathbf{x}'_i;\mathbf{w})$$ = $$n_i + (t_i - f(\mathbf{x}'_i;\mathbf{w}))$$ = $$n_i + e_i$$, since $$y'_i = t_i + n_i $$ where $$t_i$$ is the hidden true value (before a noise is being added) and $$n_i$$ is the inherent noise. Also, $$e_i = t_i - f(\mathbf{x}'_i;\mathbf{w})$$ is the misspecified model bias error, $$\sigma_{\mathrm{bias}} ^2 = \frac{1}{N}\sum_{i=1}^N e_i^2$$ and $$\sigma_n ^2 = \frac{1}{N}\sum_{i=1}^N n_i^2$$.

Thus, the likelihood for a given $$\mathbf{w}$$ is shaped by both the misspecified model bias error and the inherent error. As the ratio of the ratio $$\frac{\sigma_{\mathrm{bias}}^2}{\sigma_y^2}$$ goes up, the likelihood goes down. In addition, the normalizing factor $$\frac{1}{\sqrt{(2\pi)} \sigma_y}$$ decreases the likelihood with higher $$\sigma_y$$.

Second, the term $$\alpha E_W = \frac{(\mathbf{w} - \mu_{q(\mathbf{\mathbf{w} \mid \lambda)} })^2}{2 {\sigma_{q(\mathbf{\mathbf{w} \mid \lambda)}}}^2 }$$ where $$\alpha = \frac{1}{2 {\sigma_{q(\mathbf{\mathbf{w} \mid \lambda)}}}^2 } $$ put different weights over $$\mathbf{w}$$ values. That is, $$\mathbf{\mathbf{w}}$$ closer to the current prior mean $$\mu_{q(\mathbf{\mathbf{w} \mid \lambda)} }$$ is more weighted, allowing for the tendency toward a smaller deviation from the current prior. In particular, when $$\mu_{q(\mathbf{\mathbf{w} \mid \lambda)} } \simeq 0$$, $$\;\; \alpha E_W \simeq \frac{\alpha}{2} \mid \mathbf{w} \mid ^2$$ corresponds to a regularization term preferring small values of $$\mathbf{w}$$ decreasing the tendency of overfitting and higher *model parameter uncertainty* (= *model variance error*).


<!--
The likelihood for a given $$\mathbf{w}$$ can be also described in terms of the inherent noise error (the *irreducible* error due to the random nature of the underlying system or the observation process) and the model bias error (e.g., model misspecification).

$$
\begin{aligned}
p(D|\mathbf{w},\beta, \mathcal{H}) &=
p(\mathbf{y} \mid \mathbf{w}, \mathbf{X}, \sigma_y^2, \mathcal{H})\\ &=
\textstyle\prod_{i=1}^N \; p(y_i \mid \mathbf{w}, \mathbf{x}_i, \sigma_y^2, \mathcal{H})
= \textstyle\prod_{i=1}^N \;\mathrm{Normal}(y_i ; f(\mathbf{x}_i; \mathbf{w}), \sigma_y^2 ) \\ &=
\frac{1}{Z_D(\beta)}\exp(-\frac{\beta}{2} \sum_{i=1}^N (y_i - f(\mathbf{x}_i; \mathbf{w}))^2)
\end{aligned}
$$
-->


In general, when the set of features used in modeling are:
  * only the essential features
    - we tend to obtain a generalizable model with a good fit
  * essential features + many irrelevant features
    - too complicated model (overfit) and high model variance (model parameter uncertainty)
  * insufficient essential features
    - too simple model (underfit) and high model bias (model misspecification)
  * insufficient essential features + many irrelevant features
    - non-generalizable model with a bad fit



    <!---
    $$p(y' \mid \mathbf{x}', y, \mathbf{x}, \sigma_y^2, \sigma_w^2)
     = \int p(y' \mid \mathbf{w},
    \mathbf{x}', \sigma_y^2) p(\mathbf{w} \mid y, \mathbf{x}, \sigma_y^2, \sigma_w^2)  \,\mathrm{d} \mathbf{w}$$.
    --->

    <!--
    When the posterior $$p(\mathbf{w} \mid D, \theta, \mathcal{H})$$ is approxmiated by $$q(\mathbf{w} \mid \lambda)$$, the predicted $$y'$$ for a new $$\mathbf{x}'$$ has the distribution

    $$
    \begin{aligned}
    p(y' \mid \mathbf{x}', D, \theta,\mathcal{H}) &=
    \int p(y' \mid \mathbf{w},
    \mathbf{x}', \sigma_y^2,\mathcal{H}) \; q(\mathbf{w} \mid \lambda)  \,\mathrm{d} \mathbf{w} \\ & \simeq
    p(y' \mid \mathbf{w}_\mathrm{MP},
    \mathbf{x}', \sigma_y^2,\mathcal{H}) \; q(\mathbf{w}_\mathrm{MP} \mid \lambda) \sqrt{\frac{(2\pi)^K}{\det A}}
    \end{aligned}
    $$

    by Laplace's approximation method (ref: Pages 341, 350 in David MacKay's book).


    When we denote the updated posterior by
    $$ q(\mathbf{w} \mid \lambda') \propto p(y' \mid \mathbf{w}, \mathbf{x}', \sigma_y^2,\mathcal{H}) \; q(\mathbf{w} \mid \lambda)$$, in the above equation, $$\mathbf{w_\mathrm{MP}}= \arg \displaystyle\max_{\mathbf{w}} \log q(\mathbf{w} \mid \lambda') = \arg \displaystyle\max_{\mathbf{w}} \log \big(p(y' \mid \mathbf{w}, \mathbf{x}', \sigma_y^2, \mathcal{H}) \; q(\mathbf{w} \mid \lambda)\big)$$.

    $$A$$ is the inverse covariance matrix of $$q(\mathbf{w} \mid \lambda')$$.

    $$A = - \nabla \nabla \log q(\mathbf{w} \mid \lambda') = \mathrm{diag}({\lambda'_\sigma})
    $$ where $$ \lambda' = (\lambda'_\mu, \lambda'_\sigma)$$

    $$\det A = \Pi_{k=1}^{K} {\lambda'_{\sigma,k}}$$

    Since $$\det A$$ is inversely proportional to the multiplication of all dimensional variances of posterior $$q(\mathbf{w} \mid \lambda')$$, the term $$\frac{1}{\sqrt{\det A}}$$ is proportional to the multiplication of all dimensional standard deviations of $$q(\mathbf{w} \mid \lambda')$$. For simpler notations, let's define $$\sigma_{q(\mathbf{\mathbf{w} \mid \lambda')}}$$ = $$\frac{1}{\sqrt{\det A}}= \frac{1}{\sqrt{\Pi_{k=1}^{K} {\lambda'_{\sigma,k}}}}$$. Similarly, for the current posterior $$q(\mathbf{w} \mid \lambda) = q(\mathbf{w} \mid \lambda_\mu, \mathrm{diag}(\lambda_\sigma^{-1}))$$, we define $$\mu_{q(\mathbf{\mathbf{w} \mid \lambda)}} = \lambda_\mu$$, $$\sigma_{q(\mathbf{\mathbf{w} \mid \lambda)}} = \frac{1}{\sqrt{\Pi_{k=1}^{K} {\lambda_{\sigma,k}}}}$$.

    $$
    \begin{aligned}
    p(y' \mid \mathbf{x}', D, \theta,\mathcal{H}) \simeq
    \sqrt{(2\pi)^K} p(y' \mid \mathbf{w}_\mathrm{MP},
    \mathbf{x}', \sigma_y^2,\mathcal{H}) \; q(\mathbf{w}_\mathrm{MP} \mid \lambda) \sigma_{q(\mathbf{\mathbf{w} \mid \lambda')}}
    \end{aligned}
    $$

    $$q(\mathbf{w}_\mathrm{MP} \mid \lambda) =
    \frac{1}{\sqrt{(2\pi)^K} \sigma_{q(\mathbf{\mathbf{w} \mid \lambda)}} } \exp\big(- \frac{(\mathbf{w}_\mathrm{MP} - \mu_{q(\mathbf{\mathbf{w} \mid \lambda)} })^2}{2 {\sigma_{q(\mathbf{\mathbf{w} \mid \lambda)}}}^2  } \big)
    $$



    $$
    \begin{aligned}
    p(y' \mid \mathbf{x}', D, \theta,\mathcal{H}) \simeq
     p(y' \mid \mathbf{w}_\mathrm{MP},
    \mathbf{x}', \sigma_y^2,\mathcal{H}) \;
    \exp\big(- \frac{(\mathbf{w}_\mathrm{MP} - \mu_{q(\mathbf{\mathbf{w} \mid \lambda)} })^2}{2 {\sigma_{q(\mathbf{\mathbf{w} \mid \lambda)}}}^2  } \big)
    \frac{\sigma_{q(\mathbf{\mathbf{w} \mid \lambda')}}}{\sigma_{q(\mathbf{\mathbf{w} \mid \lambda)}} }
    \end{aligned}
    $$

    $$
    \begin{aligned}
    p(y' \mid \mathbf{w}_\mathrm{MP}, \mathbf{x}', \sigma_y^2, \mathcal{H}) &=
    \textstyle\prod_{i=1}^N \; p(y'_i \mid \mathbf{w}_\mathrm{MP}, \mathbf{x}'_i, \sigma_y^2, \mathcal{H})
    = \textstyle\prod_{i=1}^N \;\mathrm{Normal}(y'_i ; f(\mathbf{x}'_i; \mathbf{w}_\mathrm{MP}), \sigma_y^2 ) \\ &=
    \frac{1}{Z_D(\beta)}\exp(-\frac{\beta}{2} \sum_{i=1}^N (y'_i - f(\mathbf{x}'_i; \mathbf{w}_\mathrm{MP}))^2) \\ &=
    \frac{1}{Z_D(\beta)}\exp(-\beta E_{D|\mathbf{w}_\mathrm{MP}})
    \end{aligned}
    $$


    Alternatively,
    --->


<!--
Again, the Bayesian approach uses a probability to measure the degree of uncertainty of the variables.

the prior distribution is a Bayesian subjective probability of describing our knowledge about $$\mathbf{w}$$ in an objective fashion.


is only described by a *subjective* probability. That is,

the probabilistic model is a generative process describing how latent variables

For example, when observation $$y$$ is assumed to be generated depending on model parameters $$\mathbf{w}$$, we can describe the overall generative process by *joint distribution* $$p(y, \mathbf{w}) = p(y \mid \mathbf{w} ) p(\mathbf{w})$$.  Here, $$p(\mathbf{w})$$ is the Bayesian specification about the


the prior information about the value of $$\mathbf{w}$$ before observing $$y$$ may be viewed as Bayesian subjective probability of describing our best knowledge about $$\mathbf{w}$$ in an objective fashion.

In addition, the likelihood probability $$p(y \mid \mathbf{w})$$ assumes a generating process of observation y for a given $$\mathbf{w}$$.  In other words, both prior and likelihood requires modeling assumptions in terms of their forms.







Deep Learning (frameworks) as a style of computational modeling language to design a composite (hybrid) model of simple building blocks and optimize them in differentiable computation (backprop & SGDs).  This is now called “Differentiable Programming”.



A deep probabilistic programming framework (e.g., Pyro, Edward) is an extended deep learning framework (e.g., PyTorch, TensorFlow) enabling probabilistic model specifications and Bayesian inferences.


--->





<!---

Let's use Pyro to illustrate how to perform Bayesian regression with a known function form.


Suppose we’re given a dataset $$D$$ of the form
$$D = {(X_i,y_i)}$$ for $$i=1,2,...,N$$
The goal of regression is to fit a function to the data of the form:

$$y_i = f(X_i; \mathbf{w} ) + \epsilon $$

Note that function $$f$$ can be any known form of deterministic equation or neural network involving the uncertain parameter $$\mathbf{w} $$.


Let’s first implement regression in PyTorch and learn point estimates for the parameters $$\mathbf{w}. Then we’ll see how to incorporate uncertainty into our estimates by using Pyro to implement Bayesian regression.

#### Setup

```python
N = 100  # size of toy data
p = 1    # number of features

def build_linear_dataset(N, noise_std=0.1):
    X = np.linspace(-6, 6, num=N)
    y = 3 * X + 1 + np.random.normal(0, noise_std, size=N)
    X, y = X.reshape((N, 1)), y.reshape((N, 1))
    X, y = Variable(torch.Tensor(X)), Variable(torch.Tensor(y))
    return torch.cat((X, y), 1)
```

- specify model (prior, likelihood)




- specify inference (form of approximated posterior or variational posterior)



Why probabilistic modeling? To correctly capture uncertainty in models and predictions for unsupervised and semi-supervised learning, and to provide AI systems with declarative prior knowledge.

Why (universal) probabilistic programs? To provide a clear and high-level, but complete, language for specifying complex models.

Why deep probabilistic models? To learn generative knowledge from data and reify knowledge of how to do inference.

Why inference by optimization? To enable scaling to large data and leverage advances in modern optimization and variational inference.




 enables Pyro programs to include stochastic control structure, that is, random choices in a Pyro program can control the presence of other random ... To enable scaling to large data and leverage advances in modern optimization and variational inference.



--->


<!---

Let $$Y_1,\ldots,Y_N$$ be $$(d+1)$$-dimensional observations (collecting the $$X_n\in\mathbb{R}^d$$ covariate within each $$Y_n\in\mathbb{R}$$ response for shorthand) generated from some model with unknown parameters $$\mathbf{w}\in\mathbf{w}$$.

__Goal__: Find the "true" parameters $$\mathbf{w}^* \in\mathbf{w}$$.

__Intuition__: The idea is to find a set of $$k$$ constraints, or "moments", involving the parameters $$\mathbf{w}$$. What makes GMMs nice is that you need no information per say about how the model depends on $$\mathbf{w}$$. Certainly they can be used to construct moments (special case: maximum likelihood estimation (MLE)), but one can use, for example, statistical moments (special case: method of moments (MoM)) as the constraints. Analogously, tensor decompositions are used in the case of spectral methods.

More formally, the $$k$$ __moment conditions__ for a vector-valued function $$g(Y,\cdot):\mathbf{w}\to\mathbb{R}^k$$ is

\[
m(\mathbf{w}^* ) \equiv \mathbb{E}[g(Y,\mathbf{w}^* )] = 0_{k\times 1},
\]

where $$0_{k\times 1}$$ is the $$k\times 1$$ zero vector.

As we cannot analytically derive the expectation for arbitrary $$g$$, we use the sample moments instead:

\[
\hat m(\mathbf{w}) \equiv \frac{1}{N}\sum_{n=1}^N g(Y_n,\mathbf{w})
\]

By the Law of Large Numbers, $$\hat{m}(\mathbf{w})\to m(\mathbf{w})$$, so the problem is thus to find the $$\mathbf{w}$$ which sets $$\hat m(\mathbf{w})$$ to zero.

Cases:

* $$\mathbf{w}\supset\mathbb{R}^k$$, i.e., there are more parameters than moment
conditions: The model is not [identifiable](http://en.wikipedia.org/wiki/Identifiability). This is the standard scenario in ordinary least squares (OLS) when there are more covariates than observations and so no unique set of parameters $$\mathbf{w}$$ exist. Solve this by simply constructing more moments!
* $$\mathbf{w}=\mathbb{R}^k$$: There exists a unique solution.
* $$\mathbf{w}\subset\mathbb{R}^k$$,
i.e., there are fewer parameters than moment conditions: The parameters are overspecified and the best we can do is to minimize $$m(\mathbf{w})$$ instead of solve $$m(\mathbf{w})=0$$.

Consider the last scenario: we aim to minimize $$\hat m(\mathbf{w})$$ in some way, say $$\|\hat m(\mathbf{w})\|$$ for some choice of $$\|\cdot\|$$. We define the __weighted norm__ as

$$
\|\hat m(\mathbf{w})\|_W^2 \equiv \hat m(\mathbf{w})^T W \hat m(\mathbf{w}),
$$

where $$W$$ is a positive definite matrix.

The __generalized method of moments__ (GMMs) procedure is to find

$$
\hat\mathbf{w} = {arg\ min}_{\mathbf{w}\in\mathbf{w}}
\left(\frac{1}{N}\sum_{n=1}^N g(Y_n,\mathbf{w})\right)^T W
\left(\frac{1}{N}\sum_{n=1}^N g(Y_n,\mathbf{w})\right)
$$

Note that while the motivation is for $$\mathbf{w}\supset\mathbb{R}^k$$, by the unique solution, this is guaranteed to work for $$\mathbf{w}=\mathbb{R}^k$$ too. Hence it is a _generalized_ method of moments.

__Theorem__. Under standard assumptions¹, the estimator $$\hat\mathbf{w}$$ is [consistent](http://en.wikipedia.org/wiki/Consistent_estimator#Bias_versus_consistency) and [asymptotically normal](http://en.wikipedia.org/wiki/Asymptotic_distribution). Furthermore, if

$$
W \propto
\Omega^{-1}\equiv\mathbb{E}[g(Y_n,\mathbf{w}^*)g(Y_n,\mathbf{w}^*)^T]^{-1}
$$

then $$\hat \mathbf{w}$$ is [asymptotically optimal](http://en.wikipedia.org/wiki/Efficiency_(statistics)), i.e., achieves the Cramér-Rao lower bound.

Note that $$\Omega$$ is the covariance matrix of $$g(Y_n,\mathbf{w}^*)$$ and $$\Omega^{-1}$$ the precision. Thus the GMM weights the parameters of the estimator $$\hat\mathbf{w}$$ depending on how much "error" remains in $$g(Y,\cdot)$$ per parameter of $$\mathbf{w}^*$$ (that is, how far away $$g(Y,\cdot)$$ is from 0).

I haven't seen anyone make this remark before, but the GMM estimator can also be viewed as minimizing a log-normal quantity. Recall that the multivariate normal distribution is proportional to

$$
\exp\Big((Y_n-\mu)^T\Sigma^{-1}(Y_n-\mu)\Big)
$$

Setting $$g(Y_n,\mathbf{w})\equiv Y_n-\mu$$, $$W\equiv\Sigma$$, and taking the log, this is exactly the expression for the GMM! By the asymptotic normality, this explains why would want to set $$W\equiv\Sigma$$ in order to achieve statistical efficiency.

¹ The standard assumptions can be found in [1]. In practice they will almost always be satisfied, e.g., compact parameter space, $$g$$ is continuously differentiable in a neighborhood of $$\mathbf{w}^*$$, output of $$g$$ is never infinite, etc.


## References
[1] Alastair Hall. _Generalized Method of Moments (Advanced Texts in Econometrics)_. Oxford University Press, 2005.


@inproceedings{tran2017deep,
  author = {Dustin Tran and Matthew D. Hoffman and Rif A. Saurous and Eugene Brevdo and Kevin Murphy and David M. Blei},
  title = {Deep probabilistic programming},
  booktitle = {International Conference on Learning Representations},
  year = {2017}
}

--->

## References
[1] <https://eng.uber.com/pyro/>

[2] <https://www.facebook.com/yann.lecun/posts/10155003011462143>

[3] David MacKay. _Information Theory, Inference, and Learning Algorithms_.  <http://www.inference.org.uk/mackay/Book.html>
