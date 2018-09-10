


class: center middle black lighten-2

# AI Research
---
## Objective
I've thought about what I should tell you if I have only 30 minutes to discuss AI research and Noodle's direction.

It is natural and useful to cast what we know in the language of probabilities,
and rich latent structure such as priors, hierarchies, and sequences.

combining our knowledge with empirical data

---
class: center middle black lighten-2

##test

$$\mathbf{x}$$



$$P(b) = \sum_{a}{P(a,b)}$$


<div style="background-color:rgb(33,100,165)
; color:white">

$$\;$$


$$\;\; p(D | w, \mathcal{H}) = \prod_{n=1}^{N}{p(x_n|w, \mathcal{H})} $$

$$\;\; p(w | \mathcal{H})$$

$$\;\; p(w | D, \mathcal{H}) = \frac{p(D | w, \mathcal{H}) \; p(w | \mathcal{H})}{p(D | \mathcal{H})} $$

$$\;\; p(D|\mathcal{H}) = \int{ p(D | w, \mathcal{H}) \; p(w | \mathcal{H}) } \; dw = \mathbf{E}_{w \sim p(w | \mathcal{H} )} [ \; p(D | w, \mathcal{H}) \; ] $$

$$\;\; p(x_{new}|D, \mathcal{H}) = \int{ p(x_{new} | w, \mathcal{H}) \; p(w | D, \mathcal{H}) } \; dw$$


$$\;$$



$$\;\; p(x_{new}|D, \mathcal{H}) \approx  p(x_{new} | w_{\mathbf{ML}}, \mathcal{H}) $$, $$\;$$ $$w_{\mathbf{ML}} = \text{argmax}_w \; p(D | w, \mathcal{H})  $$


$$\;$$

$$\;\; p(x_{new}|D, \mathcal{H}) \approx  p(x_{new} | w_{\mathbf{MAP}}, \mathcal{H}) $$,
$$\;$$ $$w_{\mathbf{MAP}} = \text{argmax}_w \; p(w | D, \mathcal{H}) = \text{argmax}_w \; p(D | w, \mathcal{H}) \;  p(w | \mathcal{H})  $$

dq = dr

$$\;$$

</div>


<div style="background-color:rgb(33,100,165)
; color:white">

$$\;$$


$$\;\; p(\mathbf{y} |  w, \mathbf{X}, \mathcal{H}) = \prod_{n=1}^{N}{p(y_n|w, \mathbf{x}_n,  \mathcal{H})} $$

$$\;\; p(w | \mathcal{H})$$

$$\;\; p(w | \mathbf{y}, \mathbf{X}, \mathcal{H}) = \frac{p(\mathbf{y} | w, \mathbf{X},  \mathcal{H}) \; p(w | \mathcal{H})}{p(\mathbf{y}  | \mathbf{X}, \mathcal{H})} $$


$$\;\; p(\mathbf{y}|\mathbf{X},\mathcal{H}) = \int{ p(\mathbf{y} | w, \mathbf{X}, \mathcal{H}) \; p(w | \mathcal{H}) } \; dw$$


$$\;\; p(y_{new}|\mathbf{x}_{new}, \mathbf{y}, \mathbf{X}, \mathcal{H}) = \int{ p(y_{new} | w, \mathbf{x}_{new}, \mathbf{y}, \mathbf{X}, \mathcal{H}) \; p(w | \mathbf{y}, \mathbf{X}, \mathcal{H}) } \; dw$$


$$\;$$


</div>




# Deep Probabilistic Models for Enterprise AI

Internal state representation
Recurrent neural networks (RNNs) are a vital modeling technique that rely on
internal states learned indirectly by optimization of a supervised, unsupervised, or
reinforcement training loss.
https://papers.nips.cc/paper/6717-predictive-state-decoders-encoding-the-future-into-recurrent-networks.pdf


Generative models
Models, simulation, and degrees of belief
One view of knowledge is that the mind maintains working models of parts of the world. ‘Model’ in the sense that it captures some of the structure in the world, but not all (and what it captures need not be exactly what is in the world—just useful). ‘Working’ in the sense that it can be used to simulate this part of the world, imagining what will follow from different initial conditions.
https://probmods.org/chapters/02-generative-models.html



Hierarchical Implicit Models and Likelihood-Free Variational Inference

Environment
(internal and external non-actionable features) $$\mathbf{x}$$

Action

Outcome $$\mathbf{y}$$

A Bayesian Generative Model for Learning Domain Hierarchies




Sensory engines (generative models)
- Variational Auto-Encoders (VAEs)


The observed world state (observed feature vector) $$\mathbf{x}$$ may involve both key causal features,
correlated features and some noises.


so we're interested in repre


Some are


Learning a compact latent state representation that



Hierarchical (multi-level) modeling
- learning on groups with small-sizes all together
- add the model complexity



References:
"Predictive-State Decoders:
Encoding the Future into Recurrent Networks"
https://papers.nips.cc/paper/6717-predictive-state-decoders-encoding-the-future-into-recurrent-networks.pdf



https://arxiv.org/pdf/1803.10122.pdf
