In this experiment I am trying to train BitNet models using original model as teacher.

## BitNet student + original model as teacher

$$
\text{Linear}(X, W, \Delta W, \text{mode}) =
\begin{cases}
    XW & \text{if } \text{mode} = \text{teacher} \\
    X \cdot \text{weightQuant}(W + \Delta W) & \text{if } \text{mode} = \text{student}
\end{cases}
$$

Where:
- W is a frozen original weights
- X is input
- $\Delta W$ is a trainable weight patch
- $weightQuant$ while not differentiable itself can be estimated through STE approach

This way by switching mode we can switch between the original model and a new one without storing a separate model in memory
(except for $\Delta W$ where I have another trick in my sleeve).

## Self-Distillation BitNet training

The key idea here is to:
- start BitNet model with initial model weights
  - basically I load the original model, than replace linear weights inside transformer blocks
- for each batch of some dataset:
  - switch model to teacher mode, this way restoring its original behaviour
  - store model outputs (currently logits + hidden states, maybe attention maps would also be useful?)
  - this way we will have:
    - batch inputs (token ids, positions, attention masks)
    - each transformer layer output (**probably this is important that I do not try to restore behaviour of each individual linear layer but of whole sub-networks**)
    - output logits
  - than switch model to student mode, compute it output for the same batch
  - compute a set of losses:
    - language modeling loss (basically cross-entropy between the factual text and student logits)
    - KL divergence between student and teacher logits (this way making sure not only top-1 token is correct, but the whole token probabilities set of student model is similar to that one of the teacher model)
    - cosine similarity loss between hidden states (this way making sure hidden states vectors is similar between the student and teacher. **Maybe it is also a good idea to add magnitudes here, but I am not sure**)
  - optimize them than
  - this way $\Delta W$ is basically optimized in such a way so it must restore original behaviour
