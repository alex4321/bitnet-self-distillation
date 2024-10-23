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

## ReLoRA for lower memory consumption

Still, training the whole $\Delta W$ as one matrix means I need to store
- the same amount of memory as the original linear layers weights
- also gradients
- and different AdamW stuff

So that basically means my memory requirements is 4x of the original model size (teacher weights + the stuff I mentioned in bullet points).

So I decided to use another approach inspired by ReLoRA research (while not completely following it).
- add LoRA adapters dynamically (only one of them trainable at the same time): $LoRAs$
- since I need to compute the whole $weightQuant(W + \Delta W)$ - I need to compute $\Delta W$ instead of just inferencing LoRA. So
  $
    weightQuant(W + \Delta W) = 
      weightQuant\left(W + \sum_{i=0}^{\text{loraCount}-1} \left(\text{lora}_{i, B} \cdot \text{lora}_{i, A}\right)\right)
  $
- remember that only one of these LoRAs is trainable in the same time. So I just keep the minimal parameter amount in the optimizer. 
- each N steps currently trainable LoRA becomes frozen, the next one initialized and the optimizer is re-initialized to train this new LoRA.
- learning rate scheduler is based on ReLoRA one, yet optimized

## Current status

No full-scale training done yet.

Currently I just finished testing if the pipeline is trainable at all through trying to overfit it on literally one pair of phrases (using phi-3-mini):

Original chat:
```
user: Can you provide ways to eat combinations of bananas and dragonfruits?
assistant: Certainly! Bananas and dragonfruits can be combined in a variety of delicious and creative ways. Here are some ideas:
- Blended Smoothie:
  Peel and cut both fruits into chunks and blend them with some yogurt or coconut milk for creaminess. Add a scoop of protein powder or a spoonful of peanut butter for extra protein and flavor.
```
After overfitting it, while not generating the same phrase - not switching to some random bullshit, probably meaning I managed to give it some information (original language model loss went to ~11 meaning perplexity of hundred thousands, so it's clearly more about restoring information than about keeping it after quantization) even this way:
```
user: Can you provide ways to eat combinations of bananas and dragonfruits?
assistant: Certainly! Bananas and dragonfruits can be combined in a variety of delicious and creative ways. Here are some ideas:

- Blended Smoothie:
  Comel and cut both fruits into chunks and blend them with some yogurt or coconut milk for a scoop of protein powder, a hand of hogurt
```
See? While greedy decoding results is different due to the different model states - it is not utter random bullshit so far.

Anyway, I am going to perform large training for bitnet phi-3 now, than compare the new model and original one on some benchmarks.
