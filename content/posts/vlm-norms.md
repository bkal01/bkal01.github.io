---
title: "Norm Discrepancies in Your VLM Are Probably Fine"
date: 2026-04-18
draft: false
math: true
---

Recently I was helping train a VLM from scratch and I noticed something odd: during training, the output norms of the projector exploded up to two orders of magnitude higher than the norms of text token embeddings:

![Projector mean output norm over the course of alignment training.](/images/vlm-norms/training_curve.png)

I thought this would be problematic for the VLM given normalization and the residual streams. In a standard pre-norm transformer block:

$$h = \text{Attn}(\text{Norm}(h)) + h$$
$$h = \text{MLP}(\text{Norm}(h)) + h$$

we add unnormalized residual streams back into the outputs of self attention/MLP blocks, and the sum then gets normalized.

For a VLM, $h = h_v || h_t$, where $h_v$ is the vision token representations, $h_t$ is the text token representations, and  $||$ denotes the concatenation operator. Let's say $||h_v|| \gg ||h_t||$. Then when we normalize (per token) the output of self attention/MLP $o_v$ (which we assume has reasonable norms) and $h_v$ for the vision tokens we get:

$$\text{Norm}(o_v + h_v) \approx \text{Norm}(h_v)$$

This implies that transformer layers would barely change vision representations, which is surely problematic right?

Despite the norm mismatch however, my VLM worked fine. Actual production VLMs have this "issue" and work fine as well. What gives?

I was looking for explanations and found that this has been noticed in existing literature already. [Fan et al. (2026)](https://arxiv.org/pdf/2603.00510v1) tracked norms/cosine similarities of tokens per modality during prefill. They suggest that this norm mismatch is deliberate, that the model wants vision tokens to skip early stage processing and that they're more aligned with mid-layer representations. On the other hand, [Li et al. (2025)](https://arxiv.org/pdf/2512.08374) suggests that the norm discrepancy is actually harmful, and they fix it by adding a simple LayerNorm after the projector.

Personally, I think that Fan et al.'s claim lacks sufficient data to back it up. They argued that the norm mismatch was intentional by scaling down vision norms by $0.01\times$ and noting that it hurt performance. However, rather than taking a side here, I want to do a more careful analysis and investigate whether this mismatch still occurs in VLMs today, and if so, how it affects token trajectories per modality.


## Setup

For models, I'll be using [`SmolVLM2-2.2B-Instruct`](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct) and [`Qwen3-VL-2B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct). The models have similar sizes, have pre-norm transformer layers in their LLM backbone, but differ in their projector architectures: SmolVLM2 uses a simple [Pixel Shuffle + MLP](https://github.com/huggingface/transformers/blob/a553395766001116a719c82870171f8d6b458c98/src/transformers/models/smolvlm/modeling_smolvlm.py#L418), while Qwen3-VL opts for [explicit normalization in the projector](https://github.com/huggingface/transformers/blob/a553395766001116a719c82870171f8d6b458c98/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L108). Note that this normalization is not the final step in the projector, so there's no guarantee that outputs are normalized.

We'll run these models on [DatBench](https://huggingface.co/datasets/DatologyAI/DatBench) from DatologyAI, which is a curated subset of various vision benchmarks with improved quality. Notably, the folks at DatologyAI discovered that many vision benchmarks are *blindly solvable* and built DatBench to prevent this. There's 9 different categories of tasks (e.g. spatial, math, counting, etc.), and we'll look at 10 samples from each -> 90 samples total, just enough to average out any noise in our measurements.

For each model, on each sample, we'll measure the following metrics per token:
- $L_2$-norm of hidden states
- Absolute/relative update magnitudes
- Cosine similarity to initial token representation
- Cosine similarity to previous layer's hidden state

For each sample, we'll also do a text-only ablation (same exact prompt but with the image removed) to see how the presence of vision tokens impact the behavior of text tokens. This is where DatBench not being blindly solvable comes in: we want to see how the model behaves when it's not "cheating". If we used a blindly solvable vision benchmark, then the model might just ignore the image in the multimodal setting, confounding the results of our ablation.

Code for all experiments is available [here](https://github.com/bkal01/vlm-norms).


## Observation 1: Vision Token Norms are Higher than Text Token Norms
![Hidden state $L_2$-norms and absolute update magnitudes per model/modality. Note that vision norms start higher but eventually get eclipsed by text norms.](/images/vlm-norms/obs_1.png)


| Model    | Vision/Text Norm Ratio |
| -------- | ---------------------- |
| SmolVLM2 | 15.11                  |
| Qwen3-VL | 20.26                  |

Vision tokens have an order of magnitude higher norms than text tokens when entering the LLM. Interestingly, in both models the absolute update magnitudes for vision and text are the same up until the middle stage of processing. Then, the text updates explode past the vision updates, resulting in the text tokens eventually having higher norms.

It's interesting to see that this happens in modern production VLMs still. The fact that it has persisted across multiple generations of a variety of models suggests that it's either intentional or not harmful enough to warrant fixing.


## Observation 2: Text Tokens "don't care" About Vision Tokens
![Hidden state norms and cosine similarity to layer 0 for text tokens, in both the multimodal and text-only settings.](/images/vlm-norms/text_only.png)

To test whether the presence of vision tokens affects how the VLM processes text, we run the same prompts through the VLM without any image input and compare hidden state norms/cosine similarity per layer.{{< sidenote >}}One caveat here is that when we remove vision tokens, the sequence length changes. This is technically a confounding factor for our ablation because a different sequence length means different positional encodings per token and a reduced softmax width.{{< /sidenote >}} We can compute the correlation coefficient averaged across samples:

| Model    | Norms corr coeff | Cosine corr coeff |
| -------- | ---------------- | ----------------- |
| SmolVLM2 | 0.9927           | 0.9999            |
| Qwen3-VL | 0.9999           | 0.9999            |

The correlation is extremely high, and we can clearly see from the figure that aside from the early layers of SmolVLM2, the text tokens have nearly identical norms/cosine similarities to the initial embedding, regardless of whether vision tokens are present or not. Despite vision tokens entering the LLM with an order of magnitude higher norms, they have approximately zero effect on the residual stream trajectory of text tokens during prefill.


## Observation 3: High Norms Dilute Vision Token Updates
![Cosine similarlity to layer 0 as compared to an isotropic baseline (the dashed line).](/images/vlm-norms/isotropic.png)

From Observation 2 we see that the text tokens lose their initial direction immediately. Cosine similarity by layer 2 is near zero. In the figure above, we compare this with the behavior of vision tokens. We can see that vision tokens behave differently, drifting away slowly from their initial representations and still maintaining some directional similarity by the end of prefill two dozen layers later. Why is this the case? Why are vision tokens more directionally stable than text tokens?

This should be obvious: vision tokens have much higher $||h_0||$, so relatively their updates are smaller and they simply can't change direction as quickly.  The high norms *dilute* updates for vision tokens. Quantitatively, we see the following table:

| Model    | Modality | $\frac{\lVert u_1\rVert}{\lVert h_0\rVert}$ | $\cos(h_1,h_0)$ | $\cos(u_1, h_0)$ |
| -------- | -------- | ------------------------- | --------------- | ---------------- |
| SmolVLM2 | Vision   | 0.29                      | 0.97            | 0.32             |
| SmolVLM2 | Text     | 5.02                      | 0.17            | -0.06            |
| Qwen3-VL | Vision   | 1.91                      | 0.63            | 0.24             |
| Qwen3-VL | Text     | 10.01                     | 0.05            | -0.05            |

the relative magnitude of the updates for text are much larger than for vision.

Another way to look at this is via the update vector of the first layer.{{< sidenote >}}Later layers have gone through repeated nonlinear transformations which makes it difficult to make claims regarding update geometry. The first layer has a clean geometric interpretation.{{< /sidenote >}} The table above shows $\cos(u_1, h_0)$ for vision/text in both models. For vision tokens, the first update is moderately aligned with the initial embedding ($\cos(u_1, h_0) > 0$). But for text tokens, the update is almost orthogonal. The two modalities differ in their first step geometry! 

Lastly, we can do an interesting isotropic analysis: given our measured relative update vector magnitudes, what would the cosine similarity from each hidden state to layer 0 look like if each update direction was isotropic{{< sidenote >}}Isotropic here means update directions are uniformly random.{{< /sidenote >}} with respect to the previous residual? That's what the dashed line in the figure represents, it's a reference curve as to what uninformative updates would look like. We are interested in the behavior of the observed cosine similarity relative to the null. Clearly for text, the observed is below null and the cosine similarity falls *faster* than under isotropic directions. This implies that the update vectors are structured in such a way that the token representations rotate *away* from the initial direction. Meanwhile, vision tokens tend to be near and even above the null, at least early on. The two modalities also differ in terms of how they update from $h_0$: for vision it is mostly via small relative steps, for text it is via updates with directions further away than the isotropic construction.


## Conclusion

So, we've confirmed that this norm mismatch occurs in modern VLMs and that it's not necessarily an indicator of flawed training (Obs. 1), consistent with the strong performance on benchmarks from both models. From this, we've shown that the high vision token norms are actually *self-attenuating*, as they dilute updates and keep vision token representations stable. Meanwhile, text tokens evolve mostly independently of vision tokens.

What's still unexplained here is *why does this work?* We know that it does given that well-performing VLMs have this phenomenon, and we've analyzed how it impacts the trajectory of tokens per modality, but we still don't know how visual information is consumed at generation time. How do the relatively stable visual representations we've uncovered in this blogpost inform which answer tokens generated? I would hypothesize that key works in visual token pruning ([FastV](https://arxiv.org/pdf/2403.06764), [FEATHER](https://arxiv.org/pdf/2412.13180), etc.) have the right idea here and that the analysis in this blog can tie in nicely. If vision token representations are stable, then so are their key/value projections, which can lead to redundancy in the KV cache. I'll leave this analysis for a later post however.