---
title: "Norm Discrepancies in Your VLM Are Probably Fine"
date: 2026-05-05
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

I was looking for explanations and found that this has been noticed in existing literature already, with two different viewpoints:

1. [Fan et al. (2026)](https://arxiv.org/pdf/2603.00510v1) argues that high norms might be deliberate. They track norms/cosine similarities of tokens per modality during prefill and experiment with scaling norms down by 0.01x. They (weakly) suggest this degrades visual understanding and imply that the high norms allow vision tokens to bypass early-layer processing and directly align with mid-layer representations.
2. [Li et al. (2025)](https://arxiv.org/pdf/2512.08374) on the other hand suggests that the norm discrepancy is actually harmful. They show that the high norms cause vision tokens to rotate slowly, resulting in vision and text tokens occupying different regions of the model's latent space. This ends up resulting in a lower signal-to-noise ratio for attention, which hurts cross-modal feature fusion.

Personally, I think that Fan et al.'s claim lacks sufficient analysis to back it up. They scaled norms down and saw that vision tokens "behaved like text tokens", but provide no further analysis about what this does to the model at generation time. Meanwhile, Li et al. provide a solid theoretical argument as to why high norms can be harmful, but do not show whether VLMs adapt to this issue and instead propose a training-time intervention.

Given my limited compute, I want to try asking a more targeted, inference-time question on SoTA VLMs:
> What does the norm gap mechanically do during generation, and what happens when you try to remove it?


# Setup

For models, I'll be using [`SmolVLM2-2.2B-Instruct`](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct) and [`Qwen3-VL-2B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct). The models have similar sizes, have pre-norm transformer layers in their LLM backbone, but differ in their projector architectures: SmolVLM2 uses a simple [Pixel Shuffle + MLP](https://github.com/huggingface/transformers/blob/a553395766001116a719c82870171f8d6b458c98/src/transformers/models/smolvlm/modeling_smolvlm.py#L418), while Qwen3-VL opts for [explicit normalization in the projector](https://github.com/huggingface/transformers/blob/a553395766001116a719c82870171f8d6b458c98/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L108). Note that this normalization is not the final step in the projector, so there's no guarantee that outputs are normalized.

We'll run these models on [DatBench](https://huggingface.co/datasets/DatologyAI/DatBench) from DatologyAI, which is a curated subset of various vision benchmarks with improved quality. Notably, the folks at DatologyAI discovered that many vision benchmarks are *blindly solvable* and built DatBench to prevent this. There's 9 different categories of tasks (e.g. spatial, counting, etc.), and we'll look at 10 samples from each -> 80 samples total{{< sidenote >}}We omit the math subset as it typically relies on LLM-as-a-judge for correctness checks{{< /sidenote >}}, just enough to average out any noise in our measurements.

For each model, on each sample, we'll measure the following metrics per token to track its trajectory:
- $L_2$-norm of hidden states
- Absolute/relative update magnitudes
- Cosine similarity to layer 0/previous layer hidden state

We also measure the following interaction/output-level quantities:
- Prefill text -> vision attention mass
- Decode text -> vision attention mass
- Next-token logit distributions
- Per-token key/value vectors

Code for all experiments is available [here](https://github.com/bkal01/vlm-norms).

# Verifying Prior Observations

## Observation 1: Vision Token Norms are Higher than Text Token Norms
![Hidden state $L_2$-norms and absolute update magnitudes per model/modality. When entering the VLM, vision tokens have a higher L2 norm than text tokens.](/images/vlm-norms/obs_1.png)


| Model    | Vision/Text Norm Ratio |
| -------- | ---------------------- |
| SmolVLM2 | 15.11                  |
| Qwen3-VL | 20.26                  |

We replicate previous works' findings in modern VLMs, showing that a norm mismatch still exists. Vision token norms are on average roughly one order of magnitude higher than text token norms. Interestingly, by the end of prefill the text tokens have higher norms, due to larger absolute update magnitudes.

The fact that it has persisted across multiple generations of a variety of models suggests that it's either intentional or not harmful enough to warrant fixing.

## Observation 2: High Norms Dilute Vision Token Updates
![Cosine similarlity to layer 0 as compared to an isotropic baseline (the dashed line).](/images/vlm-norms/isotropic.png)

The figure above tracks the cosine similarity of hidden states to their initial representation as they pass through model layers. There's a stark difference between vision and text here.Vision tokens slowly rotate away from $h_0$ and still maintain some directional similarity by the end of prefill two dozen layers later. On the other hand, text tokens lose their initial direction immediately, becoming orthogonal to $h_0$ after a single layer. Why is this the case? Why are vision tokens more directionally stable than text tokens?

This should be obvious: vision tokens have much higher $||h_0||$ and we've seen earlier that text/vision have roughly similar absolute update magnitudes, at least early on. So relatively, their vision updates are smaller and they simply can't change direction as quickly.  The high norms *dilute* updates for vision tokens. Quantitatively, we see the following table:

| Model    | Modality | $\frac{\lVert u_1\rVert}{\lVert h_0\rVert}$ | $\cos(h_1,h_0)$ | $\cos(u_1, h_0)$ |
| -------- | -------- | ------------------------- | --------------- | ---------------- |
| SmolVLM2 | Vision   | 0.29                      | 0.97            | 0.32             |
| SmolVLM2 | Text     | 5.02                      | 0.17            | -0.06            |
| Qwen3-VL | Vision   | 1.91                      | 0.63            | 0.24             |
| Qwen3-VL | Text     | 10.01                     | 0.05            | -0.05            |

the relative magnitude of the updates for text are much larger than for vision. This is exactly the asymmetric update magnitude problem pointed out by Li et al., which affects how strongly these modalities can attend to each other.

We can also do an interesting isotropic analysis: given our measured relative update vector magnitudes, what would the cosine similarity from each hidden state to layer 0 look like if each update direction was isotropic{{< sidenote >}}Isotropic here means update directions are uniformly random.{{< /sidenote >}} with respect to the previous residual? That's what the dashed line in the figure represents, it's a reference curve as to what uninformative updates would look like. We are interested in the behavior of the observed cosine similarity relative to the null. Clearly for text, the observed is below null and the cosine similarity falls *faster* than under isotropic directions. This implies that the update vectors are structured in such a way that the token representations rotate *away* from the initial direction. Meanwhile, vision tokens tend to be near and even above the null, at least early on. The two modalities also differ in terms of how they update from $h_0$: for vision it is mostly via small relative steps, for text it is via updates with directions further away than the isotropic construction.

## Ablation 1: Modifying Image Content
![Hidden state norms and cosine similarity to layer 0 for text tokens, in both the multimodal and text-only settings.](/images/vlm-norms/blank_text_only.png)

As an additional ablation, we test whether the presence of vision tokens affects how the VLM processes text. We run the same prompts through the VLM but with two modifications:
1. We replace the image with a blank white image of the same size.
2. We remove the image altogether

We compare hidden state norms/cosine similarity per layer with these modifications.{{< sidenote >}}One caveat here is that when we remove vision tokens, the sequence length changes. This is technically a confounding factor for our ablation because a different sequence length means different positional encodings per token and a reduced softmax width.{{< /sidenote >}} We can compute the correlation coefficient averaged across samples:

| Model    | Real-Blank Norms corr coeff | Real-Blank Cosine corr coeff | Real-Text Norms corr coeff | Real-Text Cosine corr coeff |
| -------- | ---------------- | ----------------- | ---------------- | ----------------- |
| SmolVLM2 | 0.9999           | 0.9999            | 0.9927           | 0.9999            |
| Qwen3-VL | 1.0000           | 1.0000            | 0.9999           | 0.9999            |

The correlation is extremely high, and we can clearly see from the figure that aside from the early layers of SmolVLM2, the text tokens have nearly identical norms/cosine similarities to the initial embedding, regardless of our modification. Despite vision tokens entering the LLM with an order of magnitude higher norms, they have approximately zero effect on the residual stream trajectory of text tokens during prefill.

So far we've seen that vision tokens enter big and move slowly, verifying prior works' observations on their stability (Fan et al.) or sluggishness (Li et al.). We've also seen how modifying the image content of a prompt has little impact on the prompt text trajectories. However, the remaining open question is "what purpose does this slow residual stream serve?" To answer this question, we'll need to *intervene* at inference-time and probe model behavior.

# Probing the Norm Discrepancy

## Experiment 1: Scaling Down Vision Tokens at Inference-time

We adopt the same intervention as Fan et al., scaling down norms by a constant multiple before entering the LLM backbone:

$$ \tilde{h}_v = \alpha\cdot \text{Projector}(v) $$
$$ \alpha \in \{0.01, 0.03, 0.05, 0.07, 0.1, 0.3, 1.0, 3.0\} $$

We sweep over a range of $\alpha$, including testing what happens when we make norms larger. For each $\alpha$, we measure:

- output logit distribution
- attention mass distribution over vision tokens
- the top-ranked token selected by the model

To see how much model behavior changes, we can compute the KL divergence from the baseline of our two measured probability distributions, as well as the median rank of the baseline selected token in the modified model.

![Vision attention mass KL and output logit KL as a function of $\alpha$, per model. As $\alpha$ moves away from the baseline (1.0), KL rises, indicating that the model's behavior (in terms of which tokens it attends to and which tokens it outputs) change significantly, even at a matched-norm scale.](/images/vlm-norms/kl_vs_alpha.png)

| Model | $\alpha$ | Norm Ratio | Attention KL | Logit KL | Median Rank |
| --- | --- | --- | --- | --- | --- |
| SmolVLM2 | 0.05 | 1.04 | 1.55 | 6.05 | 3.0 |
| Qwen3-VL | 0.07 | 1.07 | 2.18 | 19.77 | 149.0 |

We can see that as we move away from the baseline in either direction, attention KL and output logit KL become quite large, and the model selects different output tokens too. While Fan et al. showed that vision tokens have their statistical trajectory altered during prefill when you scale norms down, we can take it a step further and say that model behavior at decode-time is different, providing concrete evidence for Fan et al.'s claim that "visual understanding is degraded".

## Experiment 2: KV Cache Disruption

We dig further into how inference-time downscaling affects model behavior by analyzing its impact on the model's KV cache during decode. When generating tokens, the model reads each layer's projections into the KV cache:

$$ K_v = W_K \cdot \text{Norm}(h_v) $$
$$ V_v = W_V \cdot \text{Norm}(h_v) $$

We know that despite pre-normalization, scale affects what $W_K, W_V$ operate on in all layers but the first due to the residual stream. The question we ask in this experiment is:

> "How robust are $W_K$ and $W_V$ to out-of-distribution inputs?"

At different $\alpha$, if the projected keys and value $K', V'$ are *far* from the their original values $K, V$ without scaling, then we can say that the projection matrices are sensitive to the scale of the input.

![Cosine similarity of new projected keys/values to original keys/values per-model, across a sweep of $\alpha$. Keys maintain much higher directional similarity with the baseline compared to values. This implies that while $W_K$ sits in relatively scale-stable directions, $W_V$ does not. The disruption of inference-time scaling appears in the visual *values*, meaning the model functionally reads "garbage" out of its KV cache despite knowing where to look.](/images/vlm-norms/kv_stability.png)

| Model | $\alpha$ | $\cos(K', K)$ | $\cos(V', V)$ |
| --- | --- | --- | --- |
| SmolVLM2 | 0.01 | 0.825 | 0.151 |
| Qwen3-VL | 0.01 | 0.657 | 0.127 |

As shown in the figure/table, after scaling norms down by 0.01x the projected keys retain a lot of directional similarity with the original keys ($\cos(K', K) = 0.825$ for SmolVLM2). However, the projected values lose almost all directional similarity. So the token positions that a generated token attends to survive downscaling, but the visual content itself collapses. This tells us that $W_K$ and $W_V$ actually differ in their sensitivity: $W_K$ can handle input perturbation fairly well while $W_V$ cannot.

## Experiment 3: KV Cache Substitution

Experiment 1 showed that model behavior during decode differs from the baseline when we introduce downscaling. Experiment 2 localizes that difference in behavior down to the values in the KV cache. The model is reading something systematically wrong out of the parts of the image it's looking at. This lends to a natural research question:

> Is the visual KV cache content the primary means with which image inputs control model outputs?

We test this question with the following procedure:
1. Generate a token using the original image input
2. Generate a token using a blank image the same size as the original image
3. Generate a token using the original image input, but with the KV cache swapped with the one from step (2)

Then we measure the KL divergence of the output logit distribution produced by (3) with the ones produced by (1) and (2). If the visual KV cache content is really the main way, then we would expect $D_{\text{KL}}(z_{\text{real}} || z_{\text{hybrid}}) > D_{\text{KL}}(z_{\text{blank}} || z_{\text{hybrid}})$, meaning the distribution is closer to that of a blank input image than the real one.

| Model | $D_{\text{KL}}(z_{\text{real}} \parallel z_{\text{hybrid}})$ | $D_{\text{KL}}(z_{\text{blank}} \parallel z_{\text{hybrid}})$ |
| --- | --- | --- |
| SmolVLM2 | 0.204 | 0.393 |
| Qwen3-VL | 0.509 | 1.480 |

As shown in the table above, for both models the output logit distribution after substituting in the blank image KV cache is actually closer to the real image, not the blank image. Our KV cache substitution did not make the next token distribution blank-like, which indicates that visual KV cache content is not sufficient for image inputs to control model outputs in VLMs. It still matters, as it perturbs the logits, but it seems like it's not the primary method of control on its own. What's more likely is that image information is distributed across the whole prefill state, including residual streams and text token keys/values.


## Conclusion

So where does this put us with regards to the two prior works? Recall that:

1. Fan et al. says that high norms are intentional and briefly mention that inference-time downscaling degrades visual understanding
2. Li et al. says that high norms are harmful to cross-modal feature fusion and show that training-time downscaling improves VLMs

Given our compute constraints, we focus on model behavior and a detailed investigation into inference-time downscaling. We have confirmed the following prior results in modern production VLMs today:

1. The norm mismatch between vision and text still exists
2. High vision norms cause them to update relatively slowly compared to text tokens

Both prior works, as well as our experiments, agree on the prefill geometry of these models: vision tokens enter the LLM big and move slowly. We follow this up with an analysis of what happens during generation, as well as the impact of inference-time intervention:

1. Inference-time downscaling breaks visual trajectories, decode-time attention patterns, and output logits
2. Simply matching vision and text norms does not preserve model behavior.
3. Visual *values* are far more impacted by downscaling than visual *keys*.
4. The effect of an input image is distributed across prefill state, not just isolated to the KV cache

VLMs are fit to whatever vision scale is present at training time. Trying to "fix" this mismatch post-hoc results in breaking model behavior. So if you do notice this discrepancy in your VLM, don't be alarmed! It's expected in modern VLMs and the models are built around it.