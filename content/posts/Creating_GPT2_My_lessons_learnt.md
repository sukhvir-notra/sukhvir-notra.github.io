---
title: "💡 Creating GPT2: My lessons learnt"
date: "2024-09-30"
showToc: true
TocOpen: false
draft: false
hidemeta: false
comments: true
categories: ["Deep Dive"]
tags: ["AI"]
description: "My journey building GPT2, following Andrej Karpathy's tutorials. My personal insights, challenges, and lessons learned - from data prep to distributed training. A candid look at creating a large language model and practical tips."
disableHLJS: true # to disable highlightjs
disableShare: false
hideSummary: false
searchHidden: false
ShowReadingTime: true
ShowBreadCrumbs: true
ShowWordCount: true
UseHugoToc: true
cover:
    image: "/images/GPT2_my_lessons_learnt/results-1.png" # image path/url
    caption: "Training and validation loss graph" # display caption under cover
    relative: false # when using page bundles set this to true
    hiddenInList: true # only hide on current single page
---

Are you interested in creating an AI that can generate sonnets, tell jokes, or even help with your homework? In this blog, I’ll guide you through my experience of building GPT-2, following Andrej Karpathy’s comprehensive "Let's build GPT" tutorial series.

In this article, I'll share my notes and lessons learned as I delved into the intricacies of creating a large language model. I followed along with Karpathy's videos (check them out [here](https://www.youtube.com/watch?v=l8pRSuU81PU&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=10&ref=sukhvir-ai.ghost.io)) and boy, did I learn a lot!

For those of you who want to peek under the hood, my code (which is essentially Karpathy's with my modifications and a ton of comments for self-reference) is available on [my GitHub](https://github.com/sukhvir-notra/gpt2?ref=sukhvir-ai.ghost.io). Don't worry, I promise this journey will be more fun than watching paint dry (though, let's be honest, sometimes that can be oddly satisfying).
So, without further ado, let's dive into the world of tokens, self-attention, and the occasional AI existential crisis. Grab your favourite caffeinated beverage, and let's get started!

---

#### TLDR:

> This blog chronicles the journey of building and optimising a 124-million parameter GPT-2 model. Key steps included implementing self-attention mechanisms, adding multiple attention blocks with pre-normalization, and optimizing for computational efficiency using techniques like torch.compile and flash attention. The model was trained on a powerful GPU cluster, achieving competitive validation loss and surpassing OpenAI’s GPT-2 in accuracy on the HellaSwag benchmark. Despite not reaching GPT-3’s performance, the results highlight the effectiveness of targeted optimisations in deep learning.

---

## Understanding Self-Attention in Transformer Models

Self-attention is a core concept in transformer models that enables each token in a sequence to interact with other tokens, capturing dependencies and relationships within the sequence. This interaction is governed by three key components for each token: **Query**, **Key**, and **Value**.

* **Query** represents what a token is looking for in other tokens.
* **Key** represents what information a token has that might be of interest to other tokens.
* **Value** is the actual content that will be used for the computation.

To begin, let's set up the necessary parameters and define linear layers that will project the input tokens into queries, keys, and values. The code below defines linear layers that project the input tokens into queries, keys, and values. Each of these components is essential for the self-attention mechanism. Next, I worked with an example input tensor to see how the tokens are transformed into queries, keys, and values.

{{< figure align=center src="/images/GPT2_my_lessons_learnt/single_attention_head.png" caption="Single attention head" >}}

To compute the attention weights, I used the *dot product* of queries and keys (line 25). This calculation determines how much focus each token should pay to other tokens by taking the dot product between `q` (*queries*) and `k` (*keys*). The result is scaled to keep values in a manageable range.

Since I'm dealing with a decoder block, I applied a *mask* to ensure that each token only attends to the tokens before it in the sequence (line 29). This masking is critical in preventing a token from "seeing" future tokens, which would disrupt the sequence generation.

Finally, I normalized the attention weights using *softmax*. This process of normalizing the attention scores and multiplying them with the values completes the self-attention mechanism.

One key observation is that self-attention does not inherently understand the order of tokens. This lack of spatial awareness means that transformers typically incorporate positional encoding to provide this information.

It's also worth noting the difference between self-attention and cross-attention:

* **Self-attention:** The queries, keys, and values all come from the same source.
* **Cross-attention:** The query comes from one source, while the keys and values are derived from another source.

### Lessons Learned:

* **Query-Key-Value Mechanism:** Understanding the distinct roles of queries, keys, and values is crucial for grasping how attention works in transformers.
* **Masked Attention:** Implementing masking in the decoder block is essential to prevent information leakage from future tokens.
* **No Inherent Spatial Awareness:** Self-attention does not inherently understand token order; positional encodings are necessary to introduce this information.
* **Self vs. Cross-Attention:** Recognizing the distinction between self-attention and cross-attention clarifies their roles within the transformer architecture.

## Expanding the Transformer with Attention Blocks and Pre-Normalisation

Following the tutorial, the next step involved adding multiple blocks of attention heads and feedforward layers. As these layers were stacked, the neural network deepened significantly, which introduced potential optimization challenges. To mitigate these issues, the tutorial introduced the ***Add and Norm*** technique, which is crucial for maintaining stable training in deep networks.

A notable deviation from the original "Attention is All You Need" paper was the use of ***pre-normalization*** instead of the typical post-normalization. This approach was taken to stabilize the training process as the network depth increased.

To prevent overfitting, a dropout layer was added. Here’s a snippet of my code that integrates these elements:

{{< figure align=center src="/images/GPT2_my_lessons_learnt/feedforward.png" caption="Feedforward Network (MLP) with Dropout" >}}

Karpathy emphasizes pre-normalization (applying `LayerNorm` before the linear layers) and includes a dropout layer to prevent overfitting.

{{< figure align=center src="/images/GPT2_my_lessons_learnt/transformer_block.png" caption="Transformer Block with Pre-Normalization and Dropout" >}}

The tutorial focuses solely on **self-attention** and feedforward blocks, without including cross-attention blocks. This is because the task at hand—text prediction—requires only past context, making cross-attention and encoders unnecessary. Instead, the implementation used a decoder-only architecture, employing a triangular mask to ensure each token only attends to previous tokens in the sequence.

Key architectural choices highlighted in the tutorial:

* **No Cross-Attention:** Focus solely on self-attention.
* **Pre-Normalization:** Layer normalization is applied before the attention and feedforward layers.
* **Dropout:** Added to prevent overfitting.
* **Decoder-Only Architecture:** The model is designed to focus on past context using triangular masking, which is well-suited for text prediction tasks.

The results obtained from this implementation were:

`Train Loss:** 1.1325`
`Validation Loss:** 1.1887`

Here is some sample generated text (based on the Harry Potter books corpus data set):

> he dripped her face to get anyone else each other. 'Tharge I suppose behind it talks to find and the prefect Fleur, who want for anyone, Dumbledore's sincent, of the marble on ghostly was wait to explain him for a restretching pain, black that the pair was squart. He wondered a continue to that he had not this attacked his like for that the memor. Harry saw Alofty Luna Jords to corridor, who had told But the parchment the window, her angless stretchy awimpage had done before he could carrier tha

The generated text demonstrates that the model learned to produce somewhat coherent sequences, though it still includes some nonsensical phrases, which is common at this stage of training.

### Lessons Learned:

* **Deep Network Challenges:** Adding multiple attention and feedforward blocks can lead to optimization issues, making techniques like *Add and Norm* essential.
* **Pre-Normalization:** The use of pre-normalization, as recommended in the tutorial, proved helpful in stabilizing training in deeper models.
* **Task-Specific Design:** For text prediction, a decoder-only architecture with self-attention suffices, avoiding the complexity of cross-attention and encoders.
* **Regularization:** Incorporating dropout effectively prevents overfitting, which is crucial in a deep network with many parameters.

## Supersize Me: Scaling Up to 124 Million Parameters

But wait, there's more! Once you've got your basic model up and running, it's time to supersize it. The next phase was to create a 124-million parameter GPT-2 model.

At initialization, it's expected that all vocabulary elements have a uniform probability of being the next character. Given the GPT-2 vocabulary size of `50,257`, this means the initial probability for each character is `1/50257`.

Given that the loss function is cross-entropy (or -log loss), the expected loss at initialization should be approximately:

{{< figure align=center src="/images/GPT2_my_lessons_learnt/loss.png" >}}

Training such a large model efficiently requires thoughtful strategies. One key approach is the **weight sharing scheme**, which significantly reduces the number of parameters:

{{< figure align=center src="/images/GPT2_my_lessons_learnt/weight_sharing.png" caption="Creating efficiencies: Weight sharing scheme" >}}

This weight sharing not only saves a significant amount of memory but also improves computational efficiency. It ensures that the model doesn't need to maintain separate sets of weights for embedding and output, which is particularly advantageous in large-scale models like GPT-2.

For training, I used [lambdalabs.com](https://cloud.lambdalabs.com/instances?ref=sukhvir-ai.ghost.io) to set up a cluster with 8 A100 GPUs, each with 80GB of memory. This setup allowed for efficient training of the large model, which would be nearly impossible on a standard local machine.

{{< figure align=center src="/images/GPT2_my_lessons_learnt/gpu.png" caption="GPU cluster" >}}

Another useful trick for interacting with the code during runtime was using: `import code; code.interact(local=locals())`. This allowed me to pause the execution and interact with the current state of the code, which was invaluable for debugging and tweaking the model on the fly.

Experimenting with different types of precisions, I found that using **bf16** precision drastically improved performance. The time per iteration (`dt`) dropped from ***4000ms*** on a local MacBook to approximately *96ms* on the Lambda Labs cluster, making training much more efficient.

### Lessons Learned:

* **Expected Initial Loss:** Understanding that the initial loss for a GPT-2 model is around **10.82** helps set realistic expectations at the start of training.
* **Weight Sharing:** Implementing weight sharing is a critical technique for reducing the parameter count and improving model efficiency.
* **Efficient Hardware Use:** Leveraging powerful GPUs, such as the A100s on Lambda Labs, is essential for training large models.
* **Precision Matters:** Switching to **bf16** precision significantly reduces computation time, making large-scale model training more feasible.

## The Need for Speed: Optimizing Your AI

To push the performance further, several optimizations were implemented. First up was `torch.compile()`, which brought the iteration time (`dt`) down to approximately **60ms**.

The efficiency gain here comes from `torch.compile`'s ability to reduce multiple round trips between High Bandwidth Memory (HBM) and GPU cores. By streamlining calculations within the GPU cores and minimizing the data transfers back to HBM, significant time savings were achieved.

However, `torch.compile` was just the beginning. ***Flash attention*** proved to be even more effective, especially for handling softmax operations. Flash attention fuses all attention operations within a transformer into a single, highly efficient kernel:
`F.scaled_dot_product_attention(q,k,v, is_causal = True)`

Another optimization involved using **non-ugly numbers**—specifically, adjusting the vocabulary size from `50,257` to `50,304`, a number more amenable to power-of-2 operations. This adjustment slightly increases the tensor size, padding it with extra characters, but the resulting softmax probabilities for these padded characters are effectively ignored during computations. Despite the additional characters, this tweak boosts overall efficiency.

These optimizations collectively improved performance by **32x**.

Further algorithmic improvements were based on insights from the GPT-3 paper:

* **AdamW Optimizer:** Betas were set to 0.9 and 0.95, with an epsilon of 1e-8.
* **Gradient Clipping:** Gradients were clipped to a norm of 1.0 to prevent large updates from bad batches.
* **Learning Rate Scheduler:** Implemented cosine decay with a warmup period.
* **Weight Decay:** Applied only to weight tensors, not biases, leveraging kernel fusion.
* **Gradient Accumulation:** Simulated a large batch size (up to 0.5 million) through gradient accumulation.

To fully utilize the available hardware, **Distributed Data Parallel (DDP)** was introduced, spreading the workload across 8 GPUs. The training script was executed using:

```bash
torchrun --standalone --nproc_per_node=8 train_gpt2.py ... [other args]
```

Within this setup, gradient synchronization was carefully managed to ensure efficiency:

{{< figure align=center src="/images/GPT2_my_lessons_learnt/grad-sync.png" >}}

This ensures that gradients are only synchronized during the final accumulation step, reducing overhead. It's worth noting that this feature might be deprecated in the future, so ongoing monitoring is advised.

As the model scaled, so did the training dataset. The Hugging Face [FineWeb-edu dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu?ref=sukhvir-ai.ghost.io) (sample-10BT subset) was chosen for its high educational content, providing a substantial training corpus.

For evaluation, several strategies were implemented:

* **Evaluation Frequency:** An evaluation and sample generation were triggered every 100th step.
* **Evaluation Dataset:** The Hellswag dataset was used for this purpose.
* **Learning Rate Experimentation:** Unlike the standard approach, a higher learning rate was tested to observe its impact on the model's learning.

## Lessons Learned

* **Torch.compile() and Flash Attention:** These optimizations are key for reducing computation time and enhancing performance.
* **Power-of-2 Adjustments:** Aligning tensor sizes to power-of-2 values can improve computational efficiency.
* **Algorithmic Tweaks:** Adopting strategies from the GPT-3 paper, such as specific optimizer settings and gradient clipping, significantly stabilizes training.
* **Distributed Training:** Utilizing multiple GPUs effectively with DDP is essential for scaling large models.
* **Dataset Expansion:** Growing the dataset and incorporating high-quality content is critical as the model size increases.
* **Custom Evaluation Strategies:** Regular evaluations and testing different learning rates provide valuable insights into model performance.

## Conclusion

After implementing the various optimizations discussed, including advanced techniques like `torch.compile`, flash attention, and the use of non-ugly numbers, the 124-million parameter GPT-2 model showed significant improvements in both training efficiency and performance. As seen in the training and validation loss graph, the model reached a validation loss comparable to OpenAI's GPT-2 implementation, indicating that the optimizations were effective in maintaining model accuracy while improving computational efficiency. Notably, our model achieved a validation loss of approximately 3.0, which aligns closely with OpenAI’s GPT-2 checkpoint.

{{< figure align=center src="/images/GPT2_my_lessons_learnt/results-1.png" >}}

The HellaSwag evaluation benchmark further highlighted the strengths of our optimized GPT-2 model. While it does not yet match the performance of OpenAI’s GPT-3 model, our implementation consistently outperformed the original GPT-2 baseline in terms of accuracy, steadily climbing to nearly 30%. This demonstrates that with targeted optimizations and careful attention to both hardware and algorithmic efficiency, it is possible to build and train large-scale models that approach the performance of industry-leading implementations. These results reinforce the importance of continuous experimentation and adaptation when working with deep learning models.