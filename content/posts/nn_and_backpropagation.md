---
title: "🧠 Neural Networks and Backpropagation"
date: "2024-09-01"
showToc: true
TocOpen: false
draft: false
hidemeta: false
comments: true
categories: ["Deep Dive"]
tags: ["AI"]
description: "Follow my journey along Andrej Karpathy's YouTube series on AI/ML zero to hero training"
disableHLJS: true # to disable highlightjs
disableShare: false
hideSummary: false
searchHidden: false
ShowReadingTime: true
ShowBreadCrumbs: true
ShowWordCount: true
ShowRssButtonInSectionTermList: true
UseHugoToc: true
cover:
    image: "/images/nn_and_backpropagation/cover.jpg" # image path/url
    caption: "Source: https://serokell.io/blog/understanding-backpropagation" # display caption under cover
    relative: false # when using page bundles set this to true
    hiddenInList: true # only hide on current single page
---



---

# Neural Networks & Backpropagation: My First Steps with Karpathy's Zero to Hero

Welcome to the first installment of my blog series inspired by Andrej Karpathy's "Zero to Hero" lecture series on AI and Machine Learning. Buckle up, because I'm diving headfirst into the magical world of neural networks and backpropagation!

{{< youtube VMj-3S1tku0 >}}

**My Jupyter notebook for this post:** [here](https://github.com/sukhvir-notra/AI-Learning-Zero-to-Hero?ref=sukhvir-ai.ghost.io)

---

## A Calculus Refresher: Derivatives, Slopes, and Other Fun Stuff

Karpathy kicks off the lecture with a crash course in basic calculus, focusing specifically on derivatives. Now, if you're like me and calculus brings back memories of late-night study sessions and too much caffeine, fear not! This foundational knowledge is essential for understanding how neural networks (NNs) function.

Essentially, I'm trying to wrap my head around how changing weights and biases—the key parameters of NNs—affect the output. In other words, derivatives help me figure out how much each weight in the network impacts the final output.

**Key takeaways:**

* **Derivatives and Impact:** The derivative of the output with respect to each leaf node in the network shows me how much influence those nodes have on the overall output. This is crucial for tuning the NN effectively.
* **Backpropagation Visualized:** Karpathy walks through a detailed example of backpropagation using a pseudo neural network. This step-by-step process helps me visualize how backpropagation works and how chain rule derivatives come into play.
* **Activation Functions Galore:** I also touched on activation functions like `tanh` and `sigmoid`, which are the magic sauce that helps NNs make decisions.
* **Automating Backpropagation:** The lecture discusses how to algorithmically automate backpropagation and the importance of topological sort to ensure that I don't backpropagate until the forward pass is complete.
* **Level of Detail:** Depending on my masochistic tendencies (or dedication to understanding every little detail), I can either implement the `tanh` function directly or break it down into its individual components.
* **PyTorch to the Rescue:** PyTorch offers some handy abstractions to make life easier. For instance, its tensors are `float32` by default, but I can cast them to `.double()` for `float64` precision, matching Python's default floating-point precision. Also, I need to enable gradient calculation explicitly for leaf nodes (`requires_grad=True`), as they don't do it by default for efficiency reasons.

Here's a snippet showing this in action:

```python
import torch

'''
Casting the tensors as doubles and enabling gradient calculation
'''
x1 = torch.Tensor([2.0]).double()                ; x1.requires_grad = True
x2 = torch.Tensor([0.0]).double()                ; x2.requires_grad = True
w1 = torch.Tensor([-3.0]).double()               ; w1.requires_grad = True
w2 = torch.Tensor([1.0]).double()                ; w2.requires_grad = True
b = torch.Tensor([6.8813735870195432]).double()  ; b.requires_grad = True

# Forward pass calculation
n = x1*w1 + x2*w2 + b
o = torch.tanh(n)

print(f"Output (o): {o.data.item()}")

# Backward pass (calculating gradients)
o.backward()

print('--- Gradients ---')
print(f'x2 grad: {x2.grad.item()}')
print(f'w2 grad: {w2.grad.item()}')
print(f'x1 grad: {x1.grad.item()}')
print(f'w1 grad: {w1.grad.item()}')
print(f'b grad: {b.grad.item()}') # Added bias gradient display
```

---

## Building Blocks: Neurons, Layers, and MLPs

I then dove into the class definitions of a single neuron, a layer of neurons, and a Multi-Layer Perceptron (MLP). This section is all about understanding the fundamental building blocks of NNs and how they interconnect and work together.

### Main takeaways:

Understanding the components is one thing; training them is another!

* **The Learning Rate Dilemma:** One of the critical aspects of training NNs is choosing the appropriate learning rate. It's a bit of a Goldilocks problem: too large a step size and I might overshoot the optimal minimum loss; too small and training will be painfully slow (or get stuck in local minima).
* **Pro Tip - Zeroing the Grads:** A key step *before* backpropagation in each training iteration in PyTorch is to zero out the gradients from the previous step (`p.grad = 0.0`). If I forget this, PyTorch accumulates gradients across iterations, leading to incorrect updates and a lot of head-scratching.

Here's a basic training loop structure illustrating this:

```python
# Assuming 'n' is our MLP model, 'xs' are inputs, 'ys' are target outputs
# And we have defined a loss function (e.g., Mean Squared Error implicitly below)

learning_rate = 0.05
epochs = 20

for k in range(epochs):
  
  # Forward pass: Get predictions for all inputs
  ypred = [n(x) for x in xs]
  # Calculate loss (sum of squared errors example)
  loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
  
  # Backward pass
  # >>> Crucial Step: Zero the gradients before calculating new ones <<<
  for p in n.parameters():
    p.grad = 0.0 
  loss.backward() # Calculate gradients for this batch
  
  # Update weights and biases
  with torch.no_grad(): # Temporarily disable gradient tracking for updates
      for p in n.parameters():
        p.data += -learning_rate * p.grad # Gradient descent step
  
  print(f"Epoch {k}, Loss: {loss.item()}") # Use .item() to get Python number
```

---

## Summing It Up

In summary, this first dive covered how neural networks are structured, how to calculate their loss (how wrong they are), and how to use backpropagation (leveraging the power of derivatives via the chain rule, implemented as gradient descent) to minimize this loss and make the network learn. I explored the `tanh` activation function, built the fundamental components of a neural network from scratch, and got my hands dirty with PyTorch basics.

Stay tuned for more posts in this series as I continue my journey from zero to hero in the world of AI and ML! And remember, in the immortal words of Karpathy (and perhaps a few stressed-out students), **"Happy learning, and may your gradients always descend!"**

---