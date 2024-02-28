# BitNet

## Introduction
This project is an ikplementation of a popular paper BitNet from Microsoft [BitNet 1.58 bits(https://arxiv.org/pdf/2402.17764.pdf).

The paper introduces a novel technique called BitLinear, which utilizes binary weights in the linear layer.

![](https://arxiv.org/html/2402.17764v1/x2.png)


 The implemention the BitLinear layer should then be compared to a regular implementation.
 
 This implementation also uses the paper's logic where multiplication should be replaced by (signed) summation, to speed up inference.

## Structure
The repository is organized as follows:

- `src/`: Contains the source code for the BitNet implementation.
- `test/`: Includes notebooks or scripts for running experiments and comparing results.

## Implementation
The implementation starts with the BitLinear layer, which replaces the traditional linear layer in the neural network architecture. Detailed explanations and code snippets  are provided in the `src/` directory and testing in `src/`


## Transformer Implementation
This repo's aim is to implement a whole **Transformer** using BitLinear's logic and comparing it side by side to a regular floating point **Transformer**.

## Future work -- Comparing Results
To evaluate the performance of BitNet, it's compared it to a regular implementation of the Transformer architecture. Experiments are conducted on various datasets and measure metrics such as accuracy, training time, and memory usage. The results and analysis can be found in the `experiments/` directory.


## Results

Excerpt from test.ipynb

__This is a test to see the binary optimization in comparison with floating point__

```python
from torch import optim
#now let's optimize the binary layer on outputting all ones

# create a binary layer
binary_linear = BitLinear(10, 1)
# create an optimizer
optimizer = optim.SGD(binary_linear.parameters(), lr=0.01)
# create a loss function
criterion = nn.BCEWithLogitsLoss()
# pass dummy data
x = torch.rand(10)

# now lets update the weights
for i in range(1000):
    # first we need to zero the gradients
    optimizer.zero_grad()
    # then we can update the weights
    output = binary_linear(x)
    loss = criterion(output, torch.ones(1))
    loss.backward()
    optimizer.step()
    print(loss.item())
```

Finally, we test the output by multiplying both the **float** weight and **binary** weight with the same input

```python


x = torch.rand(10)

#first regular pass with binary weights form __forward__ implementation

print(binary_linear(x)) 


#now pass the regular float weights
non_binary_weights = binary_linear.weight

print(F.linear(x, non_binary_weights, binary_linear.bias))

```
```terminal
tensor([3.2173], grad_fn=<SqueezeBackward4>)
tensor([3.2173], grad_fn=<SqueezeBackward4>)
```


