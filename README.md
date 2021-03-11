# LaurNet

LaurNet is a machine learning library written in pure Elixir.

This library was written for educational purposes, and it does not attempt to
achieve the level of performance like libraries Numpy and Tensorflow.

## Usage

The library provides a way of constructing multi-layer neural networks with
various layers. A network is simply a list of `LaurNet.Layer` structs.

One can construct a two-layer deep neural network with ReLU and Softmax
activation functions with the following:

```elixir
alias LaurNet.Dense
alias LaurNet.ReLU
alias LaurNet.Softmax

network = [
  Dense.new(input_size: 12, output_size: 8),
  %ReLU{},
  LaurNet.Dense.new(input_size: 8, output_size: 4),
  %Softmax{}
]
```

The `LaurNet.Network` module contains functions for training and inference.

Computing the output of a network:

```elixir
LaurNet.Network.call([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
```

Note that opposed to most other machine learning libraries, LaurNet does not
work with batches. In the last example, the network is called with a single
input that consists of 12 numbers, not 12 inputs. The reason it does not provide
vectorized operations is to keep the implementation simple.

To train a network, one needs an objective function. LaurNet provides two loss
functions:

- Cross entropy (`LaurNet.SparseCategoricalCrossentropy`)
- Squared error (`LaurNet.SquaredError`)

A network can be trained with `LaurNet.Network.train/6` function.

For a more detailed example of how to train a model, see the `mnist` directory.
It trains a classification model for classifying handwritten digits.
