defmodule LaurNet.Integration.NetworkTest do
  use ExUnit.Case, async: true

  alias LaurNet.Dense
  alias LaurNet.ReLU
  alias LaurNet.Network

  @dense1 %Dense{
    kernel: [
      [1.0, 0.0, -0.1],
      [2.0, 2.0, 0.0],
      [0.0, -1.0, 0.0],
      [-1.0, -5.0, 1.0],
      [0.5, 2.5, 0.1],
      [10.0, 3.0, 0.0]
    ],
    bias: [1, 0, 0.25]
  }

  @dense2 %Dense{
    kernel: [
      [-0.25, 3],
      [1.0, 4.5],
      [2.5, 0]
    ],
    bias: [-0.25, 1.0]
  }

  @relu %ReLU{}

  test "does a forward pass through a network with one hidden layer and ReLU as activation function" do
    network = [
      @dense1,
      @relu,
      @dense2
    ]

    assert [5.75, 11.5] == Network.call(network, [1, 2, 0, 3, -4, 0.25])
  end
end
