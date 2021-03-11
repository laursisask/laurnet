defmodule LaurNet.ReLUTest do
  use ExUnit.Case, async: true

  alias LaurNet.Layer
  alias LaurNet.ReLU

  test "computes correct output" do
    assert [0.0, 0.0, 2.5, 1.25, 100.0, 0.0] ==
             Layer.call(%ReLU{}, [-1.5, 0.0, 2.5, 1.25, 100.0, -100.0])
  end

  test "computes correct gradient with respect to input" do
    input = [1.5, 20, -10, 0, 2.5, -3]
    upstream_gradient = [-2.5, 3, 0.5, 0, 0.5, 1]

    assert [-2.5, 3.0, 0.0, 0.0, 0.5, 0.0] ==
             Layer.gradient_wrt_input(%ReLU{}, input, upstream_gradient)
  end

  test "computes correct gradient with respect to weights" do
    input = [1.5, 20, -10, 0, 2.5, -3]
    upstream_gradient = [-2.5, 3, 0.5, 0, 0.5, 1]

    assert [] == Layer.gradient_wrt_weights(%ReLU{}, input, upstream_gradient)
  end

  test "does not do anything when weights updated" do
    layer = %ReLU{}

    assert layer == Layer.set_weights(layer, [])
  end
end
