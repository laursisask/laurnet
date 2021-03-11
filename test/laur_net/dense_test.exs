defmodule LaurNet.DenseTest do
  use ExUnit.Case, async: true

  alias LaurNet.Dense
  alias LaurNet.Layer
  import LaurNet.MathAssert

  test "initializes a layer" do
    layer = %Dense{} = Dense.new(input_size: 25, output_size: 4)

    assert [0, 0, 0, 0] == layer.bias

    assert length(layer.kernel) == 25
    assert length(Enum.at(layer.kernel, 0)) == 4

    sum =
      layer.kernel
      |> List.flatten()
      |> Enum.reduce(0, fn x, acc -> x + acc end)

    kernel_weight_average = sum / (25 * 4)

    assert_in_delta 0, kernel_weight_average, 1
  end

  test "computes correct output" do
    kernel = [
      [1.0, 0.0, -0.1],
      [2.0, 2.0, 0.0],
      [0.0, -1.0, 0.0],
      [-1.0, -5.0, 1.0],
      [0.5, 2.5, 0.1],
      [10.0, 3.0, 0.0]
    ]

    bias = [1, 0, 0.25]

    layer = %Dense{
      kernel: kernel,
      bias: bias
    }

    assert [10.5, -6.5, 0.35] == Layer.call(layer, [4.0, -2.0, 3.0, 0.5, 0.0, 1.0])
  end

  test "computes correct gradient wrt weights when upstream gradient is filled with ones" do
    kernel = [
      [1.0, 0.0, -0.1],
      [2.0, 2.0, 0.0],
      [0.0, -1.0, 0.0],
      [-1.0, -5.0, 1.0],
      [0.5, 2.5, 0.1],
      [10.0, 3.0, 0.0]
    ]

    bias = [1, 0, 0.25]

    layer = %Dense{
      kernel: kernel,
      bias: bias
    }

    input = [4.0, -2.0, 3.0, 0.5, 0.0, 1.0]

    expected_kernel_grad = [
      [4.0, 4.0, 4.0],
      [-2.0, -2.0, -2.0],
      [3.0, 3.0, 3.0],
      [0.5, 0.5, 0.5],
      [0.0, 0.0, 0.0],
      [1.0, 1.0, 1.0]
    ]

    expected_bias_grad = [1.0, 1.0, 1.0]

    [actual_kernel_grad, actual_bias_grad] =
      Layer.gradient_wrt_weights(layer, input, [1.0, 1.0, 1.0])

    assert_almost_equal(expected_kernel_grad, actual_kernel_grad)
    assert_almost_equal(expected_bias_grad, actual_bias_grad)
  end

  test "computes correct gradient wrt weights when upstream gradient is not filled with ones" do
    kernel = [
      [1.0, 0.0, -0.1],
      [2.0, 2.0, 0.0],
      [0.0, -1.0, 0.0],
      [-1.0, -5.0, 1.0],
      [0.5, 2.5, 0.1],
      [10.0, 3.0, 0.0]
    ]

    bias = [1, 0, 0.25]

    layer = %Dense{
      kernel: kernel,
      bias: bias
    }

    input = [4.0, -2.0, 3.0, 0.5, 0.0, 1.0]

    expected_kernel_grad = [
      [2.4, 6.8, -10.0],
      [-1.2, -3.4, 5.0],
      [1.8, 5.1, -7.5],
      [0.3, 0.85, -1.25],
      [0.0, 0.0, 0.0],
      [0.6, 1.7, -2.5]
    ]

    expected_bias_grad = [0.6, 1.7, -2.5]

    [actual_kernel_grad, actual_bias_grad] =
      Layer.gradient_wrt_weights(layer, input, [0.6, 1.7, -2.5])

    assert_almost_equal(expected_kernel_grad, actual_kernel_grad)
    assert_almost_equal(expected_bias_grad, actual_bias_grad)
  end

  test "computes correct gradient wrt input" do
    kernel = [
      [1.0, 0.0, -0.1],
      [2.0, 2.0, 0.0],
      [0.0, -1.0, 0.0],
      [-1.0, -5.0, 1.0],
      [0.5, 2.5, 0.1],
      [10.0, 3.0, 0.0]
    ]

    bias = [1, 0, 0.25]

    layer = %Dense{
      kernel: kernel,
      bias: bias
    }

    input = [4.0, -2.0, 3.0, 0.5, 0.0, 1.0]

    expected_gradient = [0.85, 4.6, -1.7, -11.6, 4.3, 11.1]

    actual_gradient = Layer.gradient_wrt_input(layer, input, [0.6, 1.7, -2.5])

    assert_almost_equal(expected_gradient, actual_gradient)
  end

  test "updates weights" do
    layer = %Dense{
      kernel: [
        [2.5, 1.0],
        [0.5, 2.0],
        [1.9, 4.0]
      ],
      bias: [3.0, 7.5]
    }

    new_kernel = [
      [0.5, 10.0],
      [3.7, 2.5],
      [-1.0, -2.5]
    ]

    new_bias = [0.0, -10.5]

    updated_layer = %Dense{} = Layer.set_weights(layer, [new_kernel, new_bias])

    assert updated_layer.kernel == new_kernel
    assert updated_layer.bias == new_bias
  end
end
