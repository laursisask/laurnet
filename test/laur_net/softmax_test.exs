defmodule LaurNet.SoftmaxTest do
  use ExUnit.Case, async: true

  alias LaurNet.Softmax
  alias LaurNet.Layer

  import LaurNet.MathAssert

  test "computes correct output" do
    expected_output = [0.5913, 0.0014, 0.3586, 0.0485, 5.990e-6]

    assert_almost_equal(expected_output, Layer.call(%Softmax{}, [2.5, -3.5, 2, 0, -9]), 1.0e-4)
  end

  test "computes correct gradient with respect to input" do
    expected_gradient = [1.646830, -0.003246, -1.511763, -0.131785, -3.42e-5]

    actual_gradient =
      Layer.gradient_wrt_input(%Softmax{}, [2.5, -3.5, 2, 0, -9], [9, 4, 2, 3.5, 0.5])

    assert_almost_equal(expected_gradient, actual_gradient, 1.0e-6)
  end

  test "computes correct gradient with respect to weights" do
    assert [] == Layer.gradient_wrt_weights(%Softmax{}, [2.5, -3.5, 2, 0, -9], [9, 4, 2, 3.5, 0])
  end
end
