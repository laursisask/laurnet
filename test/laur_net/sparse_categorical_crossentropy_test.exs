defmodule LaurNet.SparseCategoricalCrossentropyTest do
  use ExUnit.Case, async: true

  alias LaurNet.SparseCategoricalCrossentropy
  alias LaurNet.LossFunction

  @struct %SparseCategoricalCrossentropy{}

  test "computes correct loss" do
    assert_in_delta 0.510825, LossFunction.loss(@struct, 3, [0.05, 0.05, 0.1, 0.6, 0.2]), 1.0e-6
    assert_in_delta 0, LossFunction.loss(@struct, 1, [0, 1]), 1.0e-6
    assert_in_delta 16.118, LossFunction.loss(@struct, 0, [0, 1]), 1.0e-3
  end

  test "computes correct gradient" do
    assert [0, 0, 0, -1.6666666666666667, 0] ==
             LossFunction.gradient(@struct, 3, [0.05, 0.05, 0.1, 0.6, 0.2])

    assert [0, -1.00000010000001] == LossFunction.gradient(@struct, 1, [0, 1])
  end
end
