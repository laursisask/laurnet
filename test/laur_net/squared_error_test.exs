defmodule LaurNet.SquaredErrorTest do
  use ExUnit.Case, async: true

  alias LaurNet.SquaredError
  alias LaurNet.LossFunction

  @struct %SquaredError{}

  test "computes correct loss" do
    assert 25 == LossFunction.loss(@struct, [7.5], [2.5])
    assert 0 == LossFunction.loss(@struct, [-3], [-3])
    assert 9 == LossFunction.loss(@struct, [-3], [0])
  end

  test "computes correct gradient" do
    assert [-10] == LossFunction.gradient(@struct, [7.5], [2.5])
    assert [10] == LossFunction.gradient(@struct, [2.5], [7.5])
    assert [0] == LossFunction.gradient(@struct, [2.5], [2.5])
  end
end
