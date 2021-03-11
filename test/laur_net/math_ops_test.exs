defmodule LaurNet.MathOpsTest do
  use ExUnit.Case, async: true

  import LaurNet.MathOps

  test "computes mean" do
    assert 18.5 == mean([1.0, -5.0, 3.5, 4.5, 100.0, 7.0])
    assert 420 == mean([420])
  end

  test "computes standard deviation" do
    assert_in_delta 7.211, std([1.0, -3.0, 11.0]), 1.0e-3
  end

  test "computes argmax" do
    assert 0 == argmax([-5])
    assert 0 == argmax([5, 4, 3, 2, 1])
    assert 2 == argmax([4, 2, 10, 9])
  end
end
