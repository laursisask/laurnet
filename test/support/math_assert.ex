defmodule LaurNet.MathAssert do
  import ExUnit.Assertions

  def assert_almost_equal(expected, actual, delta \\ 1.0e-6)

  def assert_almost_equal([x0 | x_rest], [y0 | y_rest], delta) when is_list(x0) and is_list(y0) do
    assert_almost_equal(x0, y0, delta)
    assert_almost_equal(x_rest, y_rest, delta)
  end

  def assert_almost_equal([x0 | x_rest], [y0 | y_rest], delta)
      when is_number(x0) and is_number(y0) do
    assert_in_delta(x0, y0, delta)
    assert length(x_rest) == length(y_rest)

    assert_almost_equal(x_rest, y_rest, delta)
  end

  def assert_almost_equal([], [], _delta) do
    :ok
  end
end
