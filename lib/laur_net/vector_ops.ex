defmodule LaurNet.VectorOps do
  @doc """
  Constructs a (nested) list with same shape as the input lists where the value
  of each element is determined by calling the given function with corresponding
  elements of each array.
  """
  def apply_pairwise([x0 | x_rest], [y0 | y_rest], fun) when is_list(x0) and is_list(y0) do
    [apply_pairwise(x0, y0, fun)] ++ apply_pairwise(x_rest, y_rest, fun)
  end

  def apply_pairwise([x0 | x_rest], [y0 | y_rest], fun) when is_number(x0) and is_number(y0) do
    [fun.(x0, y0)] ++ apply_pairwise(x_rest, y_rest, fun)
  end

  def apply_pairwise([], [], _fun) do
    []
  end

  @doc """
  Applies a function to each element of a matrix, vector or other nested
  list.
  """
  def apply_elementwise([x0 | x_rest], fun) when is_list(x0) do
    [apply_elementwise(x0, fun)] ++ apply_elementwise(x_rest, fun)
  end

  def apply_elementwise([x0 | x_rest], fun) when is_number(x0) do
    [fun.(x0)] ++ apply_elementwise(x_rest, fun)
  end

  def apply_elementwise([], _fun) do
    []
  end

  @doc """
  Computes the dot product between vectors.
  """
  def dot([u0 | u_rest], [v0 | v_rest]) do
    u0 * v0 + dot(u_rest, v_rest)
  end

  def dot([], []) do
    0
  end

  @doc """
  Multiplies the given matrices.
  """
  def matrix_multiply(a, b) do
    # shape of a: n x m
    # shape of b: m x p

    n = length(a)
    m = length(List.first(a))
    p = length(List.first(b))

    for i <- 0..(n - 1) do
      for j <- 0..(p - 1) do
        Enum.reduce(0..(m - 1), 0, fn k, acc ->
          x = Enum.at(Enum.at(a, i), k)
          y = Enum.at(Enum.at(b, k), j)

          acc + x * y
        end)
      end
    end
  end

  @doc """
  Constructs a matrix filled with zeros with given shape.
  """
  def zero_matrix(n, m) do
    # n - number of rows
    # m - number of columns

    row = List.duplicate(0, m)

    List.duplicate(row, n)
  end
end
