defmodule LaurNet.MathOps do
  @doc """
  Returns the fuzz factor used in calculations.
  """
  def epsilon do
    1.0e-7
  end

  @doc """
  Clips value to a specified minimum and maximum.
  """
  def clip_by_value(value, min, max)
      when is_number(value) and is_number(min) and is_number(max) do
    value
    |> min(max)
    |> max(min)
  end

  @doc """
  Computes the standard deviation of a sample from the population.
  """
  def std(a) when length(a) > 1 do
    mean = mean(a)

    sum =
      a
      |> Enum.map(fn x -> (x - mean) * (x - mean) end)
      |> reduce_sum()

    :math.sqrt(sum / (length(a) - 1))
  end

  @doc """
  Computes the mean of the enumerable.
  """
  def mean(a) when length(a) > 0 do
    sum = reduce_sum(a)

    sum / length(a)
  end

  @doc """
  Computes the sum of enumerable elements.
  """
  def reduce_sum(a) do
    Enum.reduce(a, 0, fn x, acc -> x + acc end)
  end

  @doc """
  Returns the index of the maximum value.
  """
  def argmax(a) when length(a) > 0 do
    max_value = Enum.max(a)

    Enum.find_index(a, fn x -> x == max_value end)
  end
end
