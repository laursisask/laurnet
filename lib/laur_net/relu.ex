defmodule LaurNet.ReLU do
  @moduledoc """
  A rectifier linear unit layer.
  """

  import LaurNet.VectorOps

  defstruct []

  defimpl LaurNet.Layer do
    def call(_, input) do
      apply_elementwise(input, &relu_scalar/1)
    end

    def gradient_wrt_weights(_, _, _) do
      []
    end

    def gradient_wrt_input(layer, [x0 | x_rest], [d0 | d_rest]) do
      [derivative(x0, d0)] ++ gradient_wrt_input(layer, x_rest, d_rest)
    end

    def gradient_wrt_input(_, [], []) do
      []
    end

    def weights(_layer) do
      []
    end

    def set_weights(layer, []) do
      layer
    end

    defp relu_scalar(x), do: max(0, x)

    # d - upstream derivative
    defp derivative(x, d) when x >= 0, do: d
    defp derivative(_, _), do: 0
  end
end
