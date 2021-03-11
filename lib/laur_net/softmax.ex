defmodule LaurNet.Softmax do
  @moduledoc """
  A softmax layer.
  """

  defstruct []

  import LaurNet.VectorOps, only: [dot: 2]

  defimpl LaurNet.Layer do
    def call(_layer, input) do
      max_input = Enum.max(input)

      sum =
        Enum.reduce(input, 0, fn x, acc ->
          acc + :math.exp(x - max_input)
        end)

      Enum.map(input, fn x -> :math.exp(x - max_input) / sum end)
    end

    def gradient_wrt_weights(_layer, _input, _upstream_grad) do
      []
    end

    def gradient_wrt_input(layer, input, upstream_grad) do
      out = call(layer, input)

      for i <- 0..(length(input) - 1) do
        out_i = Enum.at(out, i)

        l =
          for j <- 0..(length(input) - 1) do
            out_j = Enum.at(out, j)

            if i == j do
              out_i * (1 - out_i)
            else
              -out_i * out_j
            end
          end

        dot(upstream_grad, l)
      end
    end

    def weights(_layer), do: []
    def set_weights(layer, []), do: layer
  end
end
