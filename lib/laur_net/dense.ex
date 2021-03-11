defmodule LaurNet.Dense do
  @moduledoc """
  Regular densely-connected neural network layer.
  """

  import LaurNet.VectorOps

  @enforce_keys [:kernel, :bias]
  defstruct [:kernel, :bias]

  @doc """
  Creates a new dense layer with given size.

  The weights of the layer are initialized with Glorot uniform initializer.
  """
  def new(opts) do
    input_size = Keyword.fetch!(opts, :input_size)
    output_size = Keyword.fetch!(opts, :output_size)

    kernel =
      apply_elementwise(zero_matrix(input_size, output_size), fn _ ->
        draw_weight(input_size, output_size)
      end)

    bias = List.duplicate(0, output_size)

    %__MODULE__{
      kernel: kernel,
      bias: bias
    }
  end

  @doc """
  Returns the size of layer's output space.
  """
  def units(%__MODULE__{} = layer_def) do
    length(layer_def.bias)
  end

  defimpl LaurNet.Layer do
    def call(layer, input) do
      matrix_multiply([input], layer.kernel)
      |> List.flatten()
      |> apply_pairwise(layer.bias, &+/2)
    end

    def gradient_wrt_weights(_layer, input, upstream_grad) do
      kernel_grad =
        for x <- input do
          for g <- upstream_grad do
            x * g
          end
        end

      [kernel_grad, upstream_grad]
    end

    def gradient_wrt_input(layer, _input, upstream_grad) do
      for row <- layer.kernel do
        dot(row, upstream_grad)
      end
    end

    def weights(layer) do
      [layer.kernel, layer.bias]
    end

    def set_weights(layer, weights) do
      [kernel, bias] = weights

      %{layer | kernel: kernel, bias: bias}
    end
  end

  defp draw_weight(input_size, output_size) do
    # Draw sample from [0, 1)
    x = :rand.uniform()

    # Move the sample to (-1, 1)
    x =
      if :rand.uniform() > 0.5 do
        x
      else
        -x
      end

    # Glorot uniform limit
    limit = :math.sqrt(6 / (input_size + output_size))

    # Move the sample to (-limit, limit)
    x * limit
  end
end
