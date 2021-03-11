defmodule LaurNet.NetworkTest do
  use ExUnit.Case, async: true

  alias LaurNet.Layer
  alias LaurNet.LossFunction
  alias LaurNet.SquaredError
  alias LaurNet.Network

  defmodule Layer1 do
    defstruct w1: [10.0, 100.0]

    defimpl LaurNet.Layer do
      def call(_layer, [4.0, 0.0, -1.0, 5.0, 3.0]) do
        [-1.0, 0.0, 27.0]
      end

      def gradient_wrt_input(_, [4.0, 0.0, -1.0, 5.0, 3.0], [5.0, 0.0, 17.0]) do
        [1.0, 4.0, -7.0, 2.0, 5.0]
      end

      def gradient_wrt_weights(_, [4.0, 0.0, -1.0, 5.0, 3.0], [5.0, 0.0, 17.0]) do
        [9.0, 47.0]
      end

      def weights(layer), do: layer.w1
      def set_weights(layer, w1), do: %{layer | w1: w1}
    end
  end

  defmodule Layer2 do
    defstruct w1: 100.0

    defimpl Layer do
      def call(_layer, [-1.0, 0.0, 27.0]) do
        [2.0, 5.0, 9.0, -4.0]
      end

      def gradient_wrt_input(_, [-1.0, 0.0, 27.0], [1.0, 6.0, 2.0, -5.0]) do
        [5.0, 0.0, 17.0]
      end

      def gradient_wrt_weights(_, [-1.0, 0.0, 27.0], [1.0, 6.0, 2.0, -5.0]) do
        [9.0]
      end

      def weights(layer), do: [layer.w1]
      def set_weights(layer, [w1]), do: %{layer | w1: w1}
    end
  end

  defmodule Layer3 do
    defstruct w1: 42.0, w2: [4.0, -9.0]

    defimpl Layer do
      def call(_layer, [2.0, 5.0, 9.0, -4.0]) do
        [500.0, 2.0]
      end

      def gradient_wrt_input(_, [2.0, 5.0, 9.0, -4.0], [12.0, 3.0]) do
        [1.0, 6.0, 2.0, -5.0]
      end

      def gradient_wrt_weights(_, [2.0, 5.0, 9.0, -4.0], [12.0, 3.0]) do
        [1.0, [-2.0, 5.0]]
      end

      def weights(layer), do: [layer.w1, layer.w2]

      def set_weights(layer, [w1, w2]), do: %{layer | w1: w1, w2: w2}
    end
  end

  defmodule QuadraticLayer do
    # f(x) = a*x^2 + b*x + c
    defstruct a: 0, b: 0, c: 0

    defimpl Layer do
      def call(%{a: a, b: b, c: c}, [x]) do
        [a * x * x + b * x + c]
      end

      def gradient_wrt_input(%{a: a, b: b}, [x], [upstream_grad]) do
        [(2 * a * x + b) * upstream_grad]
      end

      def gradient_wrt_weights(_layer, [x], [upstream_grad]) do
        [x * x * upstream_grad, x * upstream_grad, upstream_grad]
      end

      def weights(%{a: a, b: b, c: c}), do: [a, b, c]

      def set_weights(layer, [a, b, c]), do: %{layer | a: a, b: b, c: c}
    end
  end

  defmodule Loss do
    defstruct []

    defimpl LossFunction do
      def loss(_, [750.0, -5.0], [500.0, 2.0]) do
        128.5
      end

      def gradient(_, [750.0, -5.0], [500.0, 2.0]) do
        [12.0, 3.0]
      end
    end
  end

  test "does a forward pass through the network" do
    network = [
      %Layer1{},
      %Layer2{},
      %Layer3{}
    ]

    assert [500.0, 2.0] == Network.call(network, [4.0, 0.0, -1.0, 5.0, 3.0])
  end

  test "updates weights of the network for single training step" do
    network = [
      %Layer1{},
      %Layer2{},
      %Layer3{}
    ]

    x = [4.0, 0.0, -1.0, 5.0, 3.0]
    y_true = [750.0, -5.0]

    assert {updated_network, 128.5} =
             Network.train_step(network, x, y_true, %Loss{}, learning_rate: 0.5)

    expected_updated_network = [
      %Layer1{w1: [5.5, 76.5]},
      %Layer2{w1: 95.5},
      %Layer3{w1: 41.5, w2: [5, -11.5]}
    ]

    assert updated_network == expected_updated_network
  end

  test "fits a quadratic function" do
    {x, y_true} = generate_quadratic_func_examples(5000)
    validation_data = generate_quadratic_func_examples(400)

    network = [%QuadraticLayer{}]

    {trained_network, validation_loss} =
      Network.train(network, x, y_true, %SquaredError{}, validation_data,
        clip_min: -6.0,
        clip_max: 6.0,
        epochs: 20,
        learning_rate: 0.005
      )

    [%QuadraticLayer{a: final_a, b: final_b, c: final_c}] = trained_network

    assert_in_delta 5, final_a, 1
    assert_in_delta -7, final_b, 1
    assert_in_delta 20, final_c, 8

    assert_in_delta 12, validation_loss, 10
  end

  defp generate_quadratic_func_examples(n) do
    actual_fun = fn x -> 5 * x * x - 7 * x + 20 end

    1..n
    |> Enum.map(fn _ ->
      x = 25 - :rand.uniform(50)
      y = actual_fun.(x)

      {[x], [y]}
    end)
    |> Enum.unzip()
  end
end
