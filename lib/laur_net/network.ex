defmodule LaurNet.Network do
  alias LaurNet.Layer
  alias LaurNet.LossFunction

  require Logger

  import LaurNet.VectorOps, only: [apply_pairwise: 3]

  defguard is_network(x) when is_list(x)

  @type vector :: list
  @type network :: [Layer.t()]

  @doc """
  Evaluates the output of the network for the given input.
  """
  @spec call(network, vector) :: vector
  def call([layer0 | rest_layers], input) do
    layer0_output = Layer.call(layer0, input)
    call(rest_layers, layer0_output)
  end

  def call([], input) do
    input
  end

  def train(initial_network, x, y_true, loss_fun, {validation_x, validation_y_true}, opts \\ [])
      when is_network(initial_network) and length(x) == length(y_true) and is_list(opts) do
    epochs = Keyword.get(opts, :epochs, 10)

    initial_state = {
      initial_network,
      initial_network,
      evaluate(initial_network, loss_fun, validation_x, validation_y_true)
    }

    {_, best_network, best_loss} =
      Enum.reduce(1..epochs, initial_state, fn epoch,
                                               {epoch_initial_network, best_network, best_loss} ->
        Logger.info("Starting epoch #{epoch}/#{epochs}")

        training_examples = Enum.zip(x, y_true)

        {updated_network, total_loss, _counter} =
          training_examples
          |> Enum.shuffle()
          |> Enum.reduce({epoch_initial_network, 0, 1}, fn {x, y_true},
                                                           {network, total_loss, counter} ->
            {updated_network, loss} = train_step(network, x, y_true, loss_fun, opts)

            if rem(counter, 250) == 0 do
              Logger.info(
                "Training step #{counter} completed, " <>
                  "average training loss #{Float.round((loss + total_loss) / counter, 5)}"
              )
            end

            {updated_network, loss + total_loss, counter + 1}
          end)

        train_loss = total_loss / length(x)

        validation_loss = evaluate(updated_network, loss_fun, validation_x, validation_y_true)

        Logger.info(
          "Epoch #{epoch} finished, average training loss #{Float.round(train_loss, 5)}, " <>
            "validation loss #{Float.round(validation_loss, 5)}"
        )

        if validation_loss < best_loss do
          {updated_network, updated_network, validation_loss}
        else
          {updated_network, best_network, best_loss}
        end
      end)

    {best_network, best_loss}
  end

  @spec train_step(network, vector, vector, LossFunction.t(), Keyword.t()) ::
          {network, loss :: number}
  def train_step(network, x, y_true, loss_fun, opts) do
    {outputs, y_pred} =
      Enum.map_reduce(network, x, fn layer, input ->
        out = Layer.call(layer, input)

        {out, out}
      end)

    loss = LossFunction.loss(loss_fun, y_true, y_pred)

    loss_grad = LossFunction.gradient(loss_fun, y_true, y_pred)

    inputs = [x] ++ outputs

    {grads_reversed, _} =
      [network, inputs]
      |> Enum.zip()
      |> Enum.reverse()
      |> Enum.map_reduce(loss_grad, fn {layer, input}, upstream_grad ->
        grad_wrt_weights = Layer.gradient_wrt_weights(layer, input, upstream_grad)
        grad_wrt_input = Layer.gradient_wrt_input(layer, input, upstream_grad)

        {grad_wrt_weights, grad_wrt_input}
      end)

    grads = Enum.reverse(grads_reversed)

    updated_network = apply_gradients(network, grads, opts)

    {updated_network, loss}
  end

  defp apply_gradients(network, grads, opts) when length(network) == length(grads) do
    learning_rate = Keyword.get(opts, :learning_rate, 1.0e-4)
    clip_min = Keyword.get(opts, :clip_min)
    clip_max = Keyword.get(opts, :clip_max)

    [network, grads]
    |> Enum.zip()
    |> Enum.map(fn {layer, grad} ->
      current_weights = Layer.weights(layer)

      updated_weights =
        apply_pairwise(current_weights, grad, fn w, d ->
          d = clip(d, clip_min, clip_max)

          w - learning_rate * d
        end)

      Layer.set_weights(layer, updated_weights)
    end)
  end

  @spec evaluate(network, LossFunction.t(), list, list) :: number
  def evaluate(network, loss_fun, x, y_true)
      when is_network(network) and length(x) == length(y_true) do
    sum_of_losses =
      [x, y_true]
      |> Enum.zip()
      |> Task.async_stream(fn {x, y_true} ->
        y_pred = call(network, x)

        LossFunction.loss(loss_fun, y_true, y_pred)
      end)
      |> Enum.reduce(0, fn {:ok, loss}, acc -> loss + acc end)

    sum_of_losses / length(x)
  end

  defp clip(value, min, max) when is_number(value) and is_number(min) and is_number(max) do
    value
    |> max(min)
    |> min(max)
  end

  defp clip(value, min, max) when is_number(value) and is_nil(min) and is_number(max) do
    min(value, max)
  end

  defp clip(value, min, max) when is_number(value) and is_number(min) and is_nil(max) do
    max(value, min)
  end

  defp clip(value, min, max) when is_number(value) and is_nil(min) and is_nil(max) do
    value
  end
end
