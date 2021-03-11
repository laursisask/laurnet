defmodule MNIST do
  alias LaurNet.Network
  alias LaurNet.ReLU
  alias LaurNet.Dense
  alias LaurNet.Softmax

  import LaurNet.MathOps

  require Logger

  def main(_) do
    all_examples = Enum.shuffle(load_data())

    Logger.info("Data loaded to memory, total number of examples: #{length(all_examples)}")

    validation_examples_count = trunc(0.2 * length(all_examples))

    {validation_data, train_data} = Enum.split(all_examples, validation_examples_count)

    color_mean =
      all_examples
      |> Enum.map(fn {image, _} -> image end)
      |> List.flatten()
      |> mean()

    color_std =
      all_examples
      |> Enum.map(fn {image, _} -> image end)
      |> List.flatten()
      |> std()

    validation_data =
      Enum.map(validation_data, fn {image, category} ->
        image = Enum.map(image, fn x -> (x - color_mean) / color_std end)
        {image, category}
      end)

    train_data =
      Enum.map(train_data, fn {image, category} ->
        image = Enum.map(image, fn x -> (x - color_mean) / color_std end)

        {image, category}
      end)

    network = [
      Dense.new(input_size: 28 * 28, output_size: 64),
      %ReLU{},
      Dense.new(input_size: 64, output_size: 10),
      %Softmax{}
    ]

    {x, y_true} = Enum.unzip(train_data)
    {validation_x, validation_y_true} = Enum.unzip(validation_data)

    loss_fun = %LaurNet.SparseCategoricalCrossentropy{}

    Logger.info("Starting training")

    {trained_network, validation_loss} =
      Network.train(network, x, y_true, loss_fun, {validation_x, validation_y_true},
        epochs: 1,
        learning_rate: 0.01
      )

    accuracy = accuracy(trained_network, color_mean, color_std, validation_data)

    Logger.info("Training finished, validation loss: #{validation_loss}, accuracy: #{accuracy}")

    save_network(trained_network, color_mean, color_std)
  end

  defp accuracy(network, color_mean, color_std, validation_data) do
    true_positives =
      validation_data
      |> Enum.map(fn {image, true_category} ->
        image = Enum.map(image, fn x -> (x - color_mean) / color_std end)

        predicted_distribution = Network.call(network, image)

        if argmax(predicted_distribution) == true_category, do: 1, else: 0
      end)
      |> reduce_sum()

    true_positives / length(validation_data)
  end

  defp load_data do
    nested_examples =
      for category <- 0..9 do
        "data/#{category}/*.json"
        |> Path.wildcard()
        |> Task.async_stream(fn filename ->
          image =
            filename
            |> File.read!()
            |> Jason.decode!()
            |> List.flatten()

          {image, category}
        end)
        |> Enum.map(fn {:ok, {image, category}} -> {image, category} end)
      end

    List.flatten(nested_examples)
  end

  defp save_network(network, color_mean, color_std) do
    Logger.info("Saving network to file")

    encoded_network = :erlang.term_to_binary({network, color_mean, color_std})

    File.write!("network", encoded_network)
  end
end
