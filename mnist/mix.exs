defmodule MNIST.Mixfile do
  use Mix.Project

  def project do
    [
      app: :mnist,
      version: "0.0.1",
      escript: [main_module: MNIST],
      deps: deps()
    ]
  end

  defp deps do
    [
      {:jason, "~> 1.2"},
      {:laurnet, path: "../"}
    ]
  end
end
