defmodule LaurNet.SquaredError do
  @moduledoc """
  Error between true value and prediction squared.

  Like mean squared error but for one value.
  """

  defstruct []

  defimpl LaurNet.LossFunction do
    def loss(_loss_function, [y_true], [y_pred]) when is_number(y_true) and is_number(y_pred) do
      (y_true - y_pred) * (y_true - y_pred)
    end

    def gradient(_loss_function, [y_true], [y_pred])
        when is_number(y_true) and is_number(y_pred) do
      [2 * y_pred - 2 * y_true]
    end
  end
end
