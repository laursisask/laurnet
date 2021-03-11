defmodule LaurNet.SparseCategoricalCrossentropy do
  @moduledoc """
  Cross-entropy loss between labels and predictions.

  This function expects predictions to be vectors of probabilities
  and true values integers indicating the correct category.
  """

  defstruct []

  import LaurNet.MathOps, only: [clip_by_value: 3, epsilon: 0]

  defimpl LaurNet.LossFunction do
    def loss(_loss_function, y_true, y_pred) when is_integer(y_true) and is_list(y_pred) do
      clipped_value = clip_by_value(Enum.at(y_pred, y_true), epsilon(), 1 - epsilon())

      -:math.log(clipped_value)
    end

    def gradient(_loss_function, y_true, y_pred) when is_integer(y_true) and is_list(y_pred) do
      gradient = List.duplicate(0, length(y_pred))

      clipped_value = clip_by_value(Enum.at(y_pred, y_true), epsilon(), 1 - epsilon())

      List.replace_at(gradient, y_true, -1 / clipped_value)
    end
  end
end
