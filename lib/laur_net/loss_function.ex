defprotocol LaurNet.LossFunction do
  @doc """
  Computes the loss for one training example.
  """
  def loss(loss_function, y_true, y_pred)

  @doc """
  Computes gradient of the loss function with respect to the predicted
  distribution. This function returns a value with same shape as `y_pred`.
  """
  def gradient(loss_function, y_true, y_pred)
end
