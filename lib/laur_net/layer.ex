defprotocol LaurNet.Layer do
  @type vector :: [number]

  @spec call(t, input :: vector) :: vector
  def call(layer, input)

  @spec gradient_wrt_weights(t, input :: vector, upstream_gradient :: vector) :: list
  def gradient_wrt_weights(layer, input, upstream_gradient)

  @spec gradient_wrt_input(t, input :: vector, upstream_gradient :: vector) :: vector
  def gradient_wrt_input(layer, input, upstream_gradient)

  @spec weights(t) :: list
  def weights(layer)

  @spec set_weights(t, list) :: t
  def set_weights(layer, weights)
end
