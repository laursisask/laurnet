defmodule LaurNet.VectorOpsTest do
  use ExUnit.Case, async: true

  import LaurNet.VectorOps

  describe "apply_pairwise" do
    test "applies function to vector" do
      assert [5, 12, 9, -98, 6] == apply_pairwise([5, 9, 4, 2, 1], [0, 3, 5, -100, 5], &+/2)
    end

    test "applies function to matrix" do
      m1 = [
        [11, 16, 0],
        [13, 8, 7],
        [19, 2, 12],
        [18, 14, 9]
      ]

      m2 = [
        [12, 16, 13],
        [15, 13, 11],
        [4, 0, 15],
        [15, 4, 15]
      ]

      expected_result = [
        [23, 32, 13],
        [28, 21, 18],
        [23, 2, 27],
        [33, 18, 24]
      ]

      assert expected_result == apply_pairwise(m1, m2, &+/2)
    end

    test "applies function to nested variable-length lists" do
      x1 = [1, [2, 3], [4, 5, 5], [10]]
      y1 = [-3, [4, -3], [1, 0, 2], [9]]

      expected_result = [-2, [6, 0], [5, 5, 7], [19]]

      assert expected_result == apply_pairwise(x1, y1, &+/2)
    end
  end

  describe "apply_elementwise" do
    test "applies function to vector" do
      assert [6, 7, 8, 0, 5] == apply_elementwise([1, 2, 3, -5, 0], &add_five/1)
    end

    test "applies function to matrix" do
      m = [
        [1, 2, 6],
        [0, 1, 4],
        [5, 9, 1],
        [0, 1, 4]
      ]

      expected_result = [
        [6, 7, 11],
        [5, 6, 9],
        [10, 14, 6],
        [5, 6, 9]
      ]

      assert expected_result == apply_elementwise(m, &add_five/1)
    end
  end

  describe "dot" do
    test "computes correct dot product between 1D vectors" do
      assert -10.0 == dot([5.0], [-2.0])
    end

    test "computes correct dot product between high dimensional vectors" do
      u = [0.5, 1.0, 2.0, 4.5, 0.0]
      v = [-3.0, 2.0, 2.5, 15.0, -1.0]

      assert 73.0 == dot(u, v)
    end
  end

  describe "zero_matrix" do
    test "creates a matrix" do
      assert [[0]] == zero_matrix(1, 1)

      assert [[0, 0, 0, 0, 0]] == zero_matrix(1, 5)

      assert [
               [0, 0, 0],
               [0, 0, 0],
               [0, 0, 0]
             ] == zero_matrix(3, 3)

      assert [
               [0, 0, 0],
               [0, 0, 0]
             ] == zero_matrix(2, 3)
    end
  end

  describe "matrix_multiply" do
    test "multiplies matrices" do
      assert [[32]] == matrix_multiply([[1.0, 2.0, 3.0]], [[4.0], [5.0], [6.0]])
      assert [[-35]] == matrix_multiply([[7.0]], [[-5.0]])

      assert [[70, 170], [150, 370]] ==
               matrix_multiply([[10.0, 20.0], [30.0, 40.0]], [[1.0, 3.0], [3.0, 7.0]])
    end
  end

  defp add_five(x) do
    x + 5
  end
end
