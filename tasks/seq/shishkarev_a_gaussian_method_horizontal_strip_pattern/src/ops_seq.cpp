#include "seq/shishkarev_a_gaussian_method_horizontal_strip_pattern/include/ops_seq.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <vector>

using namespace std::chrono_literals;

int shishkarev_a_gaussian_method_horizontal_strip_pattern_seq::MatrixRank(Matrix matrix, std::vector<double> a) {
  int rank = matrix.cols;
  for (int i = 0; i < matrix.cols; ++i) {
    int j = 0;
    for (j = 0; j < matrix.rows; ++j) {
      if (std::abs(a[(j * matrix.rows) + i]) > 1e-6) {
        break;
      }
    }
    if (j == matrix.rows) {
      --rank;
    } else {
      for (int k = i + 1; k < matrix.cols; ++k) {
        double ml = a[(k * matrix.rows) + i] / a[(i * matrix.rows) + i];
        for (j = i; j < matrix.rows - 1; ++j) {
          a[(k * matrix.rows) + j] -= a[(i * matrix.rows) + j] * ml;
        }
      }
    }
  }
  return rank;
}
double shishkarev_a_gaussian_method_horizontal_strip_pattern_seq::Determinant(Matrix matrix, std::vector<double> a) {
  double det = 1;

  for (int i = 0; i < matrix.cols; ++i) {
    int idx = i;
    for (int k = i + 1; k < matrix.cols; ++k) {
      if (std::abs(a[(k * matrix.rows) + i]) > std::abs(a[(idx * matrix.rows) + i])) {
        idx = k;
      }
    }
    if (std::abs(a[(idx * matrix.rows) + i]) < 1e-6) {
      return 0;
    }
    if (idx != i) {
      for (int j = 0; j < matrix.rows - 1; ++j) {
        double tmp = a[(i * matrix.rows) + j];
        a[(i * matrix.rows) + j] = a[(idx * matrix.rows) + j];
        a[(idx * matrix.rows) + j] = tmp;
      }
      det *= -1;
    }
    det *= a[(i * matrix.rows) + i];
    for (int k = i + 1; k < matrix.cols; ++k) {
      double ml = a[(k * matrix.rows) + i] / a[(i * matrix.rows) + i];
      for (int j = i; j < matrix.rows - 1; ++j) {
        a[(k * matrix.rows) + j] -= a[(i * matrix.rows) + j] * ml;
      }
    }
  }
  return det;
}

template <typename InOutType>
bool shishkarev_a_gaussian_method_horizontal_strip_pattern_seq::MPIGaussHorizontalSequential<
    InOutType>::PreProcessingImpl() {
  matrix_ = std::vector<double>(task_data->inputs_count[0]);
  auto *tmp_ptr = reinterpret_cast<double *>(task_data->inputs[0]);
  std::ranges::copy(tmp_ptr, tmp_ptr + task_data->inputs_count[0], matrix_.begin());
  cols_ = static_cast<int>(task_data->inputs_count[1]);
  rows_ = static_cast<int>(task_data->inputs_count[2]);

  res_ = std::vector<double>(cols_ - 1, 0);
  return true;
}

template <typename InOutType>
bool shishkarev_a_gaussian_method_horizontal_strip_pattern_seq::MPIGaussHorizontalSequential<
    InOutType>::ValidationImpl() {
  matrix_ = std::vector<double>(task_data->inputs_count[0]);
  auto *tmp_ptr = reinterpret_cast<double *>(task_data->inputs[0]);
  std::ranges::copy(tmp_ptr, tmp_ptr + task_data->inputs_count[0], matrix_.begin());
  cols_ = static_cast<int>(task_data->inputs_count[1]);
  rows_ = static_cast<int>(task_data->inputs_count[2]);
  Matrix matrix;
  matrix.cols = cols_;
  matrix.rows = rows_;

  return task_data->inputs_count[0] > 1 && rows_ == cols_ - 1 && Determinant(matrix, matrix_) != 0 &&
         MatrixRank(matrix, matrix_) == rows_;
}

template <typename InOutType>
bool shishkarev_a_gaussian_method_horizontal_strip_pattern_seq::MPIGaussHorizontalSequential<InOutType>::RunImpl() {
  for (int i = 0; i < rows_ - 1; ++i) {
    for (int k = i + 1; k < rows_; ++k) {
      double m = matrix_[(k * cols_) + i] / matrix_[(i * cols_) + i];
      for (int j = i; j < cols_; ++j) {
        matrix_[(k * cols_) + j] -= matrix_[(i * cols_) + j] * m;
      }
    }
  }
  for (int i = rows_ - 1; i >= 0; --i) {
    double sum = matrix_[(i * cols_) + rows_];
    for (int j = i + 1; j < cols_ - 1; ++j) {
      sum -= matrix_[(i * cols_) + j] * res_[j];
    }
    res_[i] = sum / matrix_[(i * cols_) + i];
  }
  return true;
}

template <typename InOutType>
bool shishkarev_a_gaussian_method_horizontal_strip_pattern_seq::MPIGaussHorizontalSequential<
    InOutType>::PostProcessingImpl() {
  auto *this_matrix = reinterpret_cast<double *>(task_data->outputs[0]);
  std::ranges::copy(res_.begin(), res_.end(), this_matrix);
  return true;
}

template class shishkarev_a_gaussian_method_horizontal_strip_pattern_seq::MPIGaussHorizontalSequential<int32_t>;
template class shishkarev_a_gaussian_method_horizontal_strip_pattern_seq::MPIGaussHorizontalSequential<float>;
template class shishkarev_a_gaussian_method_horizontal_strip_pattern_seq::MPIGaussHorizontalSequential<double>;
template class shishkarev_a_gaussian_method_horizontal_strip_pattern_seq::MPIGaussHorizontalSequential<uint8_t>;
template class shishkarev_a_gaussian_method_horizontal_strip_pattern_seq::MPIGaussHorizontalSequential<int64_t>;