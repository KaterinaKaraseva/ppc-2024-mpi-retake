#include "mpi/shishkarev_a_gaussian_method_horizontal_strip_pattern/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/gather.hpp>
#include <boost/mpi/status.hpp>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace std::chrono_literals;

int shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::MatrixRank(Matrix matrix, std::vector<double> a) {
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
        double pivot = a[(i * matrix.rows) + i];
        if (std::abs(pivot) < 1e-6) {
          return 0;
        }
        double ml = a[(k * matrix.rows) + i] / a[(i * matrix.rows) + i];
        for (j = i; j < matrix.rows - 1; ++j) {
          a[(k * matrix.rows) + j] -= a[(i * matrix.rows) + j] * ml;
        }
      }
    }
  }
  return rank;
}
double shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::Determinant(Matrix matrix, std::vector<double> a) {
  double det = 1.0;

  for (int i = 0; i < matrix.cols; ++i) {
    int idx = i;
    for (int k = i + 1; k < matrix.cols; ++k) {
      if (std::abs(a[(k * matrix.rows) + i]) > std::abs(a[(idx * matrix.rows) + i])) {
        idx = k;
      }
    }
    if (std::abs(a[(idx * matrix.rows) + i]) < 1e-6) {
      return 0.0;
    }
    if (idx != i) {
      for (int j = 0; j < matrix.cols; ++j) {
        double tmp = a[(i * matrix.rows) + j];
        a[(i * matrix.rows) + j] = a[(idx * matrix.rows) + j];
        a[(idx * matrix.rows) + j] = tmp;
      }
      det *= -1.0;
    }
    det *= a[(i * matrix.rows) + i];
    for (int k = i + 1; k < matrix.cols; ++k) {
      double pivot = a[(i * matrix.rows) + i];
      if (std::abs(pivot) < 1e-6) {
        return 0.0;
      }
      double ml = a[(k * matrix.rows) + i] / a[(i * matrix.rows) + i];
      for (int j = i; j < matrix.cols; ++j) {
        a[(k * matrix.rows) + j] -= a[(i * matrix.rows) + j] * ml;
      }
    }
  }
  return det;
}

bool shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::MPIGaussHorizontalSequential::PreProcessingImpl() {
  matrix_ = std::vector<double>(task_data->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  std::ranges::copy(tmp_ptr, tmp_ptr + task_data->inputs_count[0], matrix_.begin());
  cols_ = static_cast<int>(task_data->inputs_count[1]);
  rows_ = static_cast<int>(task_data->inputs_count[2]);

  res_ = std::vector<double>(cols_ - 1, 0);
  return true;
}

bool shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::MPIGaussHorizontalSequential::ValidationImpl() {
  matrix_ = std::vector<double>(task_data->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  std::ranges::copy(tmp_ptr, tmp_ptr + task_data->inputs_count[0], matrix_.begin());
  cols_ = static_cast<int>(task_data->inputs_count[1]);
  rows_ = static_cast<int>(task_data->inputs_count[2]);
  Matrix matrix;
  matrix.cols = cols_;
  matrix.rows = rows_;

  return task_data->inputs_count[0] > 1 && rows_ == cols_ - 1 && Determinant(matrix, matrix_) != 0 &&
         MatrixRank(matrix, matrix_) == rows_;
}

bool shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::MPIGaussHorizontalSequential::RunImpl() {
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

bool shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::MPIGaussHorizontalSequential::PostProcessingImpl() {
  auto* this_matrix = reinterpret_cast<double*>(task_data->outputs[0]);
  std::ranges::copy(res_.begin(), res_.end(), this_matrix);
  return true;
}

bool shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::MPIGaussHorizontalParallel::PreProcessingImpl() {
  if (world_.rank() == 0) {
    matrix_ = std::vector<double>(task_data->inputs_count[0]);
    auto* tmp_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
    std::ranges::copy(tmp_ptr, tmp_ptr + task_data->inputs_count[0], matrix_.begin());
    cols_ = static_cast<int>(task_data->inputs_count[1]);
    rows_ = static_cast<int>(task_data->inputs_count[2]);

    res_ = std::vector<double>(cols_ - 1, 0);
  }
  return true;
}

bool shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::MPIGaussHorizontalParallel::ValidationImpl() {
  if (world_.rank() == 0) {
    Matrix matrix;
    matrix_ = std::vector<double>(task_data->inputs_count[0]);
    auto* tmp_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
    std::ranges::copy(tmp_ptr, tmp_ptr + task_data->inputs_count[0], matrix_.begin());
    cols_ = static_cast<int>(task_data->inputs_count[1]);
    rows_ = static_cast<int>(task_data->inputs_count[2]);
    matrix.cols = static_cast<int>(task_data->inputs_count[1]);
    matrix.rows = static_cast<int>(task_data->inputs_count[2]);

    return task_data->inputs_count[0] > 1 && rows_ == cols_ - 1 && Determinant(matrix, matrix_) != 0 &&
           MatrixRank(matrix, matrix_) == rows_;
  }
  return true;
}

void shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::BroadcastMatrixSize(boost::mpi::communicator& world,
                                                                                    int& rows, int& cols) {
  broadcast(world, cols, 0);
  broadcast(world, rows, 0);
}

std::vector<int> shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::ComputeRowDistribution(
    boost::mpi::communicator& world, int rows) {
  std::vector<int> row_num(world.size());
  int delta = rows / world.size();
  int remainder = rows % world.size();
  if (world.rank() < remainder) {
    delta++;
  }
  boost::mpi::gather(world, delta, row_num, 0);
  return row_num;
}

void shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::DistributeMatrix(boost::mpi::communicator& world,
                                                                                 const std::vector<int>& row_num,
                                                                                 int delta, int cols,
                                                                                 std::vector<double>& matrix) {
  if (world.rank() == 0) {
    std::vector<double> send_matrix((delta * cols));
    for (int proc = 1; proc < world.size(); ++proc) {
      for (int i = 0; i < row_num[proc]; ++i) {
        for (int j = 0; j < cols; ++j) {
          send_matrix[(i * cols) + j] = matrix[((proc + (world.size() * i)) * cols) + j];
        }
      }
      world.send(proc, 0, send_matrix.data(), row_num[proc] * cols);
    }
  }
}

void shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::ReceiveMatrix(boost::mpi::communicator& world,
                                                                              int delta, int cols,
                                                                              std::vector<double>& local_matrix,
                                                                              std::vector<double>& matrix) {
  local_matrix.resize(delta * cols);
  if (world.rank() == 0) {
    for (int i = 0; i < delta; ++i) {
      for (int j = 0; j < cols; ++j) {
        local_matrix[(i * cols) + j] = matrix[(i * cols * world.size()) + j];
      }
    }
  } else {
    world.recv(0, 0, local_matrix.data(), delta * cols);
  }
}

void shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::ForwardElimination(boost::mpi::communicator& world,
                                                                                   Matrix matrix, Vector& vector) {
  std::vector<double> pivot{static_cast<double>(matrix.cols)};
  int r = 0;
  for (int i = 0; i < matrix.rows - 1; ++i) {
    if (i == vector.row[r]) {
      for (int j = 0; j < matrix.cols; ++j) {
        pivot[j] = vector.local_matrix[(r * matrix.cols) + j];
      }
      broadcast(world, pivot.data(), matrix.cols, world.rank());
      r++;
    } else {
      broadcast(world, pivot.data(), matrix.cols, i % world.size());
    }
    for (int k = r; k < matrix.delta; ++k) {
      double m = vector.local_matrix[(k * matrix.cols) + i] / pivot[i];
      for (int j = i; j < matrix.cols; ++j) {
        vector.local_matrix[(k * matrix.cols) + j] -= pivot[j] * m;
      }
    }
  }
}

void shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::BackSubstitution(boost::mpi::communicator& world,
                                                                                 Matrix matrix, Vector& vector) {
  int r = matrix.delta - 1;
  for (int i = matrix.rows - 1; i > 0; --i) {
    if (r >= 0 && i == vector.row[r]) {
      vector.local_res[i] /= vector.local_matrix[(r * matrix.cols) + i];
      broadcast(world, vector.local_res[i], world.rank());
      r--;
    } else {
      broadcast(world, vector.local_res[i], i % world.size());
    }
    if (r >= 0) {
      for (int j = 0; j <= r; ++j) {
        vector.local_res[static_cast<size_t>(vector.row[j])] -=
            vector.local_matrix[(j * matrix.cols) + i] * vector.local_res[i];
      }
    }
    if (world.rank() == 0) {
      vector.local_res[0] /= vector.local_matrix[0];
      vector.res = vector.local_res;
    }
  }
}

bool shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::MPIGaussHorizontalParallel::RunImpl() {
  BroadcastMatrixSize(world_, rows_, cols_);
  std::vector<int> row_num = ComputeRowDistribution(world_, rows_);
  DistributeMatrix(world_, row_num, row_num[world_.rank()], cols_, matrix_);
  ReceiveMatrix(world_, row_num[world_.rank()], cols_, local_matrix_, matrix_);

  std::vector<double> row(row_num[world_.rank()]);
  for (int i = 0; i < static_cast<int>(row.size()); ++i) {
    row[i] = world_.rank() + world_.size() * i;
  }

  Matrix matrix;
  matrix.cols = cols_;
  matrix.rows = rows_;
  matrix.delta = row_num[world_.rank()];

  Vector vector;
  vector.local_matrix = local_matrix_;
  vector.local_res = local_res_;
  vector.res = res_;
  vector.row = row;

  ForwardElimination(world_, matrix, vector);

  local_res_.resize(cols_ - 1, 0);
  vector.local_res = local_res_;

  BackSubstitution(world_, matrix, vector);
  return true;
}

bool shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::MPIGaussHorizontalParallel::PostProcessingImpl() {
  if (world_.rank() == 0) {
    auto* this_matrix = reinterpret_cast<double*>(task_data->outputs[0]);
    std::ranges::copy(res_.begin(), res_.end(), this_matrix);
  }
  return true;
}