#include "mpi/deryabin_m_cannons_algorithm/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/collectives/broadcast.hpp>
#include <cmath>
#include <vector>

bool deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskSequential::PreProcessingImpl() {
  input_matrix_A_ = std::vector<double>(task_data->inputs_count[0]);
  input_matrix_B_ = std::vector<double>(task_data->inputs_count[1]);
  auto* tmp_ptr_a = reinterpret_cast<double*>(task_data->inputs[0]);
  auto* tmp_ptr_b = reinterpret_cast<double*>(task_data->inputs[1]);
  std::copy(tmp_ptr_a, tmp_ptr_a + task_data->inputs_count[0], input_matrix_A_.begin());
  std::copy(tmp_ptr_b, tmp_ptr_b + task_data->inputs_count[1], input_matrix_B_.begin());
  output_matrix_C_ = std::vector<double>(input_matrix_A_.size());
  return true;
}

bool deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskSequential::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->inputs_count[1] &&
         task_data->inputs_count[1] == pow((unsigned short)sqrt(task_data->inputs_count[0]), 2) &&
         task_data->outputs_count[0] == 1;
}

bool deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskSequential::RunImpl() {
  unsigned short i = 0;
  unsigned short j = 0;
  unsigned short count = 0;
  auto dimension = (unsigned short)sqrt(static_cast<unsigned short>(input_matrix_A_.size()));
  while (i != dimension) {
    j = 0;
    while (j != dimension) {
      count = 0;
      while (count != dimension) {
        output_matrix_C_[(i * dimension) + j] +=
            input_matrix_A_[(i * dimension) + count] * input_matrix_B_[(count * dimension) + j];
        count++;
      }
      j++;
    }
    i++;
  }
  return true;
}

bool deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskSequential::PostProcessingImpl() {
  reinterpret_cast<std::vector<double>*>(task_data->outputs[0])[0] = output_matrix_C_;
  return true;
}

bool deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskParallel::PreProcessingImpl() {
  if (world_.rank() == 0) {
    input_matrix_A_ = std::vector<double>(task_data->inputs_count[0]);
    input_matrix_B_ = std::vector<double>(task_data->inputs_count[1]);
    auto* tmp_ptr_a = reinterpret_cast<double*>(task_data->inputs[0]);
    auto* tmp_ptr_b = reinterpret_cast<double*>(task_data->inputs[1]);
    std::copy(tmp_ptr_a, tmp_ptr_a + task_data->inputs_count[0], input_matrix_A_.begin());
    std::copy(tmp_ptr_b, tmp_ptr_b + task_data->inputs_count[1], input_matrix_B_.begin());
  }
  return true;
}

bool deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskParallel::ValidationImpl() {
  if (world_.rank() == 0) {
    return task_data->inputs_count[0] == task_data->inputs_count[1] &&
           task_data->inputs_count[1] == pow((unsigned short)sqrt(task_data->inputs_count[0]), 2) &&
           task_data->outputs_count[0] == 1;
  }
  return true;
}

void deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskParallel::HandleTrivialCase() {
  if (world_.rank() == 0) {
    dimension_ = static_cast<unsigned short>(std::sqrt(static_cast<unsigned short>(input_matrix_A_.size())));
    output_matrix_C_.resize(dimension_ * dimension_, 0.0);
    for (unsigned short i = 0; i < dimension_; i++) {
      for (unsigned short j = 0; j < dimension_; j++) {
        for (unsigned short k = 0; k < dimension_; k++) {
          output_matrix_C_[(i * dimension_) + j] +=
              input_matrix_A_[(i * dimension_) + k] * input_matrix_B_[(k * dimension_) + j];
        }
      }
    }
  }
}

void deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskParallel::InitializeAndBroadcastParams() {
  if (world_.rank() == 0) {
    dimension_ = static_cast<unsigned short>(std::sqrt(static_cast<unsigned short>(input_matrix_A_.size())));
    block_rows_columns_ = static_cast<unsigned short>(std::sqrt(static_cast<unsigned short>(world_.size())));
    block_dimension_ = dimension_ / block_rows_columns_;
  }
  boost::mpi::broadcast(world_, dimension_, 0);
  boost::mpi::broadcast(world_, block_dimension_, 0);
  boost::mpi::broadcast(world_, block_rows_columns_, 0);
}

void deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskParallel::SendMatrixAData(unsigned short i,
                                                                                        unsigned short j,
                                                                                        unsigned short k,
                                                                                        int destination_proc) {
  if (i == 0) {
    world_.send(destination_proc, 0,
                input_matrix_A_.data() + ((i * block_dimension_ + k) * dimension_) + (j * block_dimension_),
                block_dimension_);
  } else {
    if (static_cast<int>((i * block_rows_columns_) + j - i) < static_cast<int>(i * block_rows_columns_)) {
      world_.send(destination_proc + block_rows_columns_ - i, 0,
                  input_matrix_A_.data() + ((i * block_dimension_ + k) * dimension_) + (j * block_dimension_),
                  block_dimension_);
    } else {
      world_.send(destination_proc - i, 0,
                  input_matrix_A_.data() + ((i * block_dimension_ + k) * dimension_) + (j * block_dimension_),
                  block_dimension_);
    }
  }
}

void deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskParallel::SendMatrixBData(unsigned short i,
                                                                                        unsigned short j,
                                                                                        unsigned short k,
                                                                                        int destination_proc) {
  if (j == 0) {
    world_.send(destination_proc, 1,
                input_matrix_B_.data() + ((i * block_dimension_ + k) * dimension_) + (j * block_dimension_),
                block_dimension_);
  } else {
    if ((static_cast<int>((i - j) * block_rows_columns_) + j) < 0) {
      world_.send(((i + block_rows_columns_ - j) * block_rows_columns_) + j, 1,
                  input_matrix_B_.data() + ((i * block_dimension_ + k) * dimension_) + (j * block_dimension_),
                  block_dimension_);
    } else {
      world_.send(((i - j) * block_rows_columns_) + j, 1,
                  input_matrix_B_.data() + (((i * block_dimension_) + k) * dimension_) + (j * block_dimension_),
                  block_dimension_);
    }
  }
}

void deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskParallel::DistributeDataAcrossProcesses() {
  for (unsigned short i = 0; i < block_rows_columns_; ++i) {
    for (unsigned short j = 0; j < block_rows_columns_; ++j) {
      if (i == 0 && j == 0) {
        continue;
      }
      for (unsigned short k = 0; k < block_dimension_; ++k) {
        int destination_proc = (i * block_rows_columns_) + j;
        SendMatrixAData(i, j, k, destination_proc);
        SendMatrixBData(i, j, k, destination_proc);
      }
    }
  }
}

void deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskParallel::PrepareLocalMatrices() {
  for (unsigned short k = 0; k < block_dimension_; ++k) {
    std::copy(input_matrix_A_.data() + (k * dimension_), input_matrix_A_.data() + (k * dimension_) + block_dimension_,
              local_input_matrix_A_.begin() + (k * block_dimension_));
    std::copy(input_matrix_B_.data() + (k * dimension_), input_matrix_B_.data() + (k * dimension_) + block_dimension_,
              local_input_matrix_B_.begin() + (k * block_dimension_));
  }
}

void deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskParallel::DistributeDataIfRoot() {
  PrepareLocalMatrices();
  DistributeDataAcrossProcesses();
}

void deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskParallel::ReceiveDataIfNotRoot() {
  for (unsigned short k = 0; k < block_dimension_; ++k) {
    world_.recv(0, 0, local_input_matrix_A_.data() + (k * block_dimension_), block_dimension_);
    world_.recv(0, 1, local_input_matrix_B_.data() + (k * block_dimension_), block_dimension_);
  }
}

void deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskParallel::MultiplyLocalBlocks() {
  for (unsigned short i = 0; i < block_dimension_; ++i) {
    for (unsigned short j = 0; j < block_dimension_; ++j) {
      for (unsigned short k = 0; k < block_dimension_; ++k) {
        local_output_matrix_C_[(i * block_dimension_) + j] +=
            local_input_matrix_A_[(i * block_dimension_) + k] * local_input_matrix_B_[(k * block_dimension_) + j];
      }
    }
  }
}

void deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskParallel::PerformCannonShifts() {
  for (unsigned short p = 1; p < block_rows_columns_; ++p) {
    if (block_rows_columns_ != 0 && (world_.rank() % block_rows_columns_ == 0)) {
      world_.send(world_.rank() + block_rows_columns_ - 1, 2, local_input_matrix_A_.data(),
                  block_dimension_ * block_dimension_);
    } else {
      world_.send(world_.rank() - 1, 3, local_input_matrix_A_.data(), block_dimension_ * block_dimension_);
    }
    if (world_.rank() < block_rows_columns_) {
      world_.send(world_.rank() + (block_rows_columns_ * (block_rows_columns_ - 1)), 4, local_input_matrix_B_.data(),
                  block_dimension_ * block_dimension_);
    } else {
      world_.send(world_.rank() - block_rows_columns_, 5, local_input_matrix_B_.data(),
                  block_dimension_ * block_dimension_);
    }
    if (block_rows_columns_ != 0 && ((world_.rank() + 1) % block_rows_columns_ == 0)) {
      world_.recv(world_.rank() - block_rows_columns_ + 1, 2, local_input_matrix_A_.data(),
                  block_dimension_ * block_dimension_);
    } else {
      world_.recv(world_.rank() + 1, 3, local_input_matrix_A_.data(), block_dimension_ * block_dimension_);
    }
    if (world_.rank() >= block_rows_columns_ * (block_rows_columns_ - 1)) {
      world_.recv(world_.rank() - (block_rows_columns_ * (block_rows_columns_ - 1)), 4, local_input_matrix_B_.data(),
                  block_dimension_ * block_dimension_);
    } else {
      world_.recv(world_.rank() + block_rows_columns_, 5, local_input_matrix_B_.data(),
                  block_dimension_ * block_dimension_);
    }
    for (unsigned short i = 0; i < block_dimension_; ++i) {
      for (unsigned short j = 0; j < block_dimension_; ++j) {
        for (unsigned short k = 0; k < block_dimension_; ++k) {
          local_output_matrix_C_[(i * block_dimension_) + j] +=
              local_input_matrix_A_[(i * block_dimension_) + k] * local_input_matrix_B_[(k * block_dimension_) + j];
        }
      }
    }
  }
}

void deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskParallel::GatherResults() {
  if (world_.rank() != 0) {
    for (unsigned short block_row = 0; block_row < block_dimension_; ++block_row) {
      world_.send(0, 0, local_output_matrix_C_.data() + (block_row * block_dimension_), block_dimension_);
    }
  } else {
    for (int proc = 1; proc < world_.size(); ++proc) {
      for (unsigned short block_row = 0; block_row < block_dimension_; ++block_row) {
        std::copy(local_output_matrix_C_.begin() + (block_row * block_dimension_),
                  local_output_matrix_C_.begin() + ((block_row + 1) * block_dimension_),
                  output_matrix_C_.begin() +
                      (((world_.rank() / block_rows_columns_) * block_dimension_ + block_row) * dimension_) +
                      ((world_.rank() % block_rows_columns_) * block_dimension_));
        world_.recv(proc, 0,
                    output_matrix_C_.data() +
                        ((((proc / block_rows_columns_) * block_dimension_) + block_row) * dimension_) +
                        ((proc % block_rows_columns_) * block_dimension_),
                    block_dimension_);
      }
    }
  }
}

void deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskParallel::PerformCannonAlgorithm() {
  InitializeAndBroadcastParams();
  output_matrix_C_.resize(dimension_ * dimension_, 0.0);
  local_input_matrix_A_.resize(block_dimension_ * block_dimension_, 0.0);
  local_input_matrix_B_.resize(block_dimension_ * block_dimension_, 0.0);
  local_output_matrix_C_.resize(block_dimension_ * block_dimension_, 0.0);
  if (world_.rank() == 0) {
    DistributeDataIfRoot();
  } else {
    ReceiveDataIfNotRoot();
  }
  MultiplyLocalBlocks();
  PerformCannonShifts();
  GatherResults();
}

bool deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskParallel::RunImpl() {
  if (world_.size() != 1 && world_.size() == pow((unsigned short)sqrt(world_.size()), 2) &&
      static_cast<unsigned short>(std::sqrt(static_cast<unsigned short>(input_matrix_A_.size()))) %
              static_cast<unsigned short>(std::sqrt(world_.size())) ==
          0) {
    PerformCannonAlgorithm();
  } else {
    HandleTrivialCase();
  }
  world_.barrier();
  return true;
}

bool deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskParallel::PostProcessingImpl() {
  if (world_.rank() == 0) {
    reinterpret_cast<std::vector<double>*>(task_data->outputs[0])[0] = output_matrix_C_;
  }
  output_matrix_C_.clear();
  return true;
}
