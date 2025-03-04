#include "mpi/komshina_d_num_of_alternating_signs_of_values/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>
#include <cstddef>
#include <vector>

bool komshina_d_num_of_alternations_signs_mpi::TestTaskMPI::PreProcessingImpl() {
  if (world_.rank() == 0) {
    unsigned int input_size = task_data->inputs_count[0];
    auto *in_ptr = reinterpret_cast<int *>(task_data->inputs[0]);
    input_ = std::vector<int>(in_ptr, in_ptr + input_size);
  }
  result_ = 0;
  return true;
}

bool komshina_d_num_of_alternations_signs_mpi::TestTaskMPI::ValidationImpl() {
  if (world_.rank() == 0) {
    return task_data->inputs_count[0] >= 2 && task_data->outputs_count[0] == 1;
  }
  return true;
}

bool komshina_d_num_of_alternations_signs_mpi::TestTaskMPI::RunImpl() {
  unsigned int delta = 0;
  unsigned int remainder = 0;

  if (world_.rank() == 0) {
    const auto input_size = static_cast<std::size_t>(task_data->inputs_count[0]);
    delta = input_size / world_.size();
    remainder = input_size % world_.size();
  }

  MPI_Bcast(&delta, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
  MPI_Bcast(&remainder, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

  std::vector<int> local_input;
  const auto local_size = static_cast<std::size_t>(delta) + (world_.rank() == world_.size() - 1 ? remainder : 0);

  if (world_.rank() == 0) {
    local_input.assign(input_.begin(), input_.begin() + static_cast<int>(local_size));
    for (int proc = 1; proc < world_.size(); ++proc) {
      const std::size_t send_count = delta + (proc == world_.size() - 1 ? remainder : 0);
      MPI_Send(input_.data() + (proc * static_cast<int>(delta)), static_cast<int>(send_count), MPI_INT, proc, 0,
               MPI_COMM_WORLD);
    }
  } else {
    local_input.resize(local_size);
    MPI_Recv(local_input.data(), static_cast<int>(local_size), MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  int local_count = 0;
  if (local_input.size() > 1) {
    for (std::size_t i = 1; i < local_input.size(); ++i) {
      if (local_input[i - 1] * local_input[i] < 0) {
        ++local_count;
      }
    }
  }

  if (world_.rank() > 0) {
    int prev_value = 0;
    MPI_Recv(&prev_value, 1, MPI_INT, world_.rank() - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (!local_input.empty() && prev_value * local_input[0] < 0) {
      ++local_count;
    }
  }

  if (world_.rank() < world_.size() - 1) {
    const int last_value = local_input.empty() ? 0 : local_input.back();
    MPI_Send(&last_value, 1, MPI_INT, world_.rank() + 1, 0, MPI_COMM_WORLD);
  }

  int global_count = 0;
  MPI_Reduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  if (world_.rank() == 0) {
    result_ = global_count;
  }

  return true;
}

bool komshina_d_num_of_alternations_signs_mpi::TestTaskMPI::PostProcessingImpl() {
  if (world_.rank() == 0) {
    reinterpret_cast<int *>(task_data->outputs[0])[0] = result_;
  }
  return true;
}