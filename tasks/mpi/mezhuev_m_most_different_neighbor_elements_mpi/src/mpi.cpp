#include "mpi/mezhuev_m_most_different_neighbor_elements_mpi/include/mpi.hpp"

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/operations.hpp>
#include <climits>
#include <cmath>
#include <vector>

namespace mezhuev_m_most_different_neighbor_elements_mpi {

bool MostDifferentNeighborElements::PreProcessingImpl() {
  if (!task_data || task_data->inputs.empty() || task_data->outputs.empty()) {
    return false;
  }

  auto* tmp = reinterpret_cast<int*>(task_data->inputs[0]);
  input_ = std::vector<int>(tmp, tmp + task_data->inputs_count[0]);

  result_ = {0, 0};

  return true;
}

bool MostDifferentNeighborElements::ValidationImpl() {
  if (!task_data || task_data->inputs.empty() || task_data->outputs.empty()) {
    return false;
  }
  if (task_data->inputs_count[0] < 2 || task_data->outputs_count[0] < 2) {
    return false;
  }

  return true;
}

bool MostDifferentNeighborElements::RunImpl() {
  int rank = world_.rank();
  int size = world_.size();

  if (input_.size() < 2) {
    return false;
  }

  int delta_size = static_cast<int>(input_.size()) / size;
  int extra_elements = static_cast<int>(input_.size()) % size;

  int start_index = (rank * delta_size) + std::min(rank, extra_elements);
  int end_index = start_index + delta_size + (rank < extra_elements ? 1 : 0);

  int local_max_diff = 0;
  int local_max_index = -1;

  for (int i = start_index; i < end_index - 1; ++i) {
    int diff = std::abs(input_[i + 1] - input_[i]);
    if (diff > local_max_diff) {
      local_max_diff = diff;
      local_max_index = i;
    }
  }

  int global_max_diff = 0;
  int global_max_index = -1;
  // NOLINTNEXTLINE(misc-include-cleaner)
  boost::mpi::all_reduce(world_, local_max_diff, global_max_diff, boost::mpi::maximum<int>());
  // NOLINTNEXTLINE(misc-include-cleaner)
  boost::mpi::all_reduce(world_, local_max_index, global_max_index, boost::mpi::maximum<int>());

  if (rank == 0) {
    if (global_max_index != -1) {
      result_.first = input_[global_max_index];
      result_.second = input_[global_max_index + 1];
    } else {
      return false;
    }
  }

  return true;
}

bool MostDifferentNeighborElements::PostProcessingImpl() {
  if (world_.rank() == 0) {
    auto* output_data = reinterpret_cast<int*>(task_data->outputs[0]);
    auto* output_data2 = reinterpret_cast<int*>(task_data->outputs[1]);

    output_data[0] = result_.first;
    output_data2[0] = result_.second;
  }
  return true;
}

}  // namespace mezhuev_m_most_different_neighbor_elements_mpi
