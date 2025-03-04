#include "mpi/kavtorev_d_most_different_neighbor_elements/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/reduce.hpp>
#include <boost/mpi/operations.hpp>
#include <climits>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

bool kavtorev_d_most_different_neighbor_elements_mpi::MostDifferentNeighborElementsSeq::PreProcessingImpl() {
  auto input = std::vector<int>(task_data->inputs_count[0]);
  auto* tmp = reinterpret_cast<int*>(task_data->inputs[0]);
  std::copy(tmp, tmp + task_data->inputs_count[0], input.begin());

  input_ = std::vector<std::pair<int, int>>(input.size() - 1);

  for (size_t i = 1; i < input.size(); ++i) {
    input_[i - 1] = {std::abs(input[i] - input[i - 1]), std::min(input[i], input[i - 1])};
  }

  res_ = input_[0];

  return true;
}

bool kavtorev_d_most_different_neighbor_elements_mpi::MostDifferentNeighborElementsSeq::ValidationImpl() {
  return task_data->inputs_count[0] > 1 && task_data->outputs_count[0] == 1;
}

bool kavtorev_d_most_different_neighbor_elements_mpi::MostDifferentNeighborElementsSeq::RunImpl() {
  for (size_t i = 1; i < input_.size(); ++i) {
    if (res_.first < input_[i].first) {
      res_ = input_[i];
    }
  }
  return true;
}

bool kavtorev_d_most_different_neighbor_elements_mpi::MostDifferentNeighborElementsSeq::PostProcessingImpl() {
  reinterpret_cast<int*>(task_data->outputs[0])[0] = res_.first;
  return true;
}

bool kavtorev_d_most_different_neighbor_elements_mpi::MostDifferentNeighborElementsMpi::PreProcessingImpl() {
  res_ = {INT_MIN, -1};
  return true;
}

bool kavtorev_d_most_different_neighbor_elements_mpi::MostDifferentNeighborElementsMpi::ValidationImpl() {
  if (world_.rank() == 0) {
    return task_data->inputs_count[0] > 1 && task_data->outputs_count[0] == 1;
  }
  return true;
}

bool kavtorev_d_most_different_neighbor_elements_mpi::MostDifferentNeighborElementsMpi::RunImpl() {
  int delta_size = 0;
  if (world_.rank() == 0) {
    delta_size = static_cast<int>(task_data->inputs_count[0]) / world_.size();
    size_ = static_cast<int>(task_data->inputs_count[0]);
    if (task_data->inputs_count[0] % world_.size() > 0) {
      delta_size++;
    }
  }
  broadcast(world_, delta_size, 0);

  if (world_.rank() == 0) {
    input_ = std::vector<int>((world_.size() * delta_size) + 2, 0);
    auto* tmp = reinterpret_cast<int*>(task_data->inputs[0]);
    std::copy(tmp, tmp + task_data->inputs_count[0], input_.begin());
    for (int proc = 1; proc < world_.size(); proc++) {
      world_.send(proc, 0, input_.data() + (proc * delta_size), delta_size + 1);
    }
  }

  local_input_ = std::vector<int>(delta_size + 1);
  st_ = world_.rank() * delta_size;
  if (world_.rank() == 0) {
    local_input_ = std::vector<int>(input_.begin(), input_.begin() + delta_size + 1);
  } else {
    world_.recv(0, 0, local_input_.data(), delta_size + 1);
  }

  std::pair<int, int> local_ans = {INT_MIN, -1};
  for (size_t i = 0; (i + st_) < size_ - 1 && i < (local_input_.size() - 1); ++i) {
    std::pair<int, int> tmp = {abs(local_input_[i + 1] - local_input_[i]), i + st_};
    local_ans = std::max(local_ans, tmp);
  }
  reduce(world_, local_ans, res_, boost::mpi::maximum<std::pair<int, int>>(), 0);
  return true;
}

bool kavtorev_d_most_different_neighbor_elements_mpi::MostDifferentNeighborElementsMpi::PostProcessingImpl() {
  if (world_.rank() == 0) {
    reinterpret_cast<int*>(task_data->outputs[0])[0] = res_.first;
  }
  return true;
}