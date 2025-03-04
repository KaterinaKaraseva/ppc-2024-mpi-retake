#include "mpi/strakhov_a_char_freq_counter/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/reduce.hpp>
#include <boost/mpi/collectives/scatter.hpp>
#include <boost/mpi/collectives/scatterv.hpp>
#include <cstddef>
#include <functional>
#include <vector>
//  Sequential

bool strakhov_a_char_freq_counter_mpi::CharFreqCounterSeq::PreProcessingImpl() {
  auto *tmp = reinterpret_cast<signed char *>(task_data->inputs[0]);
  input_ = std::vector<signed char>(task_data->inputs_count[0]);
  for (size_t i = 0; i < task_data->inputs_count[0]; i++) {
    input_[i] = tmp[i];
  }
  target_ = *reinterpret_cast<char *>(task_data->inputs[1]);
  result_ = 0;
  return true;
}

bool strakhov_a_char_freq_counter_mpi::CharFreqCounterSeq::ValidationImpl() {
  return (task_data->inputs_count[1] == 1);
}

bool strakhov_a_char_freq_counter_mpi::CharFreqCounterSeq::RunImpl() {
  result_ = static_cast<int>(std::count(input_.begin(), input_.end(), target_));
  return true;
}

bool strakhov_a_char_freq_counter_mpi::CharFreqCounterSeq::PostProcessingImpl() {
  reinterpret_cast<int *>(task_data->outputs[0])[0] = result_;
  return true;
}

//  Parallel

bool strakhov_a_char_freq_counter_mpi::CharFreqCounterPar::PreProcessingImpl() {
  if (world_.rank() == 0) {
    input_.resize(task_data->inputs_count[0]);
    auto *tmp = reinterpret_cast<signed char *>(task_data->inputs[0]);
    input_ = std::vector<signed char>(task_data->inputs_count[0]);
    for (size_t i = 0; i < task_data->inputs_count[0]; i++) {
      input_[i] = tmp[i];
    }
    target_ = *reinterpret_cast<char *>(task_data->inputs[1]);
  }

  result_ = 0;
  local_result_ = 0;
  return true;
}

bool strakhov_a_char_freq_counter_mpi::CharFreqCounterPar::ValidationImpl() {
  if (world_.rank() == 0) {
    return (task_data->inputs_count[1] == 1);
  }
  return true;
}

bool strakhov_a_char_freq_counter_mpi::CharFreqCounterPar::RunImpl() {
  int local_input_size = 0;
  std::vector<int> send_counts;
  std::vector<int> displacements{};
  int rank = world_.rank();

  broadcast(world_, target_, 0);

  if (rank == 0) {
    int input_length = static_cast<int>(task_data->inputs_count[0]);
    int world_size = world_.size();
    int segment = input_length / world_size;
    int excess = input_length % world_size;

    send_counts.resize(world_size);
    for (int i = 0; i < world_size; ++i) {
      send_counts[i] = segment + (i < excess ? 1 : 0);
    }

    displacements.resize(world_size);
    if (!(displacements.empty())) {
      displacements[0] = 0;
      for (int i = 1; i < world_size; ++i) {
        displacements[i] = displacements[i - 1] + send_counts[i - 1];
      }
    }
  } else {
    input_.clear();
  }

  boost::mpi::scatter(world_, send_counts, local_input_size, 0);
  local_input_.resize(local_input_size);
  boost::mpi::scatterv(world_, input_.data(), send_counts, displacements, local_input_.data(), local_input_size, 0);
  local_result_ = static_cast<int>(std::count(local_input_.begin(), local_input_.end(), target_));

  boost::mpi::reduce(world_, local_result_, result_, std::plus<>(), 0);

  return true;
}

bool strakhov_a_char_freq_counter_mpi::CharFreqCounterPar::PostProcessingImpl() {
  if (world_.rank() == 0) {
    reinterpret_cast<int *>(task_data->outputs[0])[0] = result_;
  }
  return true;
}
