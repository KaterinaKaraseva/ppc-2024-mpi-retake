#include "mpi/chernova_n_word_count/include/ops_mpi.hpp"

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/reduce.hpp>
#include <cstddef>
#include <functional>
#include <string>
#include <vector>

std::vector<char> chernova_n_word_count_mpi::CleanString(const std::vector<char>& input) {
  std::string result;
  std::string str(input.begin(), input.end());

  std::string::size_type pos = 0;
  while ((pos = str.find("  ", pos)) != std::string::npos) {
    str.erase(pos, 1);
  }

  pos = 0;
  while ((pos = str.find(" - ", pos)) != std::string::npos) {
    str.erase(pos, 2);
  }

  pos = 0;
  if (str[pos] == ' ') {
    str.erase(pos, 1);
  }

  pos = str.size() - 1;
  if (str[pos] == ' ') {
    str.erase(pos, 1);
  }

  result.assign(str.begin(), str.end());
  return {result.begin(), result.end()};
}

bool chernova_n_word_count_mpi::TestMPITaskSequential::PreProcessingImpl() {
  input_ = std::vector<char>(task_data->inputs_count[0]);
  space_count_ = 0;
  auto* tmp_ptr = reinterpret_cast<char*>(task_data->inputs[0]);
  for (unsigned i = 0; i < task_data->inputs_count[0]; i++) {
    input_[i] = tmp_ptr[i];
  }
  if (!input_.empty()) {
    input_ = CleanString(input_);
  }
  return true;
}

bool chernova_n_word_count_mpi::TestMPITaskSequential::ValidationImpl() {
  return task_data->inputs_count[0] > 0 && task_data->outputs_count[0] == 1;
}

bool chernova_n_word_count_mpi::TestMPITaskSequential::RunImpl() {
  if (input_.empty()) {
    space_count_ = -1;
  }
  for (std::size_t i = 0; i < input_.size(); i++) {
    char c = input_[i];
    if (c == ' ') {
      space_count_++;
    }
  }
  return true;
}

bool chernova_n_word_count_mpi::TestMPITaskSequential::PostProcessingImpl() {
  if (task_data->outputs[0] == nullptr) {
    return false;
  }
  if (input_.empty()) {
    reinterpret_cast<int*>(task_data->outputs[0])[0] = 0;
  } else {
    reinterpret_cast<int*>(task_data->outputs[0])[0] = space_count_ + 1;
  }
  return true;
}

bool chernova_n_word_count_mpi::TestMPITaskParallel::PreProcessingImpl() {
  if (world_.rank() == 0) {
    input_ = std::vector<char>(task_data->inputs_count[0]);
    space_count_ = 0;
    auto* tmp_ptr = reinterpret_cast<char*>(task_data->inputs[0]);
    for (std::size_t i = 0; i < task_data->inputs_count[0]; i++) {
      input_[i] = tmp_ptr[i];
    }
    if (!input_.empty()) {
      input_ = CleanString(input_);
    }
    task_data->inputs_count[0] = input_.size();
  }
  return true;
}

bool chernova_n_word_count_mpi::TestMPITaskParallel::ValidationImpl() {
  if (world_.rank() == 0) {
    return task_data->inputs_count[0] > 0 && task_data->outputs_count[0] == 1;
  }
  return true;
}

bool chernova_n_word_count_mpi::TestMPITaskParallel::RunImpl() {
  unsigned long total_size = 0;
  if (world_.rank() == 0) {
    total_size = input_.size();
    part_size_ = static_cast<int>(task_data->inputs_count[0] / world_.size());
  }
  boost::mpi::broadcast(world_, part_size_, 0);
  boost::mpi::broadcast(world_, total_size, 0);
  if (total_size == 0) {
    space_count_ = -1;
    boost::mpi::broadcast(world_, space_count_, 0);
    return true;
  }

  unsigned long start_pos = world_.rank() * part_size_;
  size_t actual_part_size = (start_pos + part_size_ <= total_size) ? part_size_ : (total_size - start_pos);

  local_input_.resize(actual_part_size);

  if (world_.rank() == 0) {
    for (int proc = 1; proc < world_.size(); proc++) {
      unsigned long proc_start_pos = proc * part_size_;
      unsigned long proc_part_size =
          (proc_start_pos + part_size_ <= total_size) ? part_size_ : (total_size - proc_start_pos);
      if (proc_part_size > 0) {
        world_.send(proc, 0, input_.data() + proc_start_pos, static_cast<int>(proc_part_size));
      }
    }
    local_input_.assign(input_.begin(), input_.begin() + static_cast<std::ptrdiff_t>(actual_part_size));
  } else {
    if (actual_part_size > 0) {
      world_.recv(0, 0, local_input_.data(), static_cast<int>(actual_part_size));
    }
  }
  local_space_count_ = 0;
  for (std::size_t i = 0; i < local_input_.size(); ++i) {
    if (local_input_[i] == ' ') {
      local_space_count_++;
    }
  }

  boost::mpi::reduce(world_, local_space_count_, space_count_, std::plus<>(), 0);

  return true;
}

bool chernova_n_word_count_mpi::TestMPITaskParallel::PostProcessingImpl() {
  if (world_.rank() == 0) {
    if (task_data->outputs[0] == nullptr) {
      return false;
    }
    if (space_count_ == -1) {
      reinterpret_cast<int*>(task_data->outputs[0])[0] = 0;
    } else {
      reinterpret_cast<int*>(task_data->outputs[0])[0] = space_count_ + 1;
    }
  }
  return true;
}