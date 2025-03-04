#include "seq/karaseva_e_reduce/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <numeric>
#include <vector>

bool karaseva_e_reduce_seq::TestTaskSequential::PreProcessingImpl() {
  if (!task_data || task_data->inputs.empty() || task_data->outputs.empty()) {
    return false;
  }

  unsigned int input_size = task_data->inputs_count[0];
  auto *in_ptr = reinterpret_cast<int *>(task_data->inputs[0]);
  if (in_ptr == nullptr) {
    return false;
  }
  input_.assign(in_ptr, in_ptr + input_size);

  unsigned int output_size = task_data->outputs_count[0];
  output_.resize(output_size, 0);

  rc_size_ = static_cast<int>(std::sqrt(static_cast<double>(input_size)));

  return true;
}

bool karaseva_e_reduce_seq::TestTaskSequential::ValidationImpl() {
  if (!task_data) {
    return false;
  }

  if (task_data->outputs_count.empty() || task_data->inputs_count.empty()) {
    return false;
  }

  if (task_data->outputs_count[0] != 1) {
    return false;
  }

  auto expected_input_size = static_cast<size_t>(std::sqrt(static_cast<double>(task_data->inputs_count[0])));

  return task_data->inputs_count[0] == expected_input_size * expected_input_size;
}

bool karaseva_e_reduce_seq::TestTaskSequential::RunImpl() {
  int sum = std::accumulate(input_.begin(), input_.end(), 0);
  output_[0] = sum;
  return true;
}

bool karaseva_e_reduce_seq::TestTaskSequential::PostProcessingImpl() {
  if (!task_data || task_data->outputs.empty() || task_data->outputs[0] == nullptr) {
    return false;
  }

  auto *out_ptr = reinterpret_cast<int *>(task_data->outputs[0]);
  if (out_ptr == nullptr) {
    return false;
  }

  out_ptr[0] = output_[0];
  return true;
}