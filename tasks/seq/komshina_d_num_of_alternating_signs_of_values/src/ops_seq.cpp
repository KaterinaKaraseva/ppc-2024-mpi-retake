#include <cmath>
#include <cstddef>
#include <vector>

#include "seq/komshina_d_num_of_alternating_signs_of_values/include/ops_sec.hpp"

bool komshina_d_num_of_alternations_signs_seq::TestTaskSequential::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto *in_ptr = reinterpret_cast<int *>(task_data->inputs[0]);
  input_ = std::vector<int>(in_ptr, in_ptr + input_size);
  result_ = 0;
  return true;
}

bool komshina_d_num_of_alternations_signs_seq::TestTaskSequential::ValidationImpl() {
  return task_data->inputs_count[0] >= 2 && task_data->outputs_count[0] == 1;
}

bool komshina_d_num_of_alternations_signs_seq::TestTaskSequential::RunImpl() {
  for (size_t i = 1; i < input_.size(); ++i) {
    if ((input_[i] * input_[i - 1]) < 0) {
      ++result_;
    }
  }

  return true;
}

bool komshina_d_num_of_alternations_signs_seq::TestTaskSequential::PostProcessingImpl() {
  reinterpret_cast<int *>(task_data->outputs[0])[0] = result_;
  return true;
}