#include "seq/mezhuev_m_most_different_neighbor_elements_seq/include/seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace mezhuev_m_most_different_neighbor_elements_seq {

bool MostDifferentNeighborElements::PreProcessingImpl() {
  if (!task_data || task_data->inputs.empty() || task_data->inputs_count.empty() || task_data->inputs_count[0] < 2) {
    return false;
  }

  int* input_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  input_.assign(input_ptr, input_ptr + task_data->inputs_count[0]);
  result_.resize(2);
  return true;
}

bool MostDifferentNeighborElements::ValidationImpl() {
  if (!task_data) {
    return false;
  }

  if (task_data->inputs.empty()) {
    return false;
  }

  if (task_data->inputs_count.empty() || task_data->inputs_count[0] < 2) {
    return false;
  }

  if (task_data->outputs.empty() || task_data->outputs_count.empty() || task_data->outputs_count[0] < 2) {
    return false;
  }

  return true;
}

bool MostDifferentNeighborElements::RunImpl() {
  if (input_.size() < 2) {
    return false;
  }

  int max_difference = 0;
  bool foundd = false;

  for (size_t i = 0; i < input_.size() - 1; ++i) {
    int current_difference = std::abs(input_[i] - input_[i + 1]);
    if (!foundd || current_difference > max_difference) {
      max_difference = current_difference;
      result_[0] = input_[i];
      result_[1] = input_[i + 1];
      foundd = true;
    }
  }

  if (!foundd) {
    result_[0] = input_[0];
    result_[1] = input_[0];
  }

  return true;
}

bool MostDifferentNeighborElements::PostProcessingImpl() {
  if (!task_data || task_data->outputs.empty() || task_data->outputs_count.empty() || task_data->outputs_count[0] < 2) {
    return false;
  }

  int* output_ptr = reinterpret_cast<int*>(task_data->outputs[0]);
  std::ranges::copy(result_, output_ptr);

  return true;
}

}  // namespace mezhuev_m_most_different_neighbor_elements_seq
