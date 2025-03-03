#include "seq/strakhov_a_char_freq_counter/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>
//  Sequential

bool strakhov_a_char_freq_counter_seq::CharFreqCounterSeq::PreProcessingImpl() {
  char *tmp = reinterpret_cast<char *>(task_data->inputs[0]);
  input_ = std::vector<signed char>(task_data->inputs_count[0]);
  for (size_t i = 0; i < task_data->inputs_count[0]; i++) {
    input_[i] = tmp[i];
  }
  target_ = *reinterpret_cast<char *>(task_data->inputs[1]);
  result_ = 0;
  return true;
}

bool strakhov_a_char_freq_counter_seq::CharFreqCounterSeq::ValidationImpl() { return task_data->inputs_count[1] == 1; }

bool strakhov_a_char_freq_counter_seq::CharFreqCounterSeq::RunImpl() {
  result_ = static_cast<int>(std::count(input_.begin(), input_.end(), target_));
  return true;
}

bool strakhov_a_char_freq_counter_seq::CharFreqCounterSeq::PostProcessingImpl() {
  reinterpret_cast<int *>(task_data->outputs[0])[0] = result_;
  return true;
}