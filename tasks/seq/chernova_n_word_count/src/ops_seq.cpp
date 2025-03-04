#include "seq/chernova_n_word_count/include/ops_seq.hpp"

#include <string>
#include <vector>

std::vector<char> chernova_n_word_count_seq::CleanString(const std::vector<char>& input) {
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

bool chernova_n_word_count_seq::TestTaskSequential::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<char*>(task_data->inputs[0]);
  input_ = std::vector<char>(in_ptr, in_ptr + input_size);

  input_ = CleanString(input_);
  spaceCount_ = 0;

  return true;
}

bool chernova_n_word_count_seq::TestTaskSequential::ValidationImpl() {
  return task_data->inputs_count[0] > 0 && task_data->outputs_count[0] == 1;
}

bool chernova_n_word_count_seq::TestTaskSequential::RunImpl() {
  if (input_.empty()) {
    spaceCount_ = -1;
  } else {
    for (char c : input_) {
      if (c == ' ') {
        spaceCount_++;
      }
    }
  }
  return true;
}

bool chernova_n_word_count_seq::TestTaskSequential::PostProcessingImpl() {
  if (task_data->outputs[0] == nullptr) {
    return false;
  }
  if (input_.empty()) {
    reinterpret_cast<int*>(task_data->outputs[0])[0] = 0;
  } else {
    reinterpret_cast<int*>(task_data->outputs[0])[0] = spaceCount_ + 1;
  }
  return true;
}