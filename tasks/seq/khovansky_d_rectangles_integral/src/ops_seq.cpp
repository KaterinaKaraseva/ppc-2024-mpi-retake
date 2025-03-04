#include "seq/khovansky_d_rectangles_integral/include/ops_seq.hpp"

#include <algorithm>
#include <functional>
#include <vector>

namespace khovansky_d_rectangles_integral_seq {

bool khovansky_d_rectangles_integral_seq::RectanglesSeq::PreProcessingImpl() {
  num_dimensions_ = task_data->inputs_count[0];
  lower_limits_ = std::vector<double>(num_dimensions_);
  upper_limits_ = std::vector<double>(num_dimensions_);
  integral_result_ = 0.0;
  auto* lower_bound_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  auto* upper_bound_ptr = reinterpret_cast<double*>(task_data->inputs[1]);
  num_partitions_ = task_data->inputs_count[1];
  if (lower_bound_ptr != nullptr) {
    std::copy(lower_bound_ptr, lower_bound_ptr + num_dimensions_, lower_limits_.data());
  } else {
    return false;
  }
  if (upper_bound_ptr != nullptr) {
    std::copy(upper_bound_ptr, upper_bound_ptr + num_dimensions_, upper_limits_.data());
  } else {
    return false;
  }

  return true;
}

bool khovansky_d_rectangles_integral_seq::RectanglesSeq::ValidationImpl() {
  if (!task_data) {
    return false;
  }

  if (task_data->inputs_count[0] < 1 || task_data->inputs_count[1] < 1) {
    return false;
  }

  auto* lower_bound_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  auto* upper_bound_ptr = reinterpret_cast<double*>(task_data->inputs[1]);

  if (task_data->outputs[0] == nullptr) {
    return false;
  }

  if (lower_bound_ptr == nullptr || upper_bound_ptr == nullptr) {
    return false;
  }

  for (unsigned int i = 0; i < task_data->inputs_count[0]; i++) {
    if (lower_bound_ptr[i] > upper_bound_ptr[i]) {
      return false;
    }
  }

  return true;
}

bool khovansky_d_rectangles_integral_seq::RectanglesSeq::RunImpl() {
  std::vector<double> step_size(num_dimensions_);
  unsigned long long total_points = 1;

  for (unsigned int i = 0; i < num_dimensions_; i++) {
    step_size[i] = (upper_limits_[i] - lower_limits_[i]) / num_partitions_;
    total_points *= num_partitions_;
  }

  std::vector<int> indices(num_dimensions_);

  for (unsigned long long idx = 0; idx < total_points; idx++) {
    unsigned long long temp_idx = idx;
    std::vector<double> point_coordinates(num_dimensions_);

    for (unsigned int j = 0; j < num_dimensions_; j++) {
      indices[j] = static_cast<int>(temp_idx % num_partitions_);
      temp_idx /= num_partitions_;
      point_coordinates[j] = lower_limits_[j] + (indices[j] + 0.5) * step_size[j];
    }

    integral_result_ += integrand_function(point_coordinates);
  }

  for (unsigned int j = 0; j < num_dimensions_; j++) {
    integral_result_ *= step_size[j];
  }

  return true;
}

bool khovansky_d_rectangles_integral_seq::RectanglesSeq::PostProcessingImpl() {
  reinterpret_cast<double*>(task_data->outputs[0])[0] = integral_result_;

  return true;
}

}  // namespace khovansky_d_rectangles_integral_seq
