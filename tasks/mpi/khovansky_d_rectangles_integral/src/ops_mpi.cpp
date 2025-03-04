#include "mpi/khovansky_d_rectangles_integral/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/reduce.hpp>
#include <boost/mpi/communicator.hpp>
#include <functional>
#include <vector>

namespace khovansky_d_rectangles_integral_mpi {

bool khovansky_d_rectangles_integral_mpi::RectanglesMpi::PreProcessingImpl() {
  if (world_.rank() == 0) {
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
  }
  return true;
}

bool khovansky_d_rectangles_integral_mpi::RectanglesMpi::ValidationImpl() {
  if (world_.rank() == 0) {
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
  }
  return true;
}

bool khovansky_d_rectangles_integral_mpi::RectanglesMpi::RunImpl() {
  broadcast(world_, num_dimensions_, 0);
  broadcast(world_, num_partitions_, 0);
  lower_limits_.resize(num_dimensions_);
  upper_limits_.resize(num_dimensions_);
  broadcast(world_, lower_limits_.data(), static_cast<int>(num_dimensions_), 0);
  broadcast(world_, upper_limits_.data(), static_cast<int>(num_dimensions_), 0);

  double local_result = 0.0;
  std::vector<double> step_size(num_dimensions_);
  unsigned long long total_points = 1;

  for (unsigned int i = 0; i < num_dimensions_; i++) {
    step_size[i] = (upper_limits_[i] - lower_limits_[i]) / num_partitions_;
    total_points *= num_partitions_;
  }

  unsigned long long chunk_size = total_points / world_.size();
  unsigned long long start_index = world_.rank() * chunk_size;
  unsigned long long end_index = (world_.rank() == world_.size() - 1) ? total_points : start_index + chunk_size;

  std::vector<int> indices(num_dimensions_);

  for (unsigned long long idx = start_index; idx < end_index; idx++) {
    unsigned long long temp_idx = idx;
    std::vector<double> point_coordinates(num_dimensions_);

    for (unsigned int j = 0; j < num_dimensions_; j++) {
      indices[j] = static_cast<int>(temp_idx % num_partitions_);
      temp_idx /= num_partitions_;
      point_coordinates[j] = lower_limits_[j] + (indices[j] + 0.5) * step_size[j];
    }

    local_result += integrand_function(point_coordinates);
  }

  for (unsigned int j = 0; j < num_dimensions_; j++) {
    local_result *= step_size[j];
  }

  boost::mpi::reduce(world_, local_result, integral_result_, std::plus<>(), 0);
  return true;
}

bool khovansky_d_rectangles_integral_mpi::RectanglesMpi::PostProcessingImpl() {
  if (world_.rank() == 0) {
    reinterpret_cast<double*>(task_data->outputs[0])[0] = integral_result_;
  }
  return true;
}

}  // namespace khovansky_d_rectangles_integral_mpi
