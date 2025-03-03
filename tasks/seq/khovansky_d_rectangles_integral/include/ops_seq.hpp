#pragma once

#include <functional>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace khovansky_d_rectangles_integral_seq {

class RectanglesSeq : public ppc::core::Task {
 public:
  explicit RectanglesSeq(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  std::function<double(const std::vector<double>&)> integrand_function;

 private:
  unsigned int num_dimensions_;
  unsigned int num_partitions_;
  std::vector<double> lower_limits_;
  std::vector<double> upper_limits_;
  double integral_result_;
};

}  // namespace khovansky_d_rectangles_integral_seq
