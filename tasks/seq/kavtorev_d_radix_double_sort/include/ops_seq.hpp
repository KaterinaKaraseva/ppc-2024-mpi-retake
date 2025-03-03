#pragma once

#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace kavtorev_d_radix_double_sort {

class RadixSortSequential : public ppc::core::Task {
 public:
  explicit RadixSortSequential(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> data_;
  int n_ = 0;

  static void RadixSortDoubles(std::vector<double>& data);
  static void RadixSortUint64(std::vector<uint64_t>& keys);
};

}  // namespace kavtorev_d_radix_double_sort