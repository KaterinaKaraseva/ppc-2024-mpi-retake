#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace shishkarev_a_gaussian_method_horizontal_strip_pattern_seq {

struct Matrix {
  int rows;
  int cols;
  int delta;
};

int MatrixRank(Matrix matrix, std::vector<double> a);

double Determinant(Matrix matrix, std::vector<double> a);

template <class InOutType>
class MPIGaussHorizontalSequential : public ppc::core::Task {
 public:
  explicit MPIGaussHorizontalSequential(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> matrix_, res_;
  int delta_, rows_{}, cols_{};
};

}  // namespace shishkarev_a_gaussian_method_horizontal_strip_pattern_seq