#pragma once

#include <queue>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace solovev_a_binary_image_marking {

struct Point {
  int x, y;
};

using Matrix = std::vector<int>;
using Directions = std::vector<Point>;

void Bfs(int i, int j, int label, Matrix& labels_tmp, const Matrix& data_tmp, int m_tmp, int n_tmp,
         const Directions& directions);
void ProcessNeighbor(std::queue<Point>& q, int new_x, int new_y, Matrix& labels_tmp, const Matrix& data_tmp, int label,
                     int n_tmp);
bool ShouldProcess(int i, int j, const Matrix& data_tmp, const Matrix& labels_tmp, int n_tmp);
bool IsValid(int x, int y, int m_tmp, int n_tmp);

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> data_;
  std::vector<int> labels_;
  int m_, n_;
};
}  // namespace solovev_a_binary_image_marking