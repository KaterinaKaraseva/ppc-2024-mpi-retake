#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cstring>
#include <memory>
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

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> data_seq_;
  std::vector<int> labels_seq_;
  int m_seq_, n_seq_;
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  void MPIBfs(int* p_local_image, int local_pixel_count, int* p_local_labels, int curr_label);
  [[nodiscard]] bool IsValidMPI(int nr, int local_pixel_count, int nc) const;
  void BFSCheck(const int* p_local_image, int curr_label, int* p_local_labels, int local_pixel_count, Point cp,
                std::queue<Point> bfs_queue) const;
  [[nodiscard]] std::vector<int> MakeMPIResult(std::vector<int> global_labels) const;
  void MergeLabels(int* p_global, std::vector<int> parent);

 private:
  std::vector<int> data_;
  std::vector<int> labels_;
  int m_, n_;
  boost::mpi::communicator world_;
};
}  // namespace solovev_a_binary_image_marking