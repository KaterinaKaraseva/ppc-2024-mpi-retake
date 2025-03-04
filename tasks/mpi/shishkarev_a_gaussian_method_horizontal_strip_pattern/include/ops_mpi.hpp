#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi {

struct Matrix {
  int rows;
  int cols;
  int delta;
};

struct Vector {
  std::vector<double> local_matrix;
  std::vector<double> local_res;
  std::vector<double> res;
  std::vector<double> row;
};

int MatrixRank(Matrix matrix, std::vector<double> a);

double Determinant(Matrix matrix, std::vector<double> a);

std::vector<double> GetRandomMatrix(int sz);

bool IsSingular(const std::vector<double>& matrix, Matrix mat);

double AxB(int n, int m, std::vector<double> a, std::vector<double> res);

void BroadcastMatrixSize(boost::mpi::communicator& world, int& rows, int& cols);

std::vector<int> ComputeRowDistribution(boost::mpi::communicator& world, int rows);

void DistributeMatrix(boost::mpi::communicator& world, const std::vector<int>& row_num, int delta, int cols,
                      std::vector<double>& matrix);

void ReceiveMatrix(boost::mpi::communicator& world, int delta, int cols, std::vector<double>& local_matrix,
                   std::vector<double>& matrix);

void ForwardElimination(boost::mpi::communicator& world, Matrix matrix, Vector& vector);

void BackSubstitution(boost::mpi::communicator& world, Matrix matrix, Vector& vector);

class MPIGaussHorizontalSequential : public ppc::core::Task {
 public:
  explicit MPIGaussHorizontalSequential(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> matrix_, res_;
  int rows_{}, cols_{};
};

class MPIGaussHorizontalParallel : public ppc::core::Task {
 public:
  explicit MPIGaussHorizontalParallel(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> matrix_, local_matrix_, res_, local_res_;
  int rows_{}, cols_{};
  boost::mpi::communicator world_;
};

}  // namespace shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi