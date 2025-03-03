#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace deryabin_m_cannons_algorithm_mpi {

class CannonsAlgorithmMPITaskSequential : public ppc::core::Task {
 public:
  explicit CannonsAlgorithmMPITaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> input_matrix_A_;
  std::vector<double> input_matrix_B_;
  std::vector<double> output_matrix_C_;
};
class CannonsAlgorithmMPITaskParallel : public ppc::core::Task {
 public:
  explicit CannonsAlgorithmMPITaskParallel(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  // --- Приватные методы для декомпозиции ---
  void HandleTrivialCase();
  void PerformCannonAlgorithm();

  // вспомогательные под-этапы алгоритма Каннона
  void InitializeAndBroadcastParams();
  void DistributeDataIfRoot();
  void PrepareLocalMatrices();
  void DistributeDataAcrossProcesses();
  void SendMatrixAData(unsigned short i, unsigned short j, unsigned short k, int destination_proc);
  void SendMatrixBData(unsigned short i, unsigned short j, unsigned short k, int destination_proc);
  void ReceiveDataIfNotRoot();
  void MultiplyLocalBlocks();
  void PerformCannonShifts();
  void GatherResults();

  std::vector<double> input_matrix_A_, local_input_matrix_A_;
  std::vector<double> input_matrix_B_, local_input_matrix_B_;
  std::vector<double> output_matrix_C_, local_output_matrix_C_;
  unsigned short dimension_ = 0;
  unsigned short block_dimension_ = 0;
  unsigned short block_rows_columns_ = 0;
  boost::mpi::communicator world_;
};
}  // namespace deryabin_m_cannons_algorithm_mpi
