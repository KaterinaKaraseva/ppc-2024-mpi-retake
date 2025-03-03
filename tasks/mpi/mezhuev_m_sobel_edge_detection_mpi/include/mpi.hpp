#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace mezhuev_m_sobel_edge_detection_mpi {

class SobelEdgeDetection : public ppc::core::Task {
 public:
  SobelEdgeDetection(boost::mpi::communicator& world, std::shared_ptr<ppc::core::TaskData> task_data)
      : Task(std::move(task_data)), world_(world) {}

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  [[nodiscard]] const std::vector<int>& GetGradientX() const { return gradient_x_; }
  [[nodiscard]] const std::vector<int>& GetGradientY() const { return gradient_y_; }

 private:
  boost::mpi::communicator& world_;
  std::vector<int> gradient_x_;
  std::vector<int> gradient_y_;
};

}  // namespace mezhuev_m_sobel_edge_detection_mpi