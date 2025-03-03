#include <cmath>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace mezhuev_m_sobel_edge_detection_seq {

class SobelEdgeDetectionSeq : public ppc::core::Task {
 public:
  explicit SobelEdgeDetectionSeq(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool ValidationImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int16_t> gradient_x_;
  std::vector<int16_t> gradient_y_;
};

}  // namespace mezhuev_m_sobel_edge_detection_seq