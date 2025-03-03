#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace mezhuev_m_most_different_neighbor_elements_seq {

class MostDifferentNeighborElements : public ppc::core::Task {
 public:
  explicit MostDifferentNeighborElements(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  [[nodiscard]] const std::vector<int>& GetResult() const { return result_; }

 private:
  std::vector<int> input_;
  std::vector<int> result_;
};

}  // namespace mezhuev_m_most_different_neighbor_elements_seq