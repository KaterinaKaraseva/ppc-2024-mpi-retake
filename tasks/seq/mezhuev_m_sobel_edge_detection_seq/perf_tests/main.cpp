#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/mezhuev_m_sobel_edge_detection_seq/include/seq.hpp"

TEST(mezhuev_m_sobel_edge_detection_seq, test_pipeline_run) {
  constexpr int kWidth = 512;
  constexpr int kHeight = 512;
  constexpr int kImageSize = kWidth * kHeight;

  std::vector<uint8_t> in(kImageSize, 0);
  std::vector<uint8_t> out(kImageSize, 0);

  for (size_t y = 0; y < kHeight; ++y) {
    for (size_t x = 0; x < kWidth; ++x) {
      in[(y * kWidth) + x] = static_cast<uint8_t>((x + y) % 256);
    }
  }

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(in.data());
  task_data_seq->inputs_count.emplace_back(kImageSize);
  task_data_seq->outputs.emplace_back(out.data());
  task_data_seq->outputs_count.emplace_back(kImageSize);

  auto sobel_task = std::make_shared<mezhuev_m_sobel_edge_detection_seq::SobelEdgeDetectionSeq>(task_data_seq);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 400;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(sobel_task);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  bool has_edges = false;
  for (size_t i = 0; i < kImageSize; ++i) {
    if (out[i] > 0) {
      has_edges = true;
      break;
    }
  }
  ASSERT_TRUE(has_edges);
}

TEST(mezhuev_m_sobel_edge_detection_seq, test_task_run) {
  constexpr int kWidth = 512;
  constexpr int kHeight = 512;
  constexpr int kImageSize = kWidth * kHeight;

  std::vector<uint8_t> in(kImageSize, 0);
  std::vector<uint8_t> out(kImageSize, 0);

  for (size_t y = 0; y < kHeight; ++y) {
    for (size_t x = 0; x < kWidth; ++x) {
      in[(y * kWidth) + x] = static_cast<uint8_t>((x + y) % 256);
    }
  }

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(in.data());
  task_data_seq->inputs_count.emplace_back(kImageSize);
  task_data_seq->outputs.emplace_back(out.data());
  task_data_seq->outputs_count.emplace_back(kImageSize);

  auto sobel_task = std::make_shared<mezhuev_m_sobel_edge_detection_seq::SobelEdgeDetectionSeq>(task_data_seq);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 400;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(sobel_task);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  bool has_edges = false;
  for (size_t i = 0; i < kImageSize; ++i) {
    if (out[i] > 0) {
      has_edges = true;
      break;
    }
  }
  ASSERT_TRUE(has_edges);
}