#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/mezhuev_m_sobel_edge_detection_seq/include/seq.hpp"

TEST(mezhuev_m_sobel_edge_detection_seq, test_basic_case) {
  constexpr int kWidth = 5;
  constexpr int kHeight = 5;
  constexpr int kImageSize = kWidth * kHeight;
  std::vector<uint8_t> in(kImageSize, 0);
  std::vector<uint8_t> out(kImageSize, 0);

  for (size_t y = 0; y < kHeight; ++y) {
    for (size_t x = 0; x < kWidth; ++x) {
      in[(y * kWidth) + x] = static_cast<uint8_t>(x * 50);
    }
  }
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs = {in.data()};
  task_data->inputs_count = {kImageSize};
  task_data->outputs = {out.data()};
  task_data->outputs_count = {kImageSize};

  auto sobel_task = std::make_shared<mezhuev_m_sobel_edge_detection_seq::SobelEdgeDetectionSeq>(task_data);
  ASSERT_TRUE(sobel_task->PreProcessingImpl() && sobel_task->RunImpl() && sobel_task->ValidationImpl() &&
              sobel_task->PostProcessingImpl());
  ASSERT_TRUE(std::ranges::any_of(out.begin(), out.end(), [](uint8_t val) { return val > 0; }));
}

TEST(mezhuev_m_sobel_edge_detection_seq, test_uniform_image) {
  std::vector<uint8_t> in(25, 128);
  std::vector<uint8_t> out(25, 0);
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs = {in.data()};
  task_data->inputs_count = {static_cast<uint32_t>(in.size())};
  task_data->outputs = {out.data()};
  task_data->outputs_count = {static_cast<uint32_t>(out.size())};

  auto sobel_task = std::make_shared<mezhuev_m_sobel_edge_detection_seq::SobelEdgeDetectionSeq>(task_data);
  ASSERT_TRUE(sobel_task->PreProcessingImpl() && sobel_task->RunImpl() && sobel_task->ValidationImpl() &&
              sobel_task->PostProcessingImpl());
  ASSERT_TRUE(std::ranges::all_of(out, [](uint8_t val) { return val == 0; }));
}

TEST(mezhuev_m_sobel_edge_detection_seq, test_empty_input) {
  auto sobel_task = std::make_shared<mezhuev_m_sobel_edge_detection_seq::SobelEdgeDetectionSeq>(
      std::make_shared<ppc::core::TaskData>());
  ASSERT_FALSE(sobel_task->PreProcessingImpl() || sobel_task->RunImpl() || sobel_task->ValidationImpl() ||
               sobel_task->PostProcessingImpl());
}

TEST(mezhuev_m_sobel_edge_detection_seq, test_sharp_contrast) {
  std::vector<uint8_t> in(25, 0);
  std::vector<uint8_t> out(25, 0);
  for (size_t y = 0; y < 5; ++y) {
    for (size_t x = 0; x < 5; ++x) {
      in[(y * 5) + x] = (x < 2) ? 0 : 255;
    }
  }
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs = {in.data()};
  task_data->inputs_count = {static_cast<uint32_t>(in.size())};
  task_data->outputs = {out.data()};
  task_data->outputs_count = {static_cast<uint32_t>(out.size())};

  auto sobel_task = std::make_shared<mezhuev_m_sobel_edge_detection_seq::SobelEdgeDetectionSeq>(task_data);
  ASSERT_TRUE(sobel_task->PreProcessingImpl() && sobel_task->RunImpl() && sobel_task->ValidationImpl() &&
              sobel_task->PostProcessingImpl());
  ASSERT_TRUE(std::ranges::any_of(out.begin(), out.end(), [](uint8_t val) { return val > 0; }));
}

TEST(mezhuev_m_sobel_edge_detection_seq, test_small_image) {
  std::vector<uint8_t> in(9, 255);
  std::vector<uint8_t> out(9, 0);
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs = {in.data()};
  task_data->inputs_count = {static_cast<uint32_t>(in.size())};
  task_data->outputs = {out.data()};
  task_data->outputs_count = {static_cast<uint32_t>(out.size())};

  auto sobel_task = std::make_shared<mezhuev_m_sobel_edge_detection_seq::SobelEdgeDetectionSeq>(task_data);
  ASSERT_TRUE(sobel_task->PreProcessingImpl() && sobel_task->RunImpl() && sobel_task->ValidationImpl() &&
              sobel_task->PostProcessingImpl());
  ASSERT_TRUE(std::ranges::all_of(out, [](uint8_t val) { return val == 0; }));
}

TEST(mezhuev_m_sobel_edge_detection_seq, test_noisy_image) {
  std::vector<uint8_t> in(100, 0);
  std::vector<uint8_t> out(100, 0);
  for (uint8_t &val : in) {
    val = static_cast<uint8_t>(rand() % 256);
  }

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs = {in.data()};
  task_data->inputs_count = {static_cast<uint32_t>(in.size())};
  task_data->outputs = {out.data()};
  task_data->outputs_count = {static_cast<uint32_t>(out.size())};

  auto sobel_task = std::make_shared<mezhuev_m_sobel_edge_detection_seq::SobelEdgeDetectionSeq>(task_data);
  ASSERT_TRUE(sobel_task->PreProcessingImpl() && sobel_task->RunImpl() && sobel_task->ValidationImpl() &&
              sobel_task->PostProcessingImpl());
  ASSERT_TRUE(std::ranges::any_of(out, [](uint8_t val) { return val > 0; }));
}

TEST(mezhuev_m_sobel_edge_detection_seq, test_too_small_images) {
  std::vector<uint8_t> in1(1, 128);
  std::vector<uint8_t> out1(1, 0);
  std::vector<uint8_t> in2(4, 255);
  std::vector<uint8_t> out2(4, 0);
  auto task_data1 = std::make_shared<ppc::core::TaskData>();
  auto task_data2 = std::make_shared<ppc::core::TaskData>();

  task_data1->inputs = {in1.data()};
  task_data1->inputs_count = {static_cast<uint32_t>(in1.size())};
  task_data1->outputs = {out1.data()};
  task_data1->outputs_count = {static_cast<uint32_t>(out1.size())};

  task_data2->inputs = {in2.data()};
  task_data2->inputs_count = {static_cast<uint32_t>(in2.size())};
  task_data2->outputs = {out2.data()};
  task_data2->outputs_count = {static_cast<uint32_t>(out2.size())};

  auto sobel_task1 = std::make_shared<mezhuev_m_sobel_edge_detection_seq::SobelEdgeDetectionSeq>(task_data1);
  auto sobel_task2 = std::make_shared<mezhuev_m_sobel_edge_detection_seq::SobelEdgeDetectionSeq>(task_data2);

  ASSERT_FALSE(sobel_task1->PreProcessingImpl());
  ASSERT_FALSE(sobel_task2->PreProcessingImpl());
}

TEST(mezhuev_m_sobel_edge_detection_seq, test_horizontal_gradient) {
  std::vector<uint8_t> in(25, 0);
  std::vector<uint8_t> out(25, 0);
  for (size_t y = 0; y < 5; ++y) {
    for (size_t x = 0; x < 5; ++x) {
      in[(y * 5) + x] = static_cast<uint8_t>(x * 50);
    }
  }
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs = {in.data()};
  task_data->inputs_count = {static_cast<uint32_t>(in.size())};
  task_data->outputs = {out.data()};
  task_data->outputs_count = {static_cast<uint32_t>(out.size())};

  auto sobel_task = std::make_shared<mezhuev_m_sobel_edge_detection_seq::SobelEdgeDetectionSeq>(task_data);
  ASSERT_TRUE(sobel_task->PreProcessingImpl() && sobel_task->RunImpl() && sobel_task->ValidationImpl() &&
              sobel_task->PostProcessingImpl());
  ASSERT_TRUE(std::ranges::any_of(out, [](uint8_t val) { return val > 0; }));
}
