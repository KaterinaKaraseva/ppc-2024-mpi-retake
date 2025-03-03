
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/reduce.hpp>
#include <boost/mpi/communicator.hpp>
#include <functional>
#include <string>

#include "mpi/budazhapova_e_count_freq_character/include/count_freq_chart_mpi_header.hpp"

int budazhapova_e_count_freq_chart_mpi::CountingFreq(std::string str, char symb) {
  int resalt = 0;
  for (unsigned long i = 0; i < str.size(); i++) {
    if (str[i] == symb) {
      resalt++;
    }
  }
  return resalt;
}

bool budazhapova_e_count_freq_chart_mpi::TestMPITaskSequential::PreProcessingImpl() {
  input_ = std::string(reinterpret_cast<char*>(task_data->inputs[0]), static_cast<int>(task_data->inputs_count[0]));
  symb_ = *reinterpret_cast<char*>(task_data->inputs[1]);
  res_ = 0;
  return true;
}

bool budazhapova_e_count_freq_chart_mpi::TestMPITaskSequential::ValidationImpl() {
  return task_data->outputs_count[0] == 1;
}

bool budazhapova_e_count_freq_chart_mpi::TestMPITaskSequential::RunImpl() {
  res_ = CountingFreq(input_, symb_);
  return true;
}

bool budazhapova_e_count_freq_chart_mpi::TestMPITaskSequential::PostProcessingImpl() {
  reinterpret_cast<int*>(task_data->outputs[0])[0] = res_;
  return true;
}

bool budazhapova_e_count_freq_chart_mpi::TestMPITaskParallel::PreProcessingImpl() {
  if (world_.rank() == 0) {
    input_ = std::string(reinterpret_cast<char*>(task_data->inputs[0]), static_cast<int>(task_data->inputs_count[0]));
    symb_ = *reinterpret_cast<char*>(task_data->inputs[1]);
  }
  return true;
}

bool budazhapova_e_count_freq_chart_mpi::TestMPITaskParallel::ValidationImpl() {
  if (world_.rank() == 0) {
    return task_data->outputs_count[0] == 1;
  }
  return true;
}

bool budazhapova_e_count_freq_chart_mpi::TestMPITaskParallel::RunImpl() {
  int delta = 0;
  if (world_.rank() == 0) {
    int input_size = 0;
    input_size = static_cast<int>(task_data->inputs_count[0]);
    delta = (input_size % world_.size() == 0) ? (input_size / world_.size()) : ((input_size / world_.size()) + 1);
  }

  broadcast(world_, delta, 0);
  broadcast(world_, symb_, 0);

  if (world_.rank() == 0) {
    for (int proc = 1; proc < world_.size(); proc++) {
      world_.send(proc, 0, input_.data() + (proc * delta), delta);
    }
  }
  local_input_.resize(delta);
  if (world_.rank() == 0) {
    local_input_ = std::string(input_.begin(), input_.begin() + delta);
  } else {
    world_.recv(0, 0, local_input_.data(), delta);
  }
  local_res_ = CountingFreq(local_input_, symb_);
  boost::mpi::reduce(world_, local_res_, res_, std::plus<>(), 0);
  return true;
}

bool budazhapova_e_count_freq_chart_mpi::TestMPITaskParallel::PostProcessingImpl() {
  if (world_.rank() == 0) {
    reinterpret_cast<int*>(task_data->outputs[0])[0] = res_;
  }
  return true;
}