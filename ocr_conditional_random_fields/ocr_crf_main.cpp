#include <cstdio>
#include <iostream>

#include <unordered_map>
#include <random>
#include <chrono>
#include <algorithm>

#include <cpputil/cpputil.hpp>

#include "dataset_util.h"
#include "crf_typedef.h"
#include "crf_inference.h"
#include "crf_cost_func.h"
#include "crf_learning.h"
#include "crf_evaluate.h"

#ifdef ENABLE_PLOTTING
#include <cppyplot.hpp>
#endif

int main()
{
#ifdef ENABLE_PLOTTING
  Cppyplot::cppyplot pyp;
#endif
  const auto root = (std::filesystem::path(__FILE__)).parent_path().string();
  std::random_device seed;
  std::mt19937 gen(seed());

  auto dat_files = files_with_extension(root + "/dataset/data/processed/breta/words_gaplines", ".dat");
  size_t n = dat_files.size();

  crf::Words_t images(n);
  vector<vector<size_t>> labels_int(n);

  // Read Dataset 
  std::cout << "reading dataset ...\n";
  read_data(dat_files, images, labels_int);

  const size_t n_states = 26u;
  crf::NodeWeights_t node_weights;
  crf::TransWeights_t transition_weights; 

  // initialize node weights and transition weights
  std::uniform_real_distribution<float> uniform(0.0F, 0.01F);

  for (auto row: node_weights.rowwise())
  { std::for_each(row.begin(), row.end(), [&](float& elem){ elem = uniform(gen);}); }

  for (auto row: transition_weights.rowwise())
  { std::for_each(row.begin(), row.end(), [&](float& elem){ elem = uniform(gen);}); }
  
  // Split Dataset
  auto[train_indices, test_indices] = util::train_test_split(images.size(), 0.7F);

  auto start_time = std::chrono::system_clock::now();

  // For each dataset call forward and backward algorithm for inference
  crf::MinimizerParams solver_params;
  solver_params["n_epochs"]       = size_t{300u};
  solver_params["step_size"]      = float{0.001};
  solver_params["l2reg_factor"]   = float{0.01};
  solver_params["train_data_len"] = size_t{images.size()};
  solver_params["train_indices"]  = train_indices;
  auto [cost_trend, weights, bias] = steepest_descent(images, labels_int, 
                                                      node_weights, transition_weights, 
                                                      log_conditional_prob, log_conditional_prime, 
                                                      solver_params);
  auto end_time = std::chrono::system_clock::now();

  std::cout << "Elapsed: " << std::chrono::duration<double>(end_time - start_time).count() << '\n';

  float accuracy = calc_accuracy(images, labels_int, test_indices, weights, bias);
  std::cout << "\naccuracy across test data: " << accuracy;

#ifdef ENABLE_PLOTTING
  pyp.raw(R"pyp(
    plt.figure(figsize=(6,5))
    plt.plot(cost_trend, 'g--x', markersize=2, linewidth=1.2)
    plt.grid(True)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.show(block=True)
  )pyp", _p(cost_trend));
#endif

  // write out estimated parameters
  write_to_file("estimated_state_weights.txt", weights, {(size_t)weights.rows(), (size_t)weights.cols()});
  write_to_file("estimated_transition_weights.txt", bias, {(size_t)bias.rows(), (size_t)bias.cols()});

  return EXIT_SUCCESS;
}
