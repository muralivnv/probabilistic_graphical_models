#ifndef _CRF_LEARNING_H_
#define _CRF_LEARNING_H_

#include <tuple>
#include "crf_typedef.h"

#ifdef ENABLE_PLOTTING
#include <cppyplot.hpp>
#endif

template<typename CostFunc_t, typename CostFuncPrime_t>
tuple<vector<float>, crf::NodeWeights_t, crf::TransWeights_t>
steepest_descent(const crf::Words_t&            train_features, 
                 const vector<vector<size_t>>&  train_labels,
                 const crf::NodeWeights_t&      node_weights, 
                 const crf::TransWeights_t&     trans_weights, 
                 const CostFunc_t&              cost_func, 
                 const CostFuncPrime_t&         cost_func_gradient, 
                 const crf::MinimizerParams&    params)
{
  static const size_t n_epochs    = std::any_cast<size_t>(params.at("n_epochs"));
  static const float step_size    = std::any_cast<float>(params.at("step_size"));
  static const float l2reg_factor = std::any_cast<float>(params.at("l2reg_factor"));

  const std::vector<size_t> train_indices = std::any_cast<vector<size_t>>(params.at("train_indices"));

  size_t n              = train_indices.size();
  size_t n_states       = node_weights.rows();
  float prev_epoch_cost = 1000.0F;

  crf::MatrixX<float> weight_gradient(node_weights.size() + trans_weights.size(), 1u);
  std::vector<float> cost_trend(n_epochs, 0.0F);
  crf::NodeWeights_t  estimated_node_weights = node_weights;
  crf::TransWeights_t estimated_bias_weights = trans_weights;

#ifdef ENABLE_PLOTTING
  Cppyplot::cppyplot pyp;

  pyp.raw(R"pyp(
    plt.ion()
    fig, ax = plt.subplots(2, 5, figsize=(8,5))
    ax = np.ravel(ax)
    im_obj = [None]*ax.shape[0]
    titles = ''.join(['%c'%(k) for k in range(97, 97+ax.shape[0])])
    for i in range(0, ax.shape[0]):
      im_obj[i] = ax[i].imshow(estimated_node_weights[i, :].reshape(25, 15), cmap=cm.Greys, interpolation=None)
      ax[i].set_title(titles[i], fontsize=14)
    plt.show()
    plt.pause(0.1)
    )pyp", _p(estimated_node_weights));
#endif

  for (size_t epoch = 0u; epoch < n_epochs; epoch++)
  {
    weight_gradient.fill(0.0F);
    
    #pragma omp parallel for num_threads(2)
    for (int i = 0u; i < (int)n; i++)
    {
      size_t train_data_idx = train_indices[i];

      size_t seq_len = train_features[train_data_idx].size();
      crf::Graph graph(n_states, seq_len);

      // perform exact_inference and construct graph
      exact_inference(train_features[train_data_idx], estimated_node_weights, estimated_bias_weights, graph);

      // calculate gradient
      #pragma omp critical
      { cost_func_gradient(train_features[train_data_idx], train_labels[train_data_idx], estimated_node_weights, estimated_bias_weights, graph, params, weight_gradient); }

      // calculate cost
      cost_trend[epoch] += cost_func(train_features[train_data_idx], train_labels[train_data_idx], estimated_node_weights, estimated_bias_weights, graph, params);
    }
    weight_gradient   /= static_cast<float>(n);
    cost_trend[epoch] /= static_cast<float>(n);

    if (!std::isfinite(cost_trend[epoch]))
    {
      std::cout << " objective function is overflown because of numerical errors, terminating\n";
      break;
    }

    if ((prev_epoch_cost - cost_trend[epoch]) < 1e-5f)
    {
      std::cout << " change in objective function is within tolerance, terminating\n";
      break;
    }
    prev_epoch_cost = cost_trend[epoch];
    std::cout << "epoch: " << epoch << " , cost: " << cost_trend[epoch] << '\n';

    // update weights
    estimated_node_weights.noalias() -= step_size*weight_gradient(seq(0, estimated_node_weights.size()-1u), 0u).reshaped<eig::RowMajor>(26, 375);
    estimated_bias_weights.noalias() -= step_size*weight_gradient(seq(estimated_node_weights.size(), last), 0u).reshaped<eig::RowMajor>(26, 26);

#ifdef ENABLE_PLOTTING
    if (std::isfinite(cost_trend[epoch]))
    {
      pyp.raw(R"pyp(
      all_axes = plt.gcf().get_axes()
      for i in range(0, len(all_axes)):
        all_axes[i].imshow(estimated_node_weights[i, :].reshape(25, 15), cmap=cm.Greys, interpolation=None)
      
      plt.gcf().suptitle("epoch: {0}, cost: {1:0.2f}".format(epoch, prev_epoch_cost),fontsize=14)
      plt.show()
      plt.pause(0.1)
      )pyp", _p(estimated_node_weights), _p(prev_epoch_cost), _p(epoch));
    }
#endif
  }

#ifdef ENABLE_PLOTTING
  pyp.raw(R"pyp(
    plt.show(block=True)
  )pyp", _p(n_epochs));
#endif

  return std::make_tuple(cost_trend, estimated_node_weights, estimated_bias_weights);
}

#endif
