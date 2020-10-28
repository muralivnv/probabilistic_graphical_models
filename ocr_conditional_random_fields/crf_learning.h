#ifndef _CRF_LEARNING_H_
#define _CRF_LEARNING_H_

#include <tuple>
#include "crf_typedef.h"

template<typename CostFunc_t, typename CostFuncPrime_t>
tuple<vector<float>, crf::NodeWeights_t, crf::TransWeights_t>
steepest_descent(const crf::Words_t&            train_features, 
                 const vector<vector<size_t>>&  train_labels,
                 const crf::NodeWeights_t&      node_weights, 
                 const crf::TransWeights_t&     trans_weights, 
                 const CostFunc_t&              cost_func, 
                 const CostFuncPrime_t&         cost_func_gradient, 
                       size_t                   n_epochs=10,
                       float                    step_size=0.05F, 
                       float                    regularization_factor = 0.001F)
{
  (void)regularization_factor;
  
  crf::MatrixX<float> weight_gradient(node_weights.size() + trans_weights.size(), 1u);
  weight_gradient.fill(0.0F);

  std::vector<float> cost_trend(n_epochs, 0.0F);
  crf::NodeWeights_t  estimated_node_weights = node_weights;
  crf::TransWeights_t estimated_bias_weights = trans_weights;
  size_t n = train_features.size();
  size_t n_states = node_weights.rows();

  for (size_t epoch = 0u; epoch < n_epochs; epoch++)
  {
    weight_gradient.fill(0.0F);
    for (size_t i = 0u; i < n; i++)
    {
      size_t seq_len = train_features[i].size();
      crf::Graph graph(n_states, seq_len);

      // call forward algorithm
      forward_algorithm(train_features[i], estimated_node_weights, estimated_bias_weights, graph);
      
      // call backward algorithm
      backward_algorithm(train_features[i], estimated_node_weights, estimated_bias_weights, graph);

      // calculate gradient
      cost_func_gradient(train_features[i], train_labels[i], estimated_node_weights, estimated_bias_weights, graph, weight_gradient);

      // calculate cost
      cost_trend[epoch] += cost_func(train_features[i], train_labels[i], estimated_node_weights, estimated_bias_weights, graph);
    }
    weight_gradient   /= static_cast<float>(n);
    cost_trend[epoch] /= static_cast<float>(n);

    estimated_node_weights.noalias() += step_size*weight_gradient(seq(0, estimated_node_weights.size()-1u), 0u).reshaped<eig::RowMajor>(26, 375);
    estimated_bias_weights.noalias() += step_size*weight_gradient(seq(estimated_node_weights.size(), last), 0u).reshaped<eig::RowMajor>(26, 26);
    
    std::cout << "epoch: " << epoch << " , cost: " << cost_trend[epoch] << '\n';
  }

  return std::make_tuple(cost_trend, estimated_node_weights, estimated_bias_weights);
}

#endif
