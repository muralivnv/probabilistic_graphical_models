#ifndef _CRF_INFERENCE_H_
#define _CRF_INFERENCE_H_

#include <cmath>
#include <queue>

#include "crf_typedef.h"

void forward_algorithm(const crf::Word_t&         feature_seq, 
                       const crf::NodeWeights_t&  node_weights,
                       const crf::TransWeights_t& trans_weights,
                       crf::Graph&                out_graph)
{
  const size_t n_states  = trans_weights.rows();
  const size_t seq_len   = feature_seq.size();

  auto& forward_pass = out_graph.alpha;

  // calculate first node potentials
  out_graph.WX(all, 0) = node_weights * feature_seq[0]; 
  out_graph.unnormalized_PY1X(all, 0) = out_graph.WX(all, 0);
  forward_pass(all, 0) = out_graph.WX(all, 0);

  // calculate rest of the node potentials
  for (size_t t = 1u; t < seq_len; t++)
  {
    crf::MatrixX<float> log_forward_prev = (forward_pass(all, t-1u).array().log()).matrix();

    out_graph.WX(all, t) = node_weights * feature_seq[t]; 
    out_graph.unnormalized_PY1X(all, t) = out_graph.WX(all, t);
    out_graph.unnormalized_PY1X(all, t).noalias() += log_forward_prev;

    auto WX = out_graph.WX(all, t);
    float max_value = -10000.0F;

    for (size_t state_i = 0u; state_i < n_states; state_i++)
    {
      for (size_t state_j = 0u; state_j < n_states; state_j++)
      {
        float temp_prob = WX(state_i) + trans_weights(state_j, state_i) + log_forward_prev(state_j);
        if (temp_prob > max_value)
        { max_value = temp_prob; }

        forward_pass(state_i, t) += std::expf(temp_prob);
      }
      // log-sum-exp to avoid numerical underflow and overflow issues
      forward_pass(state_i, t) *= std::expf(-max_value);
      forward_pass(state_i, t) = max_value + std::logf(forward_pass(state_i, t));
    }
  }
}


void backward_algorithm(const crf::Word_t&          feature_seq, 
        [[maybe_unused]]const crf::NodeWeights_t&   node_weights,
                        const crf::TransWeights_t&  trans_weights,
                        crf::Graph&                 out_graph)
{
  const size_t n_states  = trans_weights.rows();
  const size_t seq_len   = feature_seq.size();

  auto& backward_pass = out_graph.beta;

  // do last node calculation
  backward_pass(all, last) = out_graph.WX(all, last);

  for (int t = (int)(seq_len-2u); t >= 0; t--)
  {
    crf::MatrixX<float> log_backward_next = (backward_pass(all, t+1u).array().log()).matrix();
    out_graph.unnormalized_PY1X(all, t).noalias() += log_backward_next;
    out_graph.unnormalized_PY1X(all, t) = eig::exp(out_graph.unnormalized_PY1X(all, t).array());

    auto WX = out_graph.WX(all, t);
    float max_value = -1000.0F;

    for (size_t state_i = 0u; state_i < n_states; state_i++)
    {
      for (size_t state_j = 0u; state_j < n_states; state_j++)
      {
        float temp_prob = WX(state_i) + trans_weights(state_i, state_j) + log_backward_next(state_j);
        
        if (temp_prob > max_value)
        { max_value = temp_prob; }

        backward_pass(state_i, t) += std::expf(temp_prob);
      }
      // log-sum-exp to avoid numerical underflow and overflow issues
      backward_pass(state_i, t) *= std::expf(-max_value);
      backward_pass(state_i, t)  = max_value + std::logf(backward_pass(state_i, t));
    }
  }
}


void forward_algorithm_scaled(const crf::Word_t&         feature_seq, 
                              const crf::NodeWeights_t&  node_weights,
                              const crf::TransWeights_t& trans_weights,
                              crf::Graph&                out_graph)
{
  const size_t n_states  = trans_weights.rows();
  const size_t seq_len   = feature_seq.size();

  auto& forward_pass = out_graph.alpha;

  // calculate first node potentials
  out_graph.WX(all, 0) = node_weights * feature_seq[0];
  out_graph.unnormalized_PY1X(all, 0) = out_graph.WX(all, 0);
  forward_pass(all, 0) = eig::exp(out_graph.WX(all, 0).array());
  
  out_graph.scaling_factors(0) = forward_pass(all, 0).sum();
  forward_pass(all, 0) /= out_graph.scaling_factors(0);

  // calculate rest of the node potentials
  for (size_t t = 1u; t < seq_len; t++)
  {
    auto alpha_prev = forward_pass(all, t-1u);

    out_graph.WX(all, t) = node_weights * feature_seq[t]; 
    out_graph.unnormalized_PY1X(all, t) = eig::exp(out_graph.WX(all, t).array());
    out_graph.unnormalized_PY1X(all, t).array() *= alpha_prev.array();

    auto WX = out_graph.WX(all, t);

    for (size_t state_i = 0u; state_i < n_states; state_i++)
    {
      for (size_t state_j = 0u; state_j < n_states; state_j++)
      {
        float temp_prob = WX(state_i) + trans_weights(state_j, state_i);
        forward_pass(state_i, t) += std::expf(temp_prob)*alpha_prev(state_j);
      }
      out_graph.scaling_factors(t) += forward_pass(state_i, t);
    }
    // scale these factors
    forward_pass(all, t) /= out_graph.scaling_factors(t);
  }
}


void backward_algorithm_scaled(const crf::Word_t&          feature_seq, 
               [[maybe_unused]]const crf::NodeWeights_t&   node_weights,
                               const crf::TransWeights_t&  trans_weights,
                               crf::Graph&                 out_graph)
{
  const size_t n_states  = trans_weights.rows();
  const size_t seq_len   = feature_seq.size();

  auto& backward_pass = out_graph.beta;

  // do last node calculation
  backward_pass(all, last) = (out_graph.WX(all, last).array().exp())/out_graph.scaling_factors(last);

  for (int t = (int)(seq_len-2u); t >= 0; t--)
  {
    auto beta_next = backward_pass(all, t+1u);
    out_graph.unnormalized_PY1X(all, t).array() *= beta_next.array();

    auto WX = out_graph.WX(all, t);

    for (size_t state_i = 0u; state_i < n_states; state_i++)
    {
      for (size_t state_j = 0u; state_j < n_states; state_j++)
      {
        float temp_prob = WX(state_i) + trans_weights(state_i, state_j);
        backward_pass(state_i, t) += std::expf(temp_prob)*beta_next(state_j);
      }
    }
    //scale values
    backward_pass(all, t) /= out_graph.scaling_factors(t);
  }
}


void exact_inference(const crf::Word_t&         feature_seq,
                     const crf::NodeWeights_t&  node_weights,
                     const crf::TransWeights_t& trans_weights,
                           crf::Graph&          out_graph)
{
  // perform forward propagation
  forward_algorithm_scaled(feature_seq, node_weights, trans_weights, out_graph);

  // perform backward propagation
  backward_algorithm_scaled(feature_seq, node_weights, trans_weights, out_graph);
}


vector<size_t> viterbi_decode(const crf::Word_t&         feature_seq,
                              const crf::NodeWeights_t&  node_weights,
                              const crf::TransWeights_t& trans_weights)
{
  const size_t n_states = node_weights.rows();
  const size_t seq_len  = feature_seq.size();
  vector<size_t> predicted_labels(seq_len);

  crf::MatrixX<float> viterbi_path_prob(n_states, seq_len);
  crf::MatrixX<size_t> viterbi_path_parent(n_states, seq_len);

  // initialize first node
  viterbi_path_prob(all, 0) = node_weights * feature_seq[0];
  
  for (size_t t = 1u; t < seq_len; t++)
  {
    viterbi_path_prob(all, t) = node_weights * feature_seq[t];
    for (size_t state_i = 0u; state_i < n_states; state_i++)
    {
      float max_prob = viterbi_path_prob(0u, t-1u) + trans_weights(0, state_i);
      size_t best_parent = 0u;
      for (size_t state_j = 1u; state_j < n_states; state_j++)
      {
        float cur_prob = viterbi_path_prob(state_j, t-1u) + trans_weights(state_j, state_i);
        if (max_prob < cur_prob)
        {
          max_prob    = cur_prob;
          best_parent = state_j;
        }
      }
      viterbi_path_prob(state_i, t) += max_prob;
      viterbi_path_parent(state_i, t) = best_parent;
    }
  }

  // take max probability state from the last and back track
  eig::Index best_idx;
  [[maybe_unused]]float max_prob = viterbi_path_prob(all, last).maxCoeff(&best_idx);
  predicted_labels.back() = static_cast<size_t>(best_idx);
  for (int t = (int)(seq_len-2u); t >= 0; t--)
  {
    predicted_labels[t] = viterbi_path_parent(best_idx, t);
    best_idx = static_cast<eig::Index>(predicted_labels[t]);
  }

  return predicted_labels;
}

#endif
