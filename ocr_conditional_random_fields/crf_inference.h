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

  auto& forward_pass = out_graph.log_alpha;

  // calculate first node potentials
  if (out_graph.potential_status[0] != crf::CALCULATED)
  { 
    out_graph.WX(all, 0) = node_weights * feature_seq[0]; 
    out_graph.potential_status[0] = crf::CALCULATED;
  }

  out_graph.unnormalized_PY1X(all, 0) = out_graph.WX(all, 0);
  forward_pass(all, 0) = out_graph.WX(all, 0);

  // calculate rest of the node potentials
  for (size_t t = 1u; t < seq_len; t++)
  {
    crf::MatrixX<float> log_forward_prev = (forward_pass(all, t-1u).array().log()).matrix();

    if (out_graph.potential_status[t] != crf::CALCULATED)
    { 
      out_graph.WX(all, t) = node_weights * feature_seq[t]; 
      out_graph.potential_status[t] = crf::CALCULATED;
    }
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
                        const crf::NodeWeights_t&   node_weights,
                        const crf::TransWeights_t&  trans_weights,
                        crf::Graph&                 out_graph)
{
  const size_t n_states  = trans_weights.rows();
  const size_t seq_len   = feature_seq.size();

  auto& backward_pass = out_graph.log_beta;

  // do last node calculation
  if (out_graph.potential_status.back() != crf::CALCULATED)
  {
    out_graph.WX(all, last) = node_weights * feature_seq.back();
    out_graph.potential_status.back() = crf::CALCULATED;
  }
  backward_pass(all, last) = out_graph.WX(all, last);

  for (int t = (int)(seq_len-2u); t >= 0; t--)
  {
    crf::MatrixX<float> log_backward_next = (backward_pass(all, t+1u).array().log()).matrix();
    if (out_graph.potential_status[t] != crf::CALCULATED)
    { 
      out_graph.WX(all, t) = node_weights * feature_seq[t];
      out_graph.potential_status[t] = crf::CALCULATED;
    }
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


#ifdef _DEBUG
void backward_algorithm_v2(const crf::Word_t&          feature_seq, 
                           const crf::NodeWeights_t&   node_weights,
                           const crf::TransWeights_t&  trans_weights,
                                 crf::Graph&           out_graph)
{
  const size_t n_states  = trans_weights.rows();
  const size_t seq_len   = feature_seq.size();

  crf::MatrixX<float>& backward_pass = out_graph.log_beta;

  // do last node calculation
  if (out_graph.potential_status.back() != crf::CALCULATED)
  {
    out_graph.WX(all, last) = node_weights * feature_seq.back();
    out_graph.potential_status.back() = crf::CALCULATED;
  }
  float max_value = -1000.0F;
  for (size_t state_i = 0u; state_i < n_states; state_i++)
  {
    for (size_t state_j = 0u; state_j < n_states; state_j++)
    {
      float temp_prob = out_graph.WX(state_i, last) + trans_weights(state_j, state_i);
      if (temp_prob > max_value)
      { max_value = temp_prob; }
      backward_pass(state_i, seq_len-1u) = std::expf(temp_prob);
    }
    // log-sum-exp to avoid numerical underflow and overflow issues
    backward_pass(state_i, seq_len-1u) *= std::expf(-max_value);
    backward_pass(state_i, seq_len-1u)  = max_value + std::logf(backward_pass(state_i, seq_len-1u));
  }

  for (int t = (int)(seq_len-2u); t >= 0; t--)
  {
    crf::MatrixX<float> log_backward_next = (backward_pass(all, t+1u).array().log()).matrix();
    if (out_graph.potential_status[t] != crf::CALCULATED)
    { 
      out_graph.WX(all, t) = node_weights * feature_seq[t];
      out_graph.potential_status[t] = crf::CALCULATED;
    }
    out_graph.unnormalized_PY1X(all, t).noalias() += log_backward_next;
    out_graph.unnormalized_PY1X(all, t) = eig::exp(out_graph.unnormalized_PY1X(all, t).array());

    auto WX = out_graph.WX(all, t);

    float max_value = -1000.0F;

    for (size_t state_i = 0u; state_i < n_states; state_i++)
    {
      for (size_t state_j = 0u; state_j < n_states; state_j++)
      {
        float temp_prob = WX(state_i)  + log_backward_next(state_j);
        if (t != 0u)
        { temp_prob += trans_weights(state_j, state_i); }
        
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

#endif

vector<size_t> viterbi_decode(const crf::Word_t&         feature_seq,
                              const crf::NodeWeights_t&  node_weights,
                              const crf::TransWeights_t& trans_weights)
{
  vector<size_t> predicted_labels(feature_seq.size());
  
  (void)node_weights;
  (void)trans_weights;

  // fill this

  return predicted_labels;
}

#endif
