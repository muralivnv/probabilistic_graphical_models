#ifndef _CRF_COST_FUNC_H_
#define _CRF_COST_FUNC_H_

#include "crf_typedef.h"
#include "crf_inference.h"

float log_conditional_prob(const crf::Word_t&          feature_seq,
                           const vector<size_t>&       label_seq,
                           const crf::NodeWeights_t&   node_weights, 
                           const crf::TransWeights_t&  transition_weights,
                           const crf::Graph&           graph,
                           const crf::MinimizerParams& params)
{
  float cost = 0.0F;
  static const float l2reg_factor = std::any_cast<float>(params.at("l2reg_factor"));
  static const size_t train_data_len = std::any_cast<size_t>(params.at("train_data_len"));
  static const float reg_scaling = l2reg_factor/(2.0F*(float)train_data_len);

  // for the first node at time t = 0u
  cost -= graph.WX(label_seq[0u], 0);
  cost += reg_scaling*(node_weights(label_seq[0], all)).squaredNorm();

  for (size_t i = 1u; i < feature_seq.size(); i++)
  {
    cost -= graph.WX(label_seq[i], i);
    cost -= transition_weights(label_seq[i-1], label_seq[i]);

    cost += reg_scaling*(node_weights(label_seq[i], all)).squaredNorm();
  }

  // sum up partition function
  cost += std::logf(graph.scaling_factors(last));

  return cost;
}


void log_conditional_prime(const crf::Word_t&          feature_seq,
                           const vector<size_t>&       label_seq,
                           const crf::NodeWeights_t&   node_weights, 
                           const crf::TransWeights_t&  transition_weights,
                           const crf::Graph&           graph,
                           const crf::MinimizerParams& params,
                                 crf::MatrixX<float>&  gradient_out)
{
  const size_t seq_len            = feature_seq.size();
  const size_t n_states           = node_weights.rows();
  const size_t feature_len        = feature_seq[0].size();
  const size_t trans_weight_start = node_weights.size();
  static const float l2reg_factor    = std::any_cast<float>(params.at("l2reg_factor"));
  static const size_t train_data_len = std::any_cast<size_t>(params.at("train_data_len"));

  // node gradient
  for (size_t t = 0u; t < seq_len; t++)
  {
    size_t label = label_seq[t];

    float normalized_PY1X = graph.unnormalized_PY1X(label, t)/graph.unnormalized_PY1X(all, t).sum();
    auto scaling = normalized_PY1X;

    for (size_t state = 0u; state < n_states; state++)
    {
      size_t index_start =  state*feature_len;
      size_t index_end   =  (state+1)*feature_len-1u;

      float this_state_scaling = scaling;
      if (state == label)
      { this_state_scaling -= 1.0F; }

      // Fill gradient
      gradient_out(seq(index_start, index_end), 0).noalias() += (this_state_scaling*feature_seq[t])
                                                                + ((l2reg_factor/train_data_len)*node_weights(state, all)).reshaped<eig::RowMajor>(375, 1);
    }

  }


  // transition gradient
  for (size_t t = 1u; t < seq_len; t++)
  {
    size_t label_prev = label_seq[t-1u];
    size_t label_now  = label_seq[t];

    float numerator = graph.WX(label_prev, t-1u);
    numerator      += graph.WX(label_now, t);
    numerator      += transition_weights(label_prev, label_now);
    numerator       = std::expf(numerator);

    if (t > 1u) // add forward pass result
    { numerator *= graph.alpha(label_prev, t-2u); }

    if (t < seq_len - 1u) // add backward pass result
    { numerator *= graph.beta(label_now, t+1u); }

    float denominator = 0.0F;
    for (size_t state_i = 0u; state_i < n_states; state_i++)
    {
      float state_i_potential = graph.WX(state_i, t-1u);
      for (size_t state_j = 0u; state_j < n_states; state_j++)
      {
        if ((state_i == label_prev) && (state_j == label_now))
        { denominator += numerator; continue; }

        float coeff = graph.WX(state_j, t);
        coeff += state_i_potential;
        coeff += transition_weights(state_i, state_j);
        coeff  = std::expf(coeff);

        if (t > 1u)
        { coeff *= graph.alpha(state_i, t-2u); }
        if (t < seq_len-1u)
        { coeff *= graph.beta(state_j, t+1u); }

        denominator += coeff;
      }
    }

    float scaling = (numerator / denominator);
    for (int state_i = 0u; state_i < (int)n_states; state_i++)
    {
      for (size_t state_j = 0u; state_j < n_states; state_j++)
      {
        float coeff = scaling;
        if ((state_i == label_prev) && (state_j == label_now))
        { coeff -= 1.0F;}
        gradient_out(trans_weight_start + (state_i*n_states) + state_j) += coeff;
      }
    }
  }
}

#endif
