#ifndef _BAYESIAN_NETWORK_TYPE_H_
#define _BAYESIAN_NETWORK_TYPE_H_

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>

#include "BN_types.h"
#include "util.h"


void add_edge(std::map<std::string, std::shared_ptr<networkNode>>& network, 
              const std::string& parent_name, 
              const std::string& child_name)
{
  if (   (network.find(parent_name) != network.end())
      && (network.find(child_name)  != network.end()) )
  {
    std::shared_ptr<networkNode> child_ptr(network[child_name]);
    network[parent_name]->children.push_back(child_ptr);
  }
}

factor make_factor_with_val(const std::initializer_list<unsigned int> & vars, 
                             const std::initializer_list<unsigned int>& cardinals,
                             const std::initializer_list<float>& values)
{
  factor return_elem;
  return_elem.variables = std::vector<unsigned int>(vars);
  return_elem.cardinals = std::vector<unsigned int>(cardinals);
  return_elem.values    = std::vector<float>(values);

  return return_elem;
}


factor make_factor(const std::initializer_list<unsigned int>& vars, 
                    const std::initializer_list<unsigned int>& cardinals)
{
  factor return_elem;
  return_elem.variables = std::vector<unsigned int>(vars);
  return_elem.cardinals = std::vector<unsigned int>(cardinals);
  unsigned int number_of_values = 1u;

  for (unsigned int& elem: return_elem.cardinals)
  { number_of_values *= elem; }
  return_elem.values = std::vector<float>(number_of_values, 0.0f);

  return return_elem;
}


void get_intersection(const std::vector<unsigned int>& vec1, 
                      const std::vector<unsigned int>& vec2, 
                            std::vector<unsigned int>& vec1_intersection,
                            std::vector<unsigned int>& vec2_intersection)
{
  vec1_intersection.reserve(vec1.size());
  vec2_intersection.reserve(vec2.size());

  for (std::size_t iter1 = 0u; iter1 < vec1.size(); iter1++)
  {
    for (std::size_t iter2 = 0u; iter2 < vec2.size(); iter2++)
    {
      if (vec1[iter1] == vec2[iter2])
      {
        vec1_intersection.push_back(iter1);
        vec2_intersection.push_back(iter2);
      }
    }
  }
  vec1_intersection.shrink_to_fit();
  vec2_intersection.shrink_to_fit();
}


void get_difference(const std::vector<UInt>& vec, 
                    const std::vector<UInt>& vec_to_find, 
                    std::vector<UInt>& difference)
{
  difference.reserve(vec.size());

  for (std::size_t iter1 = 0u; iter1 < vec.size(); iter1++)
  {
    bool found_elem = false;
    for (std::size_t iter2 = 0u; iter2 < vec_to_find.size(); iter2++)
    {
      if (vec[iter1] == vec_to_find[iter2])
      {
        found_elem = true;
        break;
      }
    }
    if (found_elem == false)
    {
      difference.push_back(vec[iter1]);
    }
  }
  
  difference.shrink_to_fit();
}


void get_factor_union(const factor& factor1, 
                      const factor& factor2,
                      const UIntVec& factor1_intersection,
                      const UIntVec& factor2_intersection,
                            factor& factor_union)
{
  std::vector<unsigned int> factor1_vars = factor1.variables;
  std::vector<unsigned int> factor2_vars = factor2.variables;

  factor_union.variables.reserve(factor1_vars.size() + factor2_vars.size() - factor1_intersection.size());
  factor_union.cardinals.reserve(factor1_vars.size() + factor2_vars.size() - factor1_intersection.size());

  for (std::size_t iter1 = 0u; iter1 < factor1_intersection.size(); iter1++)
  {
    unsigned int factor1_var = factor1.variables[factor1_intersection[iter1]];
    unsigned int factor1_car = factor1.cardinals[factor1_intersection[iter1]];
    
    factor_union.variables.push_back(factor1_var);
    factor_union.cardinals.push_back(factor1_car);

    factor1_vars[factor1_intersection[iter1]] = 1000u;
    factor2_vars[factor2_intersection[iter1]] = 1000u;
  }

  for (std::size_t iter = 0u; iter < factor1_vars.size(); iter++)
  {
    if (factor1_vars[iter] != 1000u)
    {
      factor_union.variables.push_back(factor1_vars[iter]);
      factor_union.cardinals.push_back(factor1.cardinals[iter]);
    }
  }

  for (std::size_t iter = 0u; iter < factor2_vars.size(); iter++)
  {
    if (factor2_vars[iter] != 1000u)
    {
      factor_union.variables.push_back(factor2_vars[iter]);
      factor_union.cardinals.push_back(factor2.cardinals[iter]);
    }
  }
  factor_union.variables.shrink_to_fit();
  factor_union.cardinals.shrink_to_fit();
}


void get_state_indices(const factor& factor_to_calc, 
                       const unsigned int state_index, 
                       std::vector<UIntVec>& output)
{
  UInt cardinality = factor_to_calc.cardinals[state_index];
  output.reserve(cardinality);

  UInt prod_to_state_index = vec_prod_n(factor_to_calc.cardinals, state_index);
  prod_to_state_index     -= 1u;

  UInt total_elems         = vec_prod_n(factor_to_calc.cardinals, factor_to_calc.cardinals.size());
  //  UInt elems_left          = total_elems - ((prod_to_state_index+1u)* cardinality);

  UInt start = 0u;
  UInt end   = prod_to_state_index;
  output.push_back(arange(start, end, 1u));

  /* output of each row represent indices for each state */
  for (UInt var_state = 1u; var_state < cardinality; var_state++)
  {
    start = end   + 1u;
    end   = start + prod_to_state_index;
    output.push_back(arange(start, end, 1u));
  }

  // out of the elems_left, these indices will repeat every (prod_to_state_index * cardinality)
  // out of the repeatedness, prod_to_state_index-1 is the start and end index
  for (UInt iter = end + 1u; iter < total_elems; iter += (prod_to_state_index + 1u)*cardinality)
  {
    start = iter;
    end = start + prod_to_state_index;
    auto ranges = arange(start, end, 1u);
    output[0].reserve(end - start);
    std::copy(ranges.begin(), ranges.end(), std::back_inserter(output[0]));

    for (UInt i = 1; i < cardinality; i++)
    {
      start = end + 1u;
      end   = start + prod_to_state_index;
      ranges = arange(start, end, 1u);
      output[i].reserve(end - start);
      std::copy(ranges.begin(), ranges.end(), std::back_inserter(output[i])); 
    }
  }
}


int get_var_index(const factor& factor_to_find, 
                  const UInt var_to_find)
{
  int result = -1;
  for (std::size_t iter = 0u; iter < factor_to_find.variables.size(); iter++)
  {
    if (factor_to_find.variables[iter] == var_to_find)
    {
      result = iter;
      break;
    }
  }
  return result;
}


void factor_product(const factor& factor_left, 
                    const factor& factor_right,
                          factor& product_result)
{
  bool factor_left_empty = factor_left.variables.empty();
  bool factor_right_empty = factor_right.variables.empty();
  if (factor_left_empty && !factor_right_empty)
  {
    copy_vals(factor_right.variables, product_result.variables);
    copy_vals(factor_right.cardinals, product_result.cardinals);
    copy_vals(factor_right.values, product_result.values);
  }
  else if (!factor_left_empty && factor_right_empty)
  {
    copy_vals(factor_left.variables, product_result.variables);
    copy_vals(factor_left.cardinals, product_result.cardinals);
    copy_vals(factor_left.values, product_result.values);
  }
  else if (!factor_left_empty && !factor_right_empty)
  {
    std::vector<unsigned int> intersection_indices_left, intersection_indices_right;
    get_intersection(factor_left.variables, 
                     factor_right.variables, 
                     intersection_indices_left,
                     intersection_indices_right);

    if (intersection_indices_left.empty() == false)
    {
      bool cardinals_match = true;

      // check if cardinalities matches for the intersection
      for (std::size_t iter = 0; iter < intersection_indices_left.size(); iter++)
      {
        unsigned int factor_left_idx  = intersection_indices_left[iter];
        unsigned int factor_right_idx = intersection_indices_right[iter];

        if (factor_left.cardinals[factor_left_idx] != factor_right.cardinals[factor_right_idx])
        {
          cardinals_match = false;
        }
      }
      if (cardinals_match == false)
      {
        std::cout << "Cardinals don't match, couldn't perform factor product\n";
        return;
      }
      
      // vars of final operation is the union of A and B vars
      get_factor_union(factor_left, 
                       factor_right, 
                       intersection_indices_left,
                       intersection_indices_right,
                       product_result);

      product_result.values = std::vector<float> (vec_prod(product_result.cardinals), 0.0F);

      // now need to multiply factor together
      std::vector<UIntVec> indices_left, indices_right, indices_prod;
      
      // take a sample matched variable index and get all the iterable index list
      get_state_indices(factor_left,    intersection_indices_left[0],  indices_left);
      get_state_indices(factor_right,   intersection_indices_right[0], indices_right);
      
      // assumes that first var in left and right matches with 1st var in product
      get_state_indices(product_result, 0, indices_prod); 

      // now do element wise product for each element in indices_left, indices_right;
      UInt iter_prod;
      for (std::size_t state = 0; state < indices_prod.size(); state++)
      {
        iter_prod = 0u;
        for (std::size_t iter_right = 0; iter_right < indices_right[state].size(); iter_right++)
        {
          for (std::size_t iter_left = 0; iter_left < indices_left[state].size(); iter_left++)
          {
            product_result.values[indices_prod[state][iter_prod]] = factor_left.values[indices_left[state][iter_left]];
            product_result.values[indices_prod[state][iter_prod]] *= factor_right.values[indices_right[state][iter_right]];

            ++iter_prod;
          }
        }
      }
    }
  }
}


void factor_marginalize(const factor& factor_marginalize,
                        const UInt marginalize_var,
                              factor& marginal_result)
{
  std::vector<UIntVec> factor_indices;
  int var_index = get_var_index(factor_marginalize, 
                                marginalize_var);
  if (var_index == -1)
  {
    std::cout << "given variable -> " << marginalize_var << " not found\n";
    std::cout << "passed factor is: \n" << factor_marginalize << '\n';
  }
  else
  {
    // copy variables over,
    marginal_result.variables.reserve(factor_marginalize.variables.size()-1);
    marginal_result.cardinals.reserve(factor_marginalize.cardinals.size()-1);

    for (std::size_t source_iter = 0u; source_iter < factor_marginalize.variables.size(); source_iter++)
    {
      if (source_iter != var_index)
      {
        marginal_result.variables.push_back(factor_marginalize.variables[source_iter]);
        marginal_result.cardinals.push_back(factor_marginalize.cardinals[source_iter]);
      }
    }
    marginal_result.values.reserve(vec_prod(marginal_result.cardinals));

    get_state_indices(factor_marginalize, var_index, factor_indices);
    UInt result_iter = 0u;
    for (std::size_t iter    = 0u; iter < factor_indices[0u].size(); iter++)
    {
      float result = 0.0f;
      for (std::size_t state = 0u; state < factor_indices.size(); state++)
      {
        result += factor_marginalize.values[factor_indices[state][iter]];
      }
      marginal_result.values.push_back(result);
    }
  }
}


void compute_joint(const std::vector<factor>& factors_vec, 
                        factor& jpd_result)
{
  if (factors_vec.size() >= 2u)
  {
    factor_product(factors_vec[0], factors_vec[1], jpd_result);
    factor temp;
    for (std::size_t iter = 2u; iter < factors_vec.size(); iter++)
    {
      factor_product(factors_vec[iter], jpd_result, temp);
      copy_factor(temp, jpd_result);
    }
  }
  else if (factors_vec.size() == 1u)
  {
    copy_factor(factors_vec[0], jpd_result);
  }
  else
  {
    std::cout << "Cannot compute Joint probability, given factor vector is empty";
  }
}


void observe_evidence(const std::vector<UIntVec>& evidence, 
                            std::vector<factor>& factor_vec)
{
  for (std::size_t evidence_iter = 0u; evidence_iter < evidence.size(); evidence_iter++)
  {
    UInt variable  = evidence[evidence_iter][0];
    UInt var_state = evidence[evidence_iter][1];

    for (std::size_t factor_iter = 0u; factor_iter < factor_vec.size(); factor_iter++)
    {
      int var_index = get_var_index(factor_vec[factor_iter], variable);
      if (var_index != -1)
      {
        if (   (var_state < factor_vec[factor_iter].cardinals[var_index]) 
            && (var_state >= 0u                                         ))
        {
          std::vector<UIntVec> var_state_indices;
          
          // \to-do: below function can be replaced to compute just the indices of the given var_state
          get_state_indices(factor_vec[factor_iter], var_index, var_state_indices);

          for (std::size_t state_iter = 0u; state_iter < var_state_indices.size(); state_iter++)
          {
            if (state_iter != var_state)
            {
              for (std::size_t indices_iter = 0u; indices_iter < var_state_indices[state_iter].size(); indices_iter++)
              {
                factor_vec[factor_iter].values[var_state_indices[state_iter][indices_iter]] = 0.0f;
              }
            }
          }
        }
      }
    }
  }
}


void factor_normalize(factor& factor_to_normalize)
{
  float probability_sum = vec_sum_n(factor_to_normalize.values, factor_to_normalize.values.size());
  vec_divide_n(factor_to_normalize.values, probability_sum, factor_to_normalize.values.size());
}


void compute_marginal(const std::vector<UInt>&    marginal_vars,
                      const std::vector<UIntVec>& evidence,
                      const std::vector<factor>&  factor_vec,
                            factor&               factor_marg)
{
  compute_joint(factor_vec, factor_marg);

  std::vector<factor> factor_marg_ref_vec = {factor_marg};
  observe_evidence(evidence, factor_marg_ref_vec);

  copy_factor(factor_marg_ref_vec[0], factor_marg);

  UIntVec var_to_marginalize;
  get_difference(factor_marg.variables, 
                 marginal_vars, 
                 var_to_marginalize);
  
  factor temp;
  for (const UInt& var: var_to_marginalize)
  {
    factor_marginalize(factor_marg, var, temp);
    copy_factor(temp, factor_marg);
  }
  factor_marg.cardinals.shrink_to_fit();
  factor_marg.variables.shrink_to_fit();
  factor_marg.values.shrink_to_fit();

  factor_normalize(factor_marg);
}

#endif
