#include <iostream>
#include <vector>
#include <algorithm>

#include "BN_types.h"
#include "BN_operations.h"
#include "util.h"

int main()
{
  /* 
  -- FACTOR PRODUCT -- 
  sample product factor test
  output should be, 
  'variables': {1, 2}, 'cardinales': {2, 2}, 'values': {0.0649, 0.1958, 0.0451, 0.6942}
  */
  factor sample_factor1 = make_factor_with_val({0}, {2}, {0.11f, 0.89f});
  factor sample_factor2 = make_factor_with_val({1, 0}, {2, 2}, {0.59f, 0.41f, 0.22f, 0.78f});

  factor product_result;
  factor_product(sample_factor1, sample_factor2, product_result);
  std::cout << "product_result: \n" << product_result;


  /*
  -- FACTOR MARGINALIZATION --
  output should be,
  'variables': {1}, 'cardinals': {2}, 'values': {1 1}
  for factor2
  */
  factor sample_factor3 = make_factor_with_val({2, 1}, {2, 2}, {0.39f, 0.61f, 0.06f, 0.94f});

  factor sample_factor4 = make_factor_with_val({0, 1, 2}, // vars
                                               {3, 2, 2}, // cardinals
                                               {0.25f, 0.05f, 0.15f, 
                                                0.08f, 0.0f, 0.09f,
                                                0.35f, 0.07f, 0.21f,
                                                0.16f, 0.0, 0.18f} // values
                                                );
  factor sample_factor4_marginal;
  factor_marginalize(sample_factor4, 1, sample_factor4_marginal);
  std::cout << "marginalized_result: \n" << sample_factor4_marginal;
  

  std::vector<factor> factor_vec {sample_factor1, sample_factor2, sample_factor3};
  
  /* Compute Joint probability */
  
  factor jpd_result;
  compute_joint(factor_vec, jpd_result);
  std::cout << "jpd: \n" << jpd_result;

  /* Compute marginal over a vector of factors with evidence to remove */
  factor marginal_factor;
  compute_marginal({1, 2}, {{0, 1}}, factor_vec, marginal_factor);
  std::cout << "Marginalized with evidence: \n" << marginal_factor << '\n';

  /* Given an evidence, {variable, state}, this function removes those instance */
  observe_evidence({ {1u, 0u}, {2u, 1u} }, factor_vec);
  std::cout << "after observing evidence: \n" << factor_vec << '\n';

}