#include <iostream>
#include <vector>

#include "BN_types.h"
#include "BN_operations.h"

#define NO_PARENT (-1)
#define PARENT_1  (0u)
#define PARENT_2  (1u)

void phenotype_mendelianModel(bool is_trait_from_dominant_allele, 
                              const UInt genotype_var, 
                              const UInt phenotype_var, 
                              BN::factor& phenotype_factor)
{
  /*
  Genotype has 3 states, 
    0->combination of dominant alleles (homozygous dominant) 
    1->combination of dominant and recessive alleles (heterogeneous dominant)
    2->combination of recessive alleles (homozygous recessive)

  Phenotype has 2 states,
    0->having a physical trait
    1->not having a physical trait
  
  so is allele (trait is caused by a dominant allele), 
  then the phenotype will be true for dominant allele genotype states
  */
  phenotype_factor.variables = std::vector<UInt>{phenotype_var, genotype_var};
  phenotype_factor.cardinals = std::vector<UInt>{2u, 3u};
  const UInt num_values = util::vec_prod(phenotype_factor.cardinals);
  phenotype_factor.values = std::vector<float>(num_values, 0.0f);

  std::vector<UIntVec> genotype_state_indices;
  BN::get_state_indices(phenotype_factor, 1u, genotype_state_indices);

  if (is_trait_from_dominant_allele == true)
  {
    // means states 0 and 1 will have a probabilities
    for (UInt state_iter = 0u; state_iter <= 1u; state_iter++)
    {
      for (const UInt state_value_index: genotype_state_indices[state_iter])
      {
        phenotype_factor.values[state_value_index] = 1.0F;
      }
    }
  }
  else
  {
    for (const UInt state_value_index : genotype_state_indices[2u])
    {
      phenotype_factor.values[state_value_index] = 1.0F;
    }
  }
}


void phenotype_nonMendelianModel(const std::vector<float>& genotype_prob_list, 
                                const UInt genotype_var,
                                const UInt phenotype_var,
                                BN::factor& phenotype_factor)
{
  /*
  Genotype has n states, each with probability given in genotype_prob_list for each state

  Phenotype has 2 states,
    0->having a physical trait
    1->not having a physical trait
  */
  phenotype_factor.variables = std::vector<UInt>{phenotype_var, genotype_var};
  phenotype_factor.cardinals = std::vector<UInt>{2u, static_cast<UInt>(genotype_prob_list.size())};
  const UInt num_values = util::vec_prod(phenotype_factor.cardinals);
  phenotype_factor.values = std::vector<float>(num_values, 0.0F);

  std::vector<UIntVec> genotype_state_indices;
  BN::get_state_indices(phenotype_factor, 1u, genotype_state_indices);

  float total_cpd_sum = 0.0f;
  for (std::size_t state = 0u; state < genotype_state_indices.size(); state++)
  {
    // we will only have 2-state indices for each state of genotype {Phenotype 0, Phenotype 1}
    phenotype_factor.values[genotype_state_indices[state][0]] = genotype_prob_list[state];
    phenotype_factor.values[genotype_state_indices[state][1]] = 1.0F - genotype_prob_list[state];

    total_cpd_sum += phenotype_factor.values[genotype_state_indices[state][0]];
    total_cpd_sum += phenotype_factor.values[genotype_state_indices[state][1]];
  }

  // normalize
  util::vec_divide_n(phenotype_factor.values, total_cpd_sum, phenotype_factor.values.size());
}


void genotype_alleleFreq(const std::vector<float>& allele_freq, 
                         const UInt genotype_var,
                         BN::factor& genotype_factor)
{
  /*
  for n-different allele, assuming genotype of 2-alleles, we will have
  nc2 (heterozygotes) + n (homozygotes) genes

  genotypes are arranged as follows, for alleles of type A,B,C
  G0=AA, G1=AB, G2=AC, G3=BB, G4=BC, G5=CC
  */
  const std::size_t num_of_alleles = allele_freq.size();
  const UInt num_of_genotypes = num_of_alleles*(num_of_alleles-1u)/2u + num_of_alleles;

  genotype_factor.variables = std::vector<UInt>{genotype_var};
  genotype_factor.cardinals = std::vector<UInt>{num_of_genotypes};
  genotype_factor.values    = std::vector<float>(num_of_genotypes, 0.0f);

  // fill genotype probabilities
  UInt genotype_iter  = 0u;
  float total_cpd_sum = 0.0f; 
  for (std::size_t iter_outer = 0u; iter_outer < num_of_alleles; iter_outer++)
  {
    genotype_factor.values[genotype_iter] = allele_freq[iter_outer]*allele_freq[iter_outer];
    total_cpd_sum += genotype_factor.values[genotype_iter];
    genotype_iter++;

    for (std::size_t iter_inner = iter_outer + 1; iter_inner < num_of_alleles; iter_inner++)
    {
      genotype_factor.values[genotype_iter] = allele_freq[iter_outer]*allele_freq[iter_inner];
      total_cpd_sum += genotype_factor.values[genotype_iter];
      genotype_iter++;
    }
  }

  // normalize
  util::vec_divide_n(genotype_factor.values, total_cpd_sum, genotype_factor.values.size());
}


void get_allele_from_genotype(const UInt genotype, 
                              const UInt num_of_alleles, 
                              std::vector<UInt>& allele_list)
{
  // assuming genotype of 2 alleles
  
  if (allele_list.size() < 2u)
  {  allele_list = std::vector<UInt>(2);  }

  // genotypes are arranged as follows, for alleles of type A,B,C
  // G0=AA, G1=AB, G2=AC, G3=BB, G4=BC, G5=CC
  allele_list[0] = static_cast<UInt>(genotype/num_of_alleles);
  allele_list[1] = genotype - allele_list[0];
}

void get_genotype_from_allele(const UInt allele1, 
                              const UInt allele2, 
                              const UInt num_of_alleles, 
                              UInt& genotype)
{
  genotype = allele1*num_of_alleles + allele2;
}

void genotype_parentsGenotype(const UInt num_alleles, 
                              const UInt child_genotype_var,
                              const UInt parent1_genotype_var,
                              const UInt parent2_genotype_var,
                              BN::factor& child_genotype_factor)
{
  /* for more details look at the link below 
   * https://www2.palomar.edu/anthro/mendel/mendel_2.htm
  */

  const UInt num_of_genotype = num_alleles * (num_alleles - 1u)/2u + num_alleles;

  child_genotype_factor.variables = std::vector<UInt>{child_genotype_var, 
                                                      parent1_genotype_var, 
                                                      parent2_genotype_var};

  child_genotype_factor.cardinals = std::vector<UInt>(3, num_of_genotype);
  const UInt cpd_elem_count = util::vec_prod(child_genotype_factor.cardinals);
  child_genotype_factor.values = std::vector<float>(cpd_elem_count, 0.0f);

  UInt child_genotype_counter = 0u;
  // iterate over each genotype of parent1
  for (UInt parent1_genotype_iter = 0u; parent1_genotype_iter < num_of_genotype; parent1_genotype_iter++)
  {
    // get this genotype alleles for parent1
    std::vector<UInt> parent1_allele(2);
    get_allele_from_genotype(parent1_genotype_iter, num_alleles, parent1_allele);

    // iterate over each genotype of parent2
    for (UInt parent2_genotype_iter = 0u; parent2_genotype_iter < num_of_genotype; parent2_genotype_iter++)
    {
      // get this genotype alleles for parent2
      std::vector<UInt> parent2_allele(2);
      get_allele_from_genotype(parent2_genotype_iter, num_alleles, parent2_allele);

      // now for different combinatio of alleles, create genotype and increment the necessary genotype prob in child
      for (const UInt& parent1_allele_elem:parent1_allele)
      {
        for (const UInt& parent2_allele_elem : parent2_allele)
        {
          UInt child_genotype_index;
          get_genotype_from_allele(parent1_allele_elem, parent2_allele_elem, num_alleles, child_genotype_index);
          child_genotype_factor.values[child_genotype_index] += 1.0F;
          child_genotype_counter += 1u;
        }
      }
    }
  }

  // normalize
  util::vec_divide_n(child_genotype_factor.values, (float)(child_genotype_counter), child_genotype_factor.values.size());
  std::vector<UIntVec> child_genotype_state_indices;

  // fill remaining repetitions
  BN::get_state_indices(child_genotype_factor, 0u, child_genotype_state_indices);
  for (const UIntVec& row : child_genotype_state_indices)
  {
    for (UInt row_iter = 1u; row_iter < row.size(); row_iter++)
    {
      child_genotype_factor.values[row[row_iter]] = child_genotype_factor.values[row[0]];
    }
  }
}

int main()
{
  // create BN
  std::vector<float> allele_freq {0.1f, 0.9f};
  std::vector<float> genotype_prob_list{0.8f, 0.6f, 0.1f};
  std::vector<std::vector<int> > node_link_info{{NO_PARENT, NO_PARENT}, 
                                                {0,   2}, 
                                                {NO_PARENT, NO_PARENT}, 
                                                {0,   2}, 
                                                {1,   5}, 
                                                {NO_PARENT, NO_PARENT}, 
                                                {1,   5}, 
                                                {3,   8}, 
                                                {NO_PARENT, NO_PARENT}};

  const UInt num_people = static_cast<UInt>(node_link_info.size());
  std::vector<factor> genotype_factor_vec(num_people);
  std::vector<factor> phenotype_factor_vec(num_people);
  UInt var_iter = 0u;
  for (const std::vector<int>& node: node_link_info)
  {
    // if no parent, create genotype factor from allele freqs
    if (node[PARENT_1] == NO_PARENT)
    {
      genotype_alleleFreq(allele_freq, 
                          var_iter, 
                          genotype_factor_vec[var_iter]);
    }
    else // we have parent, create genotype factor given parents genotype
    {
      genotype_parentsGenotype(static_cast<UInt>(allele_freq.size()), 
                               var_iter, node[PARENT_1], node[PARENT_2], 
                               genotype_factor_vec[var_iter]);
    }
    
    phenotype_nonMendelianModel(genotype_prob_list, 
                                var_iter, var_iter + num_people, 
                                phenotype_factor_vec[var_iter]);
    
    std::cout << "Genotype: " << genotype_factor_vec[var_iter] << "\nPhenotype: " << phenotype_factor_vec[var_iter] << "\n";

    var_iter++;
  }
  

  return EXIT_SUCCESS;
}