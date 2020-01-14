#ifndef _UTIL_H_
#define _UTIL_H_

#include <iostream>
#include <vector>

#include "BN_types.h"

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec)
{
  for (const T& vec_elem: vec)
  {  std::cout << vec_elem << " ";  }
  return os;
}


template<typename T>
void copy_vals(const T& container_source, T& dest)
{
  std::copy(container_source.begin(), container_source.end(), dest.begin());
}

void copy_factor(const factor& source, factor& dest)
{
  dest.variables = source.variables;
  dest.cardinals = source.cardinals;
  dest.values = source.values;
}

template<typename T>
T vec_prod(const std::vector<T>& vec)
{
  T result = static_cast<T>(1);
  for (const T& elem: vec)
  {
    result *= elem;
  }
  return result;
}


template<typename T>
T vec_prod_n(const std::vector<T>& vec, const std::size_t num_elems_to_prod)
{
  T result = static_cast<T>(1);
  for (std::size_t iter = 0u; iter < num_elems_to_prod; iter++)
  {
    result *= vec[iter];
  }
  return result;
}


template<typename T>
T vec_sum_n(const std::vector<T>& vec, 
            const std::size_t num_elems_to_prod)
{
  T result  = static_cast<T>(0);
  for (std::size_t iter = 0u; iter < num_elems_to_prod; iter++)
  {
    result += vec[iter];
  }
  return result;
}


template<typename T, typename U>
void vec_divide_n(std::vector<T>& vec, const U constant, const std::size_t num_elems_to_div)
{
  for (std::size_t iter = 0u; iter < num_elems_to_div; iter++)
  {
    vec[iter] /= constant;
  }
}


template <typename T>
std::vector<T> arange(const T start, const T end, const T step)
{
  std::vector<T> range_of_elems((end - start)/step + 1, static_cast<T>(0));
  std::size_t iter = 0u;
  for (T elem = start; elem <= end; elem += step)
  {
    range_of_elems[iter] = elem;
    iter++;
  }
  return range_of_elems;
}


template<typename T>
void add_range(std::vector<T>& vec, const T start, const T end, const T step)
{
  vec.reserve(vec.size() + (end - start)/step);
  for (T iter = start; iter <= end; iter += step)
  {  vec.push_back(iter);  }

  vec.shrink_to_fit();
}


std::ostream& operator<<(std::ostream& os, 
                         const factor& factor_to_output)
{
  os << "vars:       "  << factor_to_output.variables << '\n';
  os << "cardinality: " << factor_to_output.cardinals << '\n';
  os << "Cpd:         " << factor_to_output.values    << '\n';
  return os;
}


std::ostream& operator<<(std::ostream& os, 
                         const factor * const factor_to_output)
{
  os << "vars:       "  << factor_to_output->variables << '\n';
  os << "cardinality: " << factor_to_output->cardinals << '\n';
  os << "Cpd:         " << factor_to_output->values    << '\n';
  return os;
}

#endif
