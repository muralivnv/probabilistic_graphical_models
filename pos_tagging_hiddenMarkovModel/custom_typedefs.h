#ifndef _CUSTOM_TYPEDEFS_H_
#define _CUSTOM_TYPEDEFS_H_

#include <vector>
#include <array>
#include <unordered_map>

using uint_t = unsigned int;

template<typename T, std::size_t n_rows>
using VEC       = std::array<T, n_rows>;      // fixed size vector
using DVEC_STR  = std::vector<std::string>;   // dynamic vector of type std::strings
using DVEC_UINT = std::vector<uint_t>;        // dynamic vector of type unsigned int
using DVEC_INT  = std::vector<int>;           // dynamic vector of type int

template<typename T, std::size_t n_rows, std::size_t n_cols>
using MAT        = std::array<std::array<T, n_cols>, n_rows>; // fixed size matrix
using DMAT_STR   = std::vector<std::vector<std::string>>;     // dynamic matrix of type std::string
using DMAT_UINT  = std::vector<std::vector<uint_t>>;          // dynamic matrix of type unsigned int

template <typename KeyType, typename ValueType>
using HASH_MAP = std::unordered_map<KeyType, ValueType>;

struct ViterbiPathNode{
  uint_t prev_state = 0u;
  float max_prob = 0.0F;
};

#endif
