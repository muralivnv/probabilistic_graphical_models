#ifndef _CRF_TYPEDEF_H_
#define _CRF_TYPEDEF_H_

#include <vector>
#include <array>
#include <tuple>
#include <string>
#include <string_view>
#include <charconv>
#include <unordered_map>
#include <any>

#include <fstream>
#include <chrono>

#include <algorithm>
#include <numeric>

#pragma warning(push, 0)        
// includes with unfixable warnings
#include <Eigen/Core>
#pragma warning(pop)

#include <LBFGS.h>

#define CHAR_IMG_WIDTH  (15u)
#define CHAR_IMG_HEIGHT (25u)
#define N_STATES        (26u)

using std::vector, std::string, std::size_t, std::array, std::fstream, std::tuple, std::tie;
using std::for_each, std::copy, std::transform;
using Eigen::all, Eigen::seq, Eigen::last;

namespace eig    = Eigen;
namespace chrono = std::chrono;
namespace lbfgspp = LBFGSpp;
using namespace std::string_literals;

#define ITER(X) std::begin(X), std::end(X)
#define TIME_NOW chrono::system_clock::now()
#define str2float(str, result) std::from_chars(str.data(), str.data()+str.length(), result)

namespace crf {
template<typename T>
using MatrixX = eig::Matrix<T, eig::Dynamic, eig::Dynamic, eig::RowMajor>; 

template<typename T, size_t Rows, size_t Cols>
using Matrix = eig::Matrix<T, Rows, Cols, eig::RowMajor>;

template<typename T>
using VectorX = eig::Matrix<T, eig::Dynamic, 1, eig::RowMajor>;

using Word_t = vector< crf::MatrixX<float> >;
using Words_t = vector<Word_t>;

using NodeWeights_t  = crf::Matrix<float, N_STATES, CHAR_IMG_HEIGHT*CHAR_IMG_WIDTH>;
using TransWeights_t = crf::Matrix<float, N_STATES, N_STATES>;

using MinimizerParams = std::unordered_map<std::string, std::any>;

struct Graph{
  crf::MatrixX<float> WX;
  crf::MatrixX<float> unnormalized_PY1X;
  crf::MatrixX<float> alpha;
  crf::MatrixX<float> beta;
  eig::VectorXf       scaling_factors;

  Graph(size_t n_states, size_t seq_len)
  {
    WX.resize(n_states, seq_len);
    unnormalized_PY1X.resize(n_states, seq_len); 
    alpha.resize(n_states, seq_len); 
    beta.resize(n_states, seq_len);
    scaling_factors.resize(seq_len);
  }
};

struct ViterbiNode{
  size_t best_parent;
  float probability;
};


} // namespace {crf}


#endif
