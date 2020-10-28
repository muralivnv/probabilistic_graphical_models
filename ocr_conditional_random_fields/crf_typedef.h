#ifndef _CRF_TYPEDEF_H_
#define _CRF_TYPEDEF_H_

#include <vector>
#include <array>
#include <tuple>
#include <string>
#include <string_view>
#include <charconv>

#include <fstream>
#include <chrono>

#include <algorithm>
#include <numeric>

#pragma warning(push, 0)        
// includes with unfixable warnings
#include <Eigen/Eigen/Core>
#pragma warning(pop)

#define CHAR_IMG_WIDTH  (15u)
#define CHAR_IMG_HEIGHT (25u)
#define N_STATES        (26u)

using std::vector, std::string, std::size_t, std::array, std::fstream, std::tuple, std::tie;
using std::for_each, std::copy, std::transform;
using Eigen::all, Eigen::seq, Eigen::last;

namespace eig    = Eigen;
namespace chrono = std::chrono;
using namespace std::string_literals;

#define ITER(X) std::begin(X), std::end(X)
#define TIME_NOW chrono::system_clock::now()
#define str2float(str, result) std::from_chars(str.data(), str.data()+str.length(), result)

namespace crf {
template<typename T>
using MatrixX = eig::Matrix<T, eig::Dynamic, eig::Dynamic, eig::RowMajor>; 

template<typename T, size_t Rows, size_t Cols>
using Matrix = eig::Matrix<T, Rows, Cols, eig::RowMajor>;

using Word_t = vector< crf::MatrixX<float> >;
using Words_t = vector<Word_t>;

using NodeWeights_t  = crf::Matrix<float, N_STATES, CHAR_IMG_HEIGHT*CHAR_IMG_WIDTH>;
using TransWeights_t = crf::Matrix<float, N_STATES, N_STATES>;

enum StatusID{
  NOT_CALCULATED = 0u,
  CALCULATED = 1u
};

struct Graph{
  std::vector<crf::StatusID> potential_status;
  crf::MatrixX<float> WX;
  crf::MatrixX<float> unnormalized_PY1X;
  crf::MatrixX<float> unnormalized_PY1Y2X;
  crf::MatrixX<float> log_alpha;
  crf::MatrixX<float> log_beta;

  Graph(size_t n_states, size_t seq_len)
  {
    WX.resize(n_states, seq_len); 
    unnormalized_PY1X.resize(n_states, seq_len); 
    unnormalized_PY1Y2X.resize(n_states, seq_len); 
    log_alpha.resize(n_states, seq_len); 
    log_beta.resize(n_states, seq_len); 
    potential_status = std::vector<crf::StatusID>(seq_len, crf::NOT_CALCULATED);
  }
};

} // namespace {crf}


#endif
