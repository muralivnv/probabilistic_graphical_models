#ifndef _BN_TYPES_H_
#define _BN_TYPES_H_

#include <vector>

typedef unsigned int UInt;
typedef std::vector<unsigned int> UIntVec;
typedef std::vector<unsigned int>::const_iterator UIntVecConstIter;

namespace BN
{
  
struct networkNode{
  unsigned int node_index;
  std::vector<std::shared_ptr<networkNode>> children;
  networkNode(const unsigned int index)
  { node_index = index; }
};

struct factor{
  // indices of each variables
  // Example: {0,2,4}
  std::vector<unsigned int> variables;

  // cardinality(number of states) for each variable
  // Example: Binary (0 or 1) => 2 states,...
  std::vector<unsigned int> cardinals;

  // probability factors for a combination of each variable states
  // arranged as, left to right order in the variable list,
  // Example: for a 3, binary state variable, order is as follows
  // [A1, B1, C1], [A2, B1, C1], [A1, B2, C1], [A2, B2, C1], [A1, B1, C2], [A2, B2, C2]
  std::vector<float> values;
};

} // end namespace {BN}

#endif