#ifndef _BN_TYPES_H_
#define _BN_TYPES_H_

#include <vector>

typedef unsigned int UInt;
typedef std::vector<unsigned int> UIntVec;
typedef std::vector<unsigned int>::const_iterator UIntVecConstIter;

struct networkNode{
  unsigned int node_index;
  std::vector<std::shared_ptr<networkNode>> children;
  networkNode(const unsigned int index)
  { node_index = index; }
};

struct factor{
  std::vector<unsigned int> variables;
  std::vector<unsigned int> cardinals;
  std::vector<float> values;
};

#endif