#pragma once

#include <torch/torch.h>

namespace LBFGSpp {

typedef torch::Tensor Vector;
typedef torch::Tensor Matrix;
typedef std::vector<int> IndexSet;

}
