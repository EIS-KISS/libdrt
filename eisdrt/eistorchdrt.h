#include <eisgenerator/eistype.h>

#include "torchdrt.h"

/**
....
* @addtogroup TORCHAPI
*
* @{
*/

torch::Tensor calcDrtTorch(const std::vector<eis::DataPoint>& data, const std::vector<fvalue>& omegaVector, FitMetics& fm,  const FitParameters& fp);

torch::Tensor calcDrtTorch(const std::vector<eis::DataPoint>& data, FitMetics& fm,  const FitParameters& fp);

/**
....
* @}
*/
