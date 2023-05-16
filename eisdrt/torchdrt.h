#include <torch/torch.h>

#include "types.h"

/**
Api for use with libtorch applications
* @defgroup TORCHAPI libtorch API
* calculates drts with libtorch datatypes
* @{
*/

/**
 * @brief calculate a drt on eisgenerator types
 *
 * @tparam fv precision to be used, either double or float
 * @param impedanceSpectra a 1d complex tensor with the impedance mesurement data points
 * @param omegaTensor a 1d tensor with the omega values that the impedances where mesured at
 * @param fm a fit metrics struct where this function returns information on the fit aquired
 * @param fp a struct with fit parameters
 * @return a 1d tensor with the drt values
 */
template<typename fv>
torch::Tensor calcDrtTorch(torch::Tensor& impedanceSpectra, torch::Tensor& omegaTensor, FitMetics& fm, const FitParameters& fp);


/**
....
* @}
*/
