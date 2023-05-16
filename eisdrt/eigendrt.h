#pragma once
#include <Eigen/Core>

#include "types.h"

/**
Api for use with Eigen applications
* @defgroup EIGENAPI Eigen API
* calculates drts with eigen datatypes
* @{
*/

/**
 * @brief calculate the drt using Eigen datatypes
 *
 * @tparam fv precision to be used, either double or float
 * @param impedanceSpectra vector with the complex impedances of your expirament
 * @param omegaTensor vector with the omega values that the impedances where mesured at
 * @param fm a fit metrics struct where this function returns information on the fit aquired
 * @param fp a struct with fit parameters
 * @return a vector with the drt values
 */

template<typename fv>
Eigen::VectorX<fv> calcDrt(Eigen::VectorX<std::complex<fv>>& impedanceSpectra, Eigen::VectorX<fv>& omegaTensor, FitMetics& fm, const FitParameters& fp);

/**
....
* @}
*/
