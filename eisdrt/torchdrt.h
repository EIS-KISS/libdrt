/* * libeisdrt - A library to calculate EIS Drts
 * Copyright (C) 2023 Carl Klemm <carl@uvos.xyz>
 *
 * This file is part of libeisdrt.
 *
 * libeisdrt is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * libeisdrt is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with libeisdrt.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <torch/torch.h>

#include "types.h"

/**
Api for use with libtorch applications
* @defgroup TORCHAPI libtorch API
* calculates drts with libtorch datatypes
* @{
*/

/**
 * @brief Calculates a drt on eisgenerator types.
 *
 * @tparam fv The Precision to be used, either double or float.
 * @param impedanceSpectra A 1d complex tensor with the impedance measurement data points.
 * @param omegaTensor A 1d tensor with the omega values that the impedances where measured at.
 * @param fm A fit metrics struct where this function returns information on the fit acquired.
 * @param fp A struct with fit parameters.
 * @return A 1d tensor with the drt values.
 */
template<typename fv>
torch::Tensor calcDrtTorch(torch::Tensor& impedanceSpectra, torch::Tensor& omegaTensor, FitMetrics& fm, const FitParameters& fp);

/**
 * @brief Calculates impedance from drt.
 *
 * @tparam fv The Precision to be used, either double or float.
 * @param drt The drt to calculate impedance from.
 * @return A complex tensor with the impedance spectra.
 */
template<typename fv>
torch::Tensor calcImpedance(torch::Tensor& drt, fv rSeries, torch::Tensor& omegaVector);

/**
....
* @}
*/
