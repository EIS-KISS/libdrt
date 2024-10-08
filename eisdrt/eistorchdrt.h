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

#include <kisstype/type.h>

#include "torchdrt.h"

/**
....
* @addtogroup TORCHAPI
*
* @{
*/

/**
 * @brief Calculates impedance from drt using torch datatypes.
 *
 * This function ignores the frequencies in data and uses those from omegaVector
 *
 * @param data The spectra to calculate impedance from.
 * @param omegaVector A vector of radial frequencies to calculate the drt at.
 * @param fm A fit metrics struct where this function returns information on the fit acquired.
 * @param fp A struct with fit parameters.
 * @return A complex tensor with the drt values.
 */
torch::Tensor calcDrtTorch(const std::vector<eis::DataPoint>& data, const std::vector<fvalue>& omegaVector, FitMetrics& fm,  const FitParameters& fp);

/**
 * @brief Calculates impedance from drt using torch datatypes.
 *
 * @param data The spectra to calculate impedance from.
 * @param fm A fit metrics struct where this function returns information on the fit acquired.
 * @param fp A struct with fit parameters.
 * @return A complex tensor with the drt values.
 */
torch::Tensor calcDrtTorch(const std::vector<eis::DataPoint>& data, FitMetrics& fm,  const FitParameters& fp);


/**
 * @brief Calculates impedance from drt using eisgenerator datatypes.
 *
 * @param drt The drt to calculate impedance from.
 * @param omegaRange A range that describes the omega values the drt was taken at.
 * @param rSeries An optional parameter where the series resistance is stored.
 * @return A vector with the impedance values.
 */
torch::Tensor calcImpedance(torch::Tensor& drt, fvalue rSeries, const std::vector<fvalue>& omegaVector);

/**
....
* @}
*/
