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

#include <eisgenerator/eistype.h>
#include <vector>

#include "types.h"

/**
API for use with eisgenerator applications
* @defgroup EISAPI eisgenerator API
* calculates drts with eisgenerator datatypes
* @{
*/

/**
 * @brief calculate a drt on eisgenerator types
 *
 * @param data a vector of eisgenerator data points with the values to your experiment, embedded omega values are ignored
 * @param omegaVector vector with the omega values that the impedances where measured at
 * @param fm a fit metrics struct where this function returns information on the fit acquired
 * @param fp a struct with fit parameters
 * @param rSeries an optional parameter where the series resistance is stored
 * @return a vector with the drt values
 */
std::vector<fvalue> calcDrt(const std::vector<eis::DataPoint>& data, const std::vector<fvalue>& omegaVector, FitMetics& fm,
	const FitParameters& fp, fvalue* rSeries = nullptr);

/**
 * @brief calculate a drt on eisgenerator types
 *
 * @param data a vector of eisgenerator data points with the values to your experiment, embedded omega values are used
 * @param fm a fit metrics struct where this function returns information on the fit acquired
 * @param fp a struct with fit parameters
 * @param rSeries an optional parameter where the series resistance is stored
 * @return a vector with the drt values
 */
std::vector<fvalue> calcDrt(const std::vector<eis::DataPoint>& data, FitMetics& fm, const FitParameters& fp, fvalue* rSeries = nullptr);

/**
 * @brief calculate impedance from drt using eisgenerator datatypes
 *
 * @param drt the drt to calculate impedance from
 * @param omegaVector vector with the omega values that the impedances where measured at
 * @param rSeries an optional parameter where the series resistance is stored
 * @return a vector with the impedance values
 */
std::vector<eis::DataPoint> calcImpedance(const std::vector<fvalue>& drt, fvalue rSeries, const std::vector<fvalue>& omegaVector);

/**
 * @brief calculate impedance from drt using eisgenerator datatypes
 *
 * @param drt the drt to calculate impedance from
 * @param omegaRange range that describes the omega values the drt was taken at
 * @param rSeries an optional parameter where the series resistance is stored
 * @return a vector with the impedance values
 */
std::vector<eis::DataPoint> calcImpedance(const std::vector<fvalue>& drt, fvalue rSeries, const eis::Range& omegaRange);

/**
....
* @}
*/
