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
Api for use with eisgenerator applications
* @defgroup EISAPI eisgenerator API
* calculates drts with eisgenerator datatypes
* @{
*/

/**
 * @brief calculate a drt on eisgenerator types
 *
 * @param data a vector of eisgenerator datapoints with the values to your expirament, embedded omega values are ignored
 * @param omegaVector vector with the omega values that the impedances where mesured at
 * @param fm a fit metrics struct where this function returns information on the fit aquired
 * @param fp a struct with fit parameters
 * @return a vector with the drt values
 */
std::vector<fvalue> calcDrt(const std::vector<eis::DataPoint>& data, const std::vector<fvalue>& omegaVector, FitMetics& fm, const FitParameters& fp);

/**
 * @brief calculate a drt on eisgenerator types
 *
 * @param data a vector of eisgenerator datapoints with the values to your expirament, embedded omega values are used
 * @param fm a fit metrics struct where this function returns information on the fit aquired
 * @param fp a struct with fit parameters
 * @return a vector with the drt values
 */
std::vector<fvalue> calcDrt(const std::vector<eis::DataPoint>& data, FitMetics& fm, const FitParameters& fp);

/**
....
* @}
*/
