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
 * @param rSeries an optional paramter where the seires resistance is stored
 * @return a vector with the drt values
 */
template<typename fv>
Eigen::VectorX<fv> calcDrt(Eigen::VectorX<std::complex<fv>>& impedanceSpectra, Eigen::VectorX<fv>& omegaTensor,
	FitMetics& fm, const FitParameters& fp, fv* rSeries = nullptr);


/**
 * @brief calculate impedance from drt using eigen datatypes
 *
 * @tparam fv precision to be used, either double or float
 * @param drt the drt to caluclate impedance from
 * @param omegaVector vector with the omega values that the impedances where mesured at
 * @param rSeries an optional paramter where the seires resistance is stored
 * @return a vector with the drt values
 */
template<typename fv>
Eigen::VectorX<std::complex<fv>> calcImpedance(const Eigen::VectorX<fv>& drt, fv rSeries, const  Eigen::VectorX<fv>& omegaVector);

/**
....
* @}
*/
