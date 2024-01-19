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
* API for use with Eigen applications
* @defgroup EIGENAPI Eigen API
*
* Calculates drts with Eigen datatypes.
* @{
*/

/**
 * @brief Calculates the drt using Eigen datatypes.
 *
 * @tparam fv The precision to be used, either double or float.
 * @param impedanceSpectra A vector with the complex impedances of your experiment.
 * @param omegaTensor A vector with the omega values that the impedances where measured at.
 * @param fm A fit metrics struct where this function returns information on the fit acquired.
 * @param fp A struct with fit parameters.
 * @param rSeries An optional parameter where the series resistance is stored.
 * @return A vector with the drt values.
 */
template<typename fv>
Eigen::VectorX<fv> calcDrt(Eigen::VectorX<std::complex<fv>>& impedanceSpectra, Eigen::VectorX<fv>& omegaTensor,
	FitMetics& fm, const FitParameters& fp, fv* rSeries = nullptr);


/**
 * @brief Calculates impedance from drt using Eigen datatypes.
 *
 * @tparam fv The precision to be used, either double or float.
 * @param drt The drt to calculate impedance from.
 * @param omegaVector A vector with the omega values that the impedances where measured at.
 * @param rSeries An optional parameter where the series resistance is stored.
 * @return A vector with the drt values.
 */
template<typename fv>
Eigen::VectorX<std::complex<fv>> calcImpedance(const Eigen::VectorX<fv>& drt, fv rSeries, const  Eigen::VectorX<fv>& omegaVector);

/**
....
* @}
*/
