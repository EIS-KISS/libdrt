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
#include <exception>
#include <string>

/**
Types for use with all eisdrt apis
* @defgroup TYPES types
* Types for use with all eisdrt apis
* @{
*/


/**
 * @brief This exception thrown if drt could not be calculated.
 */
class drt_error: public std::exception
{
	std::string whatStr;
public:
	drt_error(const std::string& whatIn): whatStr(whatIn)
	{}
	virtual const char* what() const noexcept override
	{
		return whatStr.c_str();
	}
};

/**
 * @brief This is used to return information on a fit.
 */
struct FitMetrics
{
	int iterations;	/**< how many iterations where used */
	double fx;		/**< error function value remaining after fit */
	bool compleated;	/**< true if fit completed successfully */
};

struct FitParameters
{
	int maxIter;
	double epsilon;
	double step;
	FitParameters(int maxIterI, double epsilonI = 1e-2, double stepI = 0.001): maxIter(maxIterI), epsilon(epsilonI), step(stepI){}
};

/**
....
* @}
*/
