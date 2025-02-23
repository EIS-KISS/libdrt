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

// SPDX-License-Identifier: lgpl3

#pragma once

#include "Eigen/src/Core/Matrix.h"
#include <Eigen/Core>
#include <kisstype/type.h>
#include <vector>

Eigen::VectorX<std::complex<fvalue>> eistoeigen(const std::vector<eis::DataPoint>& data, Eigen::Vector<fvalue, Eigen::Dynamic>* omega = nullptr)
{
	Eigen::VectorX<std::complex<fvalue>> out(data.size());

	if(omega)
		*omega = Eigen::VectorX<fvalue>(data.size());

	for(size_t i = 0; i < data.size(); ++i)
	{
		out[i] = data[i].im;
		if(omega)
			(*omega)[i] = data[i].omega;
	}
	return out;
}

std::vector<eis::DataPoint> eigentoeis(const Eigen::VectorX<std::complex<fvalue>>& data, const Eigen::Vector<fvalue, Eigen::Dynamic>* omega = nullptr)
{
	assert(!omega || omega->size() == data.size());
	std::vector<eis::DataPoint> out(data.size());

	for(ssize_t i = 0 ; i < data.size(); ++i)
	{
		out[i].im = data[i];
		if(omega)
			out[i].omega = (*omega)[i];
		else
			out[i].omega = -1;
	}
	return out;
}
