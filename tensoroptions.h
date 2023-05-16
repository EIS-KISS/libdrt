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
#include <c10/core/ScalarType.h>
#include <torch/torch.h>

template <typename V>
inline torch::TensorOptions tensorOptCpu(bool grad = true)
{
	static_assert(std::is_same<V, float>::value || std::is_same<V, double>::value,
				  "This function can only be passed double or float types");
	torch::TensorOptions options;
	if constexpr(std::is_same<V, float>::value)
		options = options.dtype(torch::kFloat32);
	else
		options = options.dtype(torch::kFloat64);
	options = options.layout(torch::kStrided);
	options = options.device(torch::kCPU);
	options = options.requires_grad(grad);
	return options;
}

template <typename V>
inline torch::TensorOptions tensorOptCplxCpu(bool grad = true)
{
	static_assert(std::is_same<V, float>::value || std::is_same<V, double>::value,
				  "This function can only be passed double or float types");
	torch::TensorOptions options;
	if constexpr(std::is_same<V, float>::value)
		options = options.dtype(torch::kComplexFloat);
	else
		options = options.dtype(torch::kComplexDouble);
	options = options.layout(torch::kStrided);
	options = options.device(torch::kCPU);
	options = options.requires_grad(grad);
	return options;
}
