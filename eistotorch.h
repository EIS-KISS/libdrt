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

#include <kisstype/type.h>
#include <torch/torch.h>

#include <vector>

torch::Tensor eisToComplexTensor(const std::vector<eis::DataPoint>& data, torch::Tensor* freqs = nullptr);
torch::Tensor eisToTorch(const std::vector<eis::DataPoint>& data, torch::Tensor* freqs = nullptr);
std::vector<eis::DataPoint> torchToEis(const torch::Tensor& input);
torch::Tensor fvalueVectorToTensor(std::vector<fvalue>& vect);
