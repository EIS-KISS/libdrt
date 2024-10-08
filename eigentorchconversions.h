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

#include <c10/core/ScalarType.h>
#include <climits>
#include <sys/types.h>
#include <torch/torch.h>
#include <Eigen/Dense>
#include <torch/types.h>
#include <vector>
#include <complex>

#include "tensoroptions.h"

template <typename V>
bool checkTorchType(const torch::Tensor& tensor)
{
	static_assert(std::is_same<V, float>::value ||
		std::is_same<V, double>::value ||
		std::is_same<V, int64_t>::value ||
		std::is_same<V, int32_t>::value ||
		std::is_same<V, int8_t>::value ||
		std::is_same<V, std::complex<float>>::value ||
		std::is_same<V, std::complex<double>>::value,
				  "This function dose not work with this type");
	if constexpr(std::is_same<V, float>::value)
		return tensor.dtype() == torch::kFloat32;
	else if constexpr(std::is_same<V, double>::value)
		return tensor.dtype() == torch::kFloat64;
	else if constexpr(std::is_same<V, int64_t>::value)
		return tensor.dtype() == torch::kInt64;
	else if constexpr(std::is_same<V, int32_t>::value)
		return tensor.dtype() == torch::kInt32;
	else if constexpr(std::is_same<V, int8_t>::value)
		return tensor.dtype() == torch::kInt8;
	else if constexpr(std::is_same<V, std::complex<float>>::value)
		return tensor.dtype() == torch::kComplexFloat;
	else if constexpr(std::is_same<V, std::complex<double>>::value)
		return tensor.dtype() == torch::kComplexDouble;
}

template <typename V>
using MatrixXrm = typename Eigen::Matrix<V, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template <typename V>
torch::Tensor eigen2libtorch(Eigen::MatrixX<V> &M)
{
	Eigen::Matrix<V, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> E(M);
	std::vector<int64_t> dims = {E.rows(), E.cols()};
	auto T = torch::from_blob(E.data(), dims, tensorOptCpu<V>(false)).clone();
	return T;
}

template <typename V>
torch::Tensor eigen2libtorch(MatrixXrm<V> &E, bool copydata = true)
{
	std::vector<int64_t> dims = {E.rows(), E.cols()};
	auto T = torch::from_blob(E.data(), dims, tensorOptCpu<V>(false));
	if (copydata)
		return T.clone();
	else
		return T;
}

template <typename V>
torch::Tensor eigenVector2libtorch(Eigen::Vector<V, Eigen::Dynamic> &E, bool copydata = true)
{
	std::vector<int64_t> dims = {E.rows()};
	auto T = torch::from_blob(E.data(), dims, tensorOptCpu<V>(false));
	if (copydata)
		return T.clone();
	else
		return T;
}

template<typename V>
Eigen::Matrix<V, Eigen::Dynamic, Eigen::Dynamic> libtorch2eigenMaxtrix(torch::Tensor &Tin)
{
	static_assert(!std::is_same<V, std::complex<float>>::value,
		"libtorch2eigenMaxtrix can not be used for complex tensors use libtorch2eigenMaxtrixComplex instead");
	/*
	LibTorch is Row-major order and Eigen is Column-major order.
	MatrixXrm uses Eigen::RowMajor for compatibility.
	*/
	assert(checkTorchType<V>(Tin));
	Tin = Tin.contiguous();
	auto T = Tin.to(torch::kCPU);
	Eigen::Map<MatrixXrm<V>> E(T.data_ptr<V>(), T.size(0), T.size(1));
	return E;
}

template<typename V>
Eigen::Matrix<std::complex<V>, Eigen::Dynamic, Eigen::Dynamic> libtorch2eigenMaxtrixComplex(torch::Tensor &Tin)
{
	/*
	LibTorch is Row-major order and Eigen is Column-major order.
	MatrixXrm uses Eigen::RowMajor for compatibility.
	*/
	assert(checkTorchType<std::complex<V>>(Tin));
	auto T = Tin.contiguous().to(torch::kCPU);
	Eigen::Map<MatrixXrm<std::complex<V>>> E(reinterpret_cast<std::complex<V>*>(T.data_ptr<c10::complex<V>>()), T.size(0), T.size(1));
	return E;
}

template<typename V>
Eigen::Vector<V, Eigen::Dynamic> libtorch2eigenVector(torch::Tensor &Tin)
{
	assert(Tin.sizes().size() == 1);
	assert(checkTorchType<V>(Tin));
	Tin = Tin.contiguous();
	auto T = Tin.to(torch::kCPU);
	Eigen::Map<Eigen::Vector<V, Eigen::Dynamic>> E(T.data_ptr<V>(), T.numel());
	return E;
}
