//
// libeisdrt - A library to calculate EIS Drts
// Copyright (C) 2023 Carl Klemm <carl@uvos.xyz>
//
// This file is part of libeisdrt.
//
// libeisdrt is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// libeisdrt is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with libeisdrt.  If not, see <http://www.gnu.org/licenses/>.
//

#include "eisdrt/torchdrt.h"

#include <cassert>

#include "eisdrt/eigendrt.h"
#include "eigentorchconversions.h"

#ifdef USE_EISGEN
#include "eistoeigen.h"
#endif


template<typename fv>
torch::Tensor calcDrtTorch(torch::Tensor& impedanceSpectra, torch::Tensor& omegaTensor, FitMetrics& fm, const FitParameters& fp)
{
	assert(checkTorchType<std::complex<fv>>(impedanceSpectra));
	Eigen::Vector<std::complex<fv>, Eigen::Dynamic> impedanceSpectraEigen =
		libtorch2eigenMaxtrixComplex<fv>(impedanceSpectra);

	Eigen::Vector<fv, Eigen::Dynamic> omegaEigen =
		libtorch2eigenVector<fv>(omegaTensor);

	Eigen::VectorX<fv> drt = calcDrt(impedanceSpectraEigen, omegaEigen, fm, fp);
	torch::Tensor outputTensor = eigenVector2libtorch(drt);
	return outputTensor;
}

#ifdef USE_EISGEN
torch::Tensor calcDrtTorch(const std::vector<eis::DataPoint>& data, const std::vector<fvalue>& omegaVector, FitMetrics& fm,  const FitParameters& fp)
{
	Eigen::VectorX<std::complex<fvalue>> impedanceSpectra = eistoeigen(data);
	Eigen::VectorX<fvalue> omega = Eigen::VectorX<fvalue>::Map(omegaVector.data(), omegaVector.size());
	Eigen::VectorX<fvalue> drt = calcDrt<fvalue>(impedanceSpectra, omega, fm, fp);
	torch::Tensor outputTensor = eigenVector2libtorch(drt);
	return outputTensor;
}

torch::Tensor calcDrtTorch(const std::vector<eis::DataPoint>& data, FitMetrics& fm,  const FitParameters& fp)
{
	Eigen::VectorX<fvalue> omega;
	Eigen::VectorX<std::complex<fvalue>> impedanceSpectra = eistoeigen(data, &omega);
	Eigen::VectorX<fvalue> drt = calcDrt<fvalue>(impedanceSpectra, omega, fm, fp);
	torch::Tensor outputTensor = eigenVector2libtorch(drt);
	return outputTensor;
}
#endif
