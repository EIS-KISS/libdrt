#include "eisdrt/torchdrt.h"

#include <cassert>

#include "eisdrt/eigendrt.h"
#include "eigentorchconversions.h"

#ifdef USE_EISGEN
#include "eistoeigen.h"
#endif


template<typename fv>
torch::Tensor calcDrtTorch(torch::Tensor& impedanceSpectra, torch::Tensor& omegaTensor, FitMetics& fm, const FitParameters& fp)
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
torch::Tensor calcDrtTorch(const std::vector<eis::DataPoint>& data, const std::vector<fvalue>& omegaVector, FitMetics& fm,  const FitParameters& fp)
{
	Eigen::VectorX<std::complex<fvalue>> impedanceSpectra = eistoeigen(data);
	Eigen::VectorX<fvalue> omega = Eigen::VectorX<fvalue>::Map(omegaVector.data(), omegaVector.size());
	Eigen::VectorX<fvalue> drt = calcDrt<fvalue>(impedanceSpectra, omega, fm, fp);
	torch::Tensor outputTensor = eigenVector2libtorch(drt);
	return outputTensor;
}

torch::Tensor calcDrtTorch(const std::vector<eis::DataPoint>& data, FitMetics& fm,  const FitParameters& fp)
{
	Eigen::VectorX<fvalue> omega;
	Eigen::VectorX<std::complex<fvalue>> impedanceSpectra = eistoeigen(data, &omega);
	Eigen::VectorX<fvalue> drt = calcDrt<fvalue>(impedanceSpectra, omega, fm, fp);
	torch::Tensor outputTensor = eigenVector2libtorch(drt);
	return outputTensor;
}
#endif
