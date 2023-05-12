#include "eisdrt/drt.h"

#include <ATen/ops/ones.h>
#include <ATen/ops/zeros.h>
#include <Eigen/Core>
#include <eisgenerator/eistype.h>

#include "tensoroptions.h"
#include "eigentorchconversions.h"
#include "eistotorch.h"
#include "LBFG/LBFGSB.h"

static torch::Tensor guesStartingPoint(torch::Tensor& omega, torch::Tensor& impedanceSpectra)
{
	std::vector<int64_t> size = omega.sizes().vec();
	++size[0];
	torch::Tensor startingPoint = torch::zeros(size, tensorOptCpu<fvalue>(false));
	startingPoint[-1] = torch::abs(impedanceSpectra[-1]);
	return startingPoint;
}

static torch::Tensor aImag(torch::Tensor& omega)
{
	torch::Tensor tau = 1.0/(omega/(2*M_PI));
	torch::Tensor out = torch::zeros({omega.numel(), omega.numel()}, tensorOptCpu<fvalue>());
	auto outAccessor = out.accessor<float, 2>();
	auto omegaAccessor = omega.accessor<float, 1>();
	auto tauAccessor = tau.accessor<float, 1>();
	for(int32_t i = 0; i < out.size(0); ++i)
	{
		for(int32_t j = 0; j < out.size(1); ++j)
		{
			outAccessor[i][j] = 0.5*(omegaAccessor[i]*tauAccessor[j])/(1+std::pow(omegaAccessor[i]*tauAccessor[j], 2));
			if(j == 0)
				outAccessor[i][j] = outAccessor[i][j]*std::log(tauAccessor[j+1]/tauAccessor[j]);
			else if(j == out.size(1)-1)
				outAccessor[i][j] = outAccessor[i][j]*std::log(tauAccessor[j]/tauAccessor[j-1]);
			else
				outAccessor[i][j] = outAccessor[i][j]*std::log(tauAccessor[j+1]/tauAccessor[j-1]);
		}
	}
	return out;
}

static torch::Tensor aReal(torch::Tensor& omega)
{
	torch::Tensor tau = 1.0/(omega/(2*M_PI));
	torch::Tensor out = torch::zeros({omega.numel(), omega.numel()}, torch::TensorOptions().dtype(torch::kFloat32));
	auto outAccessor = out.accessor<float, 2>();
	auto omegaAccessor = omega.accessor<float, 1>();
	auto tauAccessor = tau.accessor<float, 1>();
	for(int32_t i = 0; i < out.size(0); ++i)
	{
		for(int32_t j = 0; j < out.size(1); ++j)
		{
			outAccessor[i][j] = -0.5/(1+std::pow(omegaAccessor[i]*tauAccessor[j], 2));
			if(j == 0)
				outAccessor[i][j] = outAccessor[i][j]*std::log(tauAccessor[j+1]/tauAccessor[j]);
			else if(j == out.size(1)-1)
				outAccessor[i][j] = outAccessor[i][j]*std::log(tauAccessor[j]/tauAccessor[j-1]);
			else
				outAccessor[i][j] = outAccessor[i][j]*std::log(tauAccessor[j+1]/tauAccessor[j-1]);
		}
	}
	return out;
}

class RtFunct
{
private:
	torch::Tensor impedanceSpectra;
	torch::Tensor aMatrixImag;
	torch::Tensor aMatrixReal;
	fvalue el;
	fvalue epsilon;

public:
	RtFunct(torch::Tensor impedanceSpectraI, torch::Tensor aMatrixImagI, torch::Tensor aMatrixRealI, fvalue elI, fvalue epsilonI):
	impedanceSpectra(impedanceSpectraI),
	aMatrixImag(aMatrixImagI),
	aMatrixReal(aMatrixRealI),
	el(elI),
	epsilon(epsilonI)
	{

	}

	fvalue function(const torch::Tensor& x)
	{
		auto xAccessor = x.accessor<fvalue, 1>();
		int64_t size = x.numel();
		torch::Tensor xLeft = x.narrow(0, 0, x.numel()-1);

		torch::Tensor MSE_re = torch::sum(torch::pow(xAccessor[size-1] + torch::matmul(aMatrixReal, xLeft) - torch::real(impedanceSpectra), 2), torch::typeMetaToScalarType(x.dtype()));
		torch::Tensor MSE_im = torch::sum(torch::pow(torch::matmul(aMatrixImag, xLeft) - torch::imag(impedanceSpectra), 2), torch::typeMetaToScalarType(x.dtype()));
		torch::Tensor reg_term = el/2*torch::sum(torch::pow(xLeft, 2), torch::typeMetaToScalarType(x.dtype()));
		torch::Tensor obj = MSE_re + MSE_im + reg_term;
		return obj.item().to<fvalue>();
	}

	static torch::Tensor getGrad(std::function<fvalue(const torch::Tensor& x)> fn, const torch::Tensor& xTensor, fvalue epsilon)
	{
		torch::Tensor out = torch::zeros(xTensor.sizes(), tensorOptCpu<fvalue>(false));
		auto outAccessor = out.accessor<fvalue, 1>();
		assert(checkTorchType<fvalue>(xTensor));
		auto xAccessor = xTensor.accessor<fvalue, 1>();
		for(int64_t i = 0; i < out.size(0); ++i)
		{
			xAccessor[i] -= epsilon;
			fvalue left = fn(xTensor);
			xAccessor[i] += 2*epsilon;
			fvalue right = fn(xTensor);
			xAccessor[i] -= epsilon;
			outAccessor[i] = (right-left)/(2*epsilon);
		}
		return out;
	}

	fvalue operator()(const Eigen::VectorX<fvalue>& x, Eigen::VectorX<fvalue>& grad)
	{
		Eigen::MatrixX<fvalue> xMatrix = x;
		torch::Tensor xTensor = eigen2libtorch(xMatrix);
		xTensor = xTensor.reshape({xTensor.numel()});
		torch::Tensor gradTensor = getGrad(std::bind(&RtFunct::function, this, std::placeholders::_1), xTensor, epsilon);
		grad = libtorch2eigenVector<fvalue>(gradTensor);
		return function(xTensor);
	}
};

static torch::Tensor calcBounds(torch::Tensor& impedanceSpectra, torch::Tensor startTensor)
{
	torch::Tensor lowerBounds = torch::zeros({1, startTensor.numel()}, tensorOptCpu<fvalue>());
	torch::Tensor upperBounds = torch::ones({1, startTensor.numel()}, tensorOptCpu<fvalue>())*torch::max(torch::abs(impedanceSpectra));
	return torch::cat({lowerBounds, upperBounds}, 0);
}

torch::Tensor calcDrt(torch::Tensor& impedanceSpectra, torch::Tensor& omegaTensor, FitMetics& fm, const FitParameters& fp)
{
	torch::Tensor aMatrixImag = aImag(omegaTensor);
	torch::Tensor aMatrixReal = aReal(omegaTensor);

	LBFGSpp::LBFGSBParam<fvalue> fitParam;
	fitParam.epsilon = fp.epsilon;
	fitParam.max_iterations = fp.maxIter;
	fitParam.max_linesearch = fp.maxIter*10;

	LBFGSpp::LBFGSBSolver<fvalue> solver(fitParam);
	RtFunct funct(impedanceSpectra, aMatrixImag, aMatrixReal, 0.01, fp.step);

	torch::Tensor startTensor = guesStartingPoint(omegaTensor, impedanceSpectra);
	torch::Tensor bounds = calcBounds(impedanceSpectra, startTensor);
	torch::Tensor lowerBoundTensor = bounds.select(0, 0);
	torch::Tensor upperBoundTensor = bounds.select(0, 1);
	Eigen::VectorX<fvalue> lowerbound = libtorch2eigenVector<fvalue>(lowerBoundTensor);
	Eigen::VectorX<fvalue> upperbound = libtorch2eigenVector<fvalue>(upperBoundTensor);

	Eigen::VectorX<fvalue> x = libtorch2eigenVector<fvalue>(startTensor);

	fm.iterations = solver.minimize(funct, x, fm.fx, lowerbound, upperbound);

	torch::Tensor xT = eigenVector2libtorch<fvalue>(x);
	return xT;
}

torch::Tensor calcDrt(const std::vector<eis::DataPoint>& data, const std::vector<fvalue>& omegaVector, FitMetics& fm, const FitParameters& fp)
{
	torch::Tensor impedanceSpectra = eisToComplexTensor(data, nullptr);
	torch::Tensor omegaTensor = fvalueVectorToTensor(const_cast<std::vector<fvalue>&>(omegaVector)).clone();
	return calcDrt(impedanceSpectra, omegaTensor, fm, fp);
}

torch::Tensor calcDrt(const std::vector<eis::DataPoint>& data, FitMetics& fm,  const FitParameters& fp)
{
	torch::Tensor omegaTensor;
	torch::Tensor impedanceSpectra = eisToComplexTensor(data, &omegaTensor);
	return calcDrt(impedanceSpectra, omegaTensor, fm, fp);
}


