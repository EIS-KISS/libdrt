#include <ATen/core/ATen_fwd.h>
#include <ATen/core/TensorBody.h>
#include <ATen/ops/imag.h>
#include <c10/core/ScalarType.h>
#include <cstdint>
#include <iostream>
#include <eisgenerator/model.h>
#include <eisgenerator/eistype.h>
#include <math.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/torch.h>
#include <cmath>
#include <functional>

#include <Eigen/Dense>
#include "Eigen/src/Core/Matrix.h"
#include "LBFGS.h"
#include "eigentorchconversions.h"

void printImpedance(const std::vector<eis::DataPoint>& data)
{
	std::cout<<'[';
	size_t colcount = 0;
	for(const eis::DataPoint& point : data)
	{
		std::cout<<point.im;
		std::cout<<' ';
		if(++colcount > 1)
		{
			std::cout<<'\n';
			colcount = 0;
		}
	}
	std::cout<<"]\n";
}

torch::Tensor eisToTensor(const std::vector<eis::DataPoint>& data, torch::Tensor* freqs)
{
	torch::TensorOptions options = tensorOptCpuNg<fvalue>();

	if constexpr(std::is_same<fvalue, float>::value)
		options = options.dtype(torch::kComplexFloat);
	else
		options = options.dtype(torch::kComplexDouble);
	torch::Tensor output = torch::empty({static_cast<long int>(data.size())}, options);
	if(freqs)
		*freqs = torch::empty({static_cast<long int>(data.size())}, tensorOptCpuNg<fvalue>());

	torch::Tensor real = torch::real(output);
	torch::Tensor imag = torch::imag(output);

	auto realAccessor = real.accessor<fvalue, 1>();
	auto imagAccessor = imag.accessor<fvalue, 1>();
	float* tensorFreqDataPtr = freqs ? freqs->contiguous().data_ptr<float>() : nullptr;

	for(size_t i = 0; i < data.size(); ++i)
	{
		fvalue real = data[i].im.real();
		fvalue imag = data[i].im.imag();
		if(std::isnan(real) || std::isinf(real))
			real = 0;
		if(std::isnan(imag) || std::isinf(imag))
			real = 0;

		realAccessor[i] = real;
		imagAccessor[i] = imag;
		if(tensorFreqDataPtr)
			tensorFreqDataPtr[i] = data[i % data.size()].omega;
	}

	return output;
}

torch::Tensor fvalueVectorToTensor(std::vector<fvalue>& vect)
{
	return torch::from_blob(vect.data(), {static_cast<int64_t>(vect.size())}, tensorOptCpuNg<fvalue>());
}

torch::Tensor guesStartingPoint(torch::Tensor& omega, torch::Tensor& impedanceSpectra)
{
	std::vector<int64_t> size = omega.sizes().vec();
	++size[0];
	torch::Tensor startingPoint = torch::zeros(size, tensorOptCpuNg<fvalue>());
	startingPoint[-1] = torch::abs(impedanceSpectra[-1]);
	return startingPoint;
}

torch::Tensor aImag(torch::Tensor& omega)
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

torch::Tensor aReal(torch::Tensor& omega)
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

/*def S(gamma_R_inf, Z_exp_re, Z_exp_im, A_re, A_im, el):
    MSE_re = np.sum((gamma_R_inf[-1] + np.matmul(A_re, gamma_R_inf[:-1]) - Z_exp_re)**2)
    MSE_im = np.sum((np.matmul(A_im, gamma_R_inf[:-1]) - Z_exp_im)**2)
    reg_term = el/2*np.sum(gamma_R_inf[:-1]**2)
    obj = MSE_re + MSE_im + reg_term
    return obj

torch::Tensor tikhnovDrt(torch::Tensor& omega, torch::Tensor& impedanceSpectra, fvalue regularaziaion = 1e-2)
{
	torch::Tensor aMatrixImag = aImag(omega);
	torch::Tensor aMatrixReal = aReal(omega);
	torch::Tensor startingPoint = guesStartingPoint(omega, impedanceSpectra);

	torch::Tensor bounds = torch::zeros({startingPoint.size(0), 1}, tensorOptCpuNg<fvalue>());
	bounds = torch::cat({bounds, torch::zeros({startingPoint.size(0), 1}, tensorOptCpuNg<fvalue>())*torch::max(torch::abs(impedanceSpectra))});

	std::cout<<"startingPoint:\n "<<startingPoint<<'\n';
	std::cout<<"bounds:\n "<<bounds<<'\n';

	result = minimize(S, x0, args=(Z_exp_re, Z_exp_im, A_re, A_im, el), method=method,
					bounds = bounds, options={'disp': True, 'ftol':1e-10, 'maxiter':200})
	gamma_R_inf = result.x
	R_inf = gamma_R_inf[-1]
	gamma = gamma_R_inf[:-1]
	return gamma, R_inf
	return bounds;
}*/

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
		assert(checkTorchType<fvalue>(x));
		assert(x.sizes().size() == 1);
		auto xAccessor = x.accessor<fvalue, 1>();
		int64_t size = x.numel();
		torch::Tensor xLeft = x.narrow(0, 0, x.numel()-1);

		std::cout<<"x:\n"<<x<<'\n';
		std::cout<<"xLeft:\n"<<xLeft<<'\n';
		std::cout<<"real:\n"<<torch::real(impedanceSpectra)<<'\n';
		std::cout<<"imag:\n"<<torch::imag(impedanceSpectra)<<'\n';
		std::cout<<"aMatrixReal:\n"<<aMatrixReal<<'\n';
		std::cout<<"aMatrixImag:\n"<<aMatrixImag<<'\n';


		torch::Tensor MSE_re = torch::sum(torch::pow(xAccessor[size-1] + torch::matmul(aMatrixReal, xLeft) - torch::real(impedanceSpectra), 2));
		torch::Tensor MSE_im = torch::sum(torch::pow(torch::matmul(aMatrixImag, xLeft) - torch::imag(impedanceSpectra), 2));
		torch::Tensor reg_term = el/2*torch::sum(torch::pow(xLeft, 2));
		torch::Tensor obj = MSE_re + MSE_im + reg_term;
		return obj.item().to<fvalue>();
	}

	static torch::Tensor getGrad(std::function<fvalue(const torch::Tensor& x)> fn, const torch::Tensor& xTensor, fvalue epsilon)
	{
		torch::Tensor out = torch::zeros(xTensor.sizes(), tensorOptCpuNg<fvalue>());
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
		//std::cout<<"xTensor\n "<<xTensor<<'\n';
		torch::Tensor gradTensor = getGrad(std::bind(&RtFunct::function, this, std::placeholders::_1), xTensor, epsilon);
		grad = libtorch2eigenVector<fvalue>(gradTensor);
		return function(xTensor);
	}
};

static void testFunc(RtFunct& funct, torch::Tensor& omega)
{
	std::cout<<__func__<<'\n';
	std::vector<int64_t> size = omega.sizes().vec();
	++size[0];
	torch::Tensor x = torch::zeros(size, tensorOptCpuNg<fvalue>());
	x[-1] = 3;
	x[0] = 0.5;
	std::cout<<"RtFunct.function: "<<funct.function(x)<<std::endl;
}

int main(int argc, char** argv)
{
	std::cout<<std::scientific;

	eis::Range omega(1, 1e6, 3, true);
	std::vector<fvalue> omegaVector = omega.getRangeVector();
	torch::Tensor omegaTensor = fvalueVectorToTensor(omegaVector);
	eis::Model model("r{10}-r{50}p{0.02, 0.8}");

	std::vector<eis::DataPoint> data = model.executeSweep(omega);
	torch::Tensor impedanceSpectra = eisToTensor(data, nullptr);

	torch::Tensor aMatrixImag = aImag(omegaTensor);
	torch::Tensor aMatrixReal = aReal(omegaTensor);

	printImpedance(data);

	LBFGSpp::LBFGSParam<fvalue> fitParam;
	fitParam.epsilon = 1e-2;
	fitParam.max_iterations = 100;
	fitParam.max_linesearch = 1000;

	LBFGSpp::LBFGSSolver<fvalue> solver(fitParam);
	RtFunct funct(impedanceSpectra, aMatrixImag, aMatrixReal, 0.01, 0.001);

	torch::Tensor startTensor = guesStartingPoint(omegaTensor, impedanceSpectra);
	Eigen::VectorX<fvalue> x = libtorch2eigenVector<fvalue>(startTensor);
	fvalue fx;
	int iterations = solver.minimize(funct, x, fx);

	std::cout<<"Iterations: "<<iterations<<'\n';
	std::cout<<"fx "<<fx<<'\n';
	std::cout<<"xVect\n"<<x<<'\n';

	return 0;
}
