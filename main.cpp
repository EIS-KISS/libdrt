#include <ATen/core/ATen_fwd.h>
#include <ATen/core/TensorBody.h>
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
	torch::Tensor output = torch::empty({static_cast<long int>(data.size()*2)}, tensorOptCpuNg<fvalue>());
	if(freqs)
		*freqs = torch::empty({static_cast<long int>(data.size()*2)}, tensorOptCpuNg<fvalue>());

	float* tensorDataPtr = output.contiguous().data_ptr<float>();
	float* tensorFreqDataPtr = freqs ? freqs->contiguous().data_ptr<float>() : nullptr;

	for(size_t i = 0; i < data.size()*2; ++i)
	{
		float datapoint = i < data.size() ? data[i].im.real() : data[i - data.size()].im.imag();
		if(std::isnan(datapoint) || std::isinf(datapoint))
			datapoint = 0;
		tensorDataPtr[i] = datapoint;
		if(tensorFreqDataPtr)
			tensorFreqDataPtr[i] = data[i % data.size()].omega;
	}

	output = torch::view_as_complex(output.reshape({static_cast<int64_t>(data.size()), 2}));
	return output;
}

torch::Tensor fvalueVectorToTensor(std::vector<fvalue>& vect)
{
	return torch::from_blob(vect.data(), {static_cast<int64_t>(vect.size())}, tensorOptCpuNg<fvalue>());
}

torch::Tensor guesStartingPoint(torch::Tensor& omega, torch::Tensor& impedanceSpectra)
{
	torch::Tensor startingPoint = torch::zeros(omega.sizes(), tensorOptCpuNg<fvalue>());
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

	static fvalue function(const torch::Tensor& x)
	{
		assert(checkTorchType<fvalue>(x));
		auto xAccessor = x.accessor<fvalue, 1>();
		fvalue accum = 0;
		for(int64_t i = 0; i < x.size(0); ++i)
			accum += (xAccessor[i]+3)*(xAccessor[i]+3)+20;
		return accum;
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
		std::cout<<"xTensor\n "<<xTensor<<'\n';
		torch::Tensor gradTensor = getGrad(&function, xTensor, epsilon);
		grad = libtorch2eigenVector<fvalue>(gradTensor);
		return function(xTensor);
	}
};

int main(int argc, char** argv)
{
	std::cout<<std::scientific;

	eis::Range omega(1, 1e6, 3, true);
	std::vector<fvalue> omegaVector = omega.getRangeVector();
	torch::Tensor omegaTensor = fvalueVectorToTensor(omegaVector);
	std::cout<<"Omega Tensor\n "<<omegaTensor<<'\n';
	eis::Model model("r{10}-r{50}p{0.02, 0.8}");

	std::vector<eis::DataPoint> data = model.executeSweep(omega);
	torch::Tensor impedanceSpectra = eisToTensor(data, nullptr);

	torch::Tensor aMatrixImag = aImag(omegaTensor);
	torch::Tensor aMatrixReal = aReal(omegaTensor);
	std::cout<<"aMatrixImag\n "<<aMatrixImag<<'\n';
	std::cout<<"aMatrixReal\n "<<aMatrixReal<<'\n';

	printImpedance(data);

	LBFGSpp::LBFGSParam<fvalue> fitParam;
	fitParam.epsilon = 1e-6;
	fitParam.max_iterations = 100;

	LBFGSpp::LBFGSSolver<fvalue> solver(fitParam);
	RtFunct funct(impedanceSpectra, aMatrixImag, aMatrixReal, 0.1, 0.001);
	Eigen::VectorX<fvalue> x = Eigen::VectorX<fvalue>::Ones(4)*3;
	fvalue fx;
	int iterations = solver.minimize(funct, x, fx);

	std::cout<<"Iterations: "<<iterations<<'\n';
	std::cout<<"fx "<<fx<<'\n';
	std::cout<<"xVect\n"<<x<<'\n';

	return 0;
}
