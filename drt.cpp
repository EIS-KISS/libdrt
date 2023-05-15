#include "eisdrt/drt.h"

#include <Eigen/Core>
#include <Eigen/StdVector>
#include <eisgenerator/eistype.h>
#include <iostream>

#include "Eigen/src/Core/Matrix.h"
#include "eistoeigen.h"
#include "LBFG/LBFGSB.h"

static Eigen::Vector<fvalue, Eigen::Dynamic> guesStartingPoint(Eigen::Vector<fvalue, Eigen::Dynamic>& omega, Eigen::Vector<std::complex<fvalue>, Eigen::Dynamic>& impedanceSpectra)
{
	Eigen::Vector<fvalue, Eigen::Dynamic> startingPoint = Eigen::Vector<fvalue, Eigen::Dynamic>::Zero(omega.size()+1);
	startingPoint[startingPoint.size()-1] = std::abs(impedanceSpectra[impedanceSpectra.size()-1]);
	return startingPoint;
}

static Eigen::Matrix<fvalue, Eigen::Dynamic, Eigen::Dynamic> aImag(Eigen::Vector<fvalue, Eigen::Dynamic>& omega)
{
	Eigen::Vector<fvalue, Eigen::Dynamic> tau = (omega * 1/static_cast<fvalue>(2*M_PI)).cwiseInverse();
	Eigen::Matrix<fvalue, Eigen::Dynamic, Eigen::Dynamic> out =
		Eigen::Matrix<fvalue, Eigen::Dynamic, Eigen::Dynamic>::Zero(omega.size(), omega.size());

	for(int32_t i = 0; i < out.cols(); ++i)
	{
		for(int32_t j = 0; j < out.rows(); ++j)
		{
			out(i,j) = 0.5*(omega[i]*tau[j])/(1+std::pow(omega[i]*tau[j], 2));
			if(j == 0)
				out(i,j) = out(i,j)*std::log(tau[j+1]/tau[j]);
			else if(j == out.rows()-1)
				out(i,j) = out(i,j)*std::log(tau[j]/tau[j-1]);
			else
				out(i,j) = out(i,j)*std::log(tau[j+1]/tau[j-1]);
		}
	}
	return out;
}

static Eigen::Matrix<fvalue, Eigen::Dynamic, Eigen::Dynamic> aReal(Eigen::Vector<fvalue, Eigen::Dynamic>& omega)
{
	Eigen::Vector<fvalue, Eigen::Dynamic> tau = (omega * 1/static_cast<fvalue>(2*M_PI)).cwiseInverse();
	Eigen::Matrix<fvalue, Eigen::Dynamic, Eigen::Dynamic> out =
		Eigen::Matrix<fvalue, Eigen::Dynamic, Eigen::Dynamic>::Zero(omega.size(), omega.size());

	for(int32_t i = 0; i < out.cols(); ++i)
	{
		for(int32_t j = 0; j < out.rows(); ++j)
		{
			out(i, j) = -0.5/(1+std::pow(omega[i]*tau[j], 2));
			if(j == 0)
				out(i, j) = out(i, j)*std::log(tau[j+1]/tau[j]);
			else if(j == out.rows()-1)
				out(i, j) = out(i, j)*std::log(tau[j]/tau[j-1]);
			else
				out(i, j) = out(i, j)*std::log(tau[j+1]/tau[j-1]);
		}
	}
	return out;
}

class RtFunct
{
private:
	Eigen::Vector<std::complex<fvalue>, Eigen::Dynamic> impedanceSpectra;
	Eigen::Matrix<fvalue, Eigen::Dynamic, Eigen::Dynamic> aMatrixImag;
	Eigen::Matrix<fvalue, Eigen::Dynamic, Eigen::Dynamic> aMatrixReal;
	fvalue el;
	fvalue epsilon;

public:
	RtFunct(Eigen::Vector<std::complex<fvalue>, Eigen::Dynamic> impedanceSpectraI,
		Eigen::Matrix<fvalue, Eigen::Dynamic, Eigen::Dynamic> aMatrixImagI,
		Eigen::Matrix<fvalue, Eigen::Dynamic, Eigen::Dynamic> aMatrixRealI,
		fvalue elI, fvalue epsilonI):
	impedanceSpectra(impedanceSpectraI),
	aMatrixImag(aMatrixImagI),
	aMatrixReal(aMatrixRealI),
	el(elI),
	epsilon(epsilonI)
	{

	}

	fvalue function(const Eigen::Vector<fvalue, Eigen::Dynamic>& x)
	{
		int64_t size = x.size();
		Eigen::Vector<fvalue, Eigen::Dynamic> xLeft = x.head(x.size()-1);

		std::cout<<"aMatrixReal:\n"<<aMatrixReal<<"\nxLeft:\n"<<xLeft<<"\nx:\n"<<x<<std::endl;
		Eigen::Vector<fvalue, Eigen::Dynamic> t = aMatrixReal*xLeft;
		std::cout<<"T1:\n"<<t<<std::endl;
		t = t - impedanceSpectra.real();
		std::cout<<"T2:\n"<<t<<std::endl;
		t = t.array() + x[size-1];
		std::cout<<"T3:\n"<<t<<std::endl;
		t = t.array().pow(2);
		std::cout<<"T4:\n"<<t<<std::endl;
		fvalue MSE_re = t.sum();
		std::cout<<"T5:\n"<<MSE_re<<std::endl;

		t = (aMatrixImag*xLeft - impedanceSpectra.imag()).array().pow(2);
		fvalue MSE_im = t.sum();
		fvalue reg_term = el/2*xLeft.array().pow(2).sum();
		fvalue obj = MSE_re + MSE_im + reg_term;
		return obj;
	}

	static Eigen::Vector<fvalue, Eigen::Dynamic> getGrad(std::function<fvalue(const Eigen::Vector<fvalue, Eigen::Dynamic>& x)> fn,
													Eigen::Vector<fvalue, Eigen::Dynamic>& x, fvalue epsilon)
	{
		Eigen::Vector<fvalue, Eigen::Dynamic> out = Eigen::Vector<fvalue, Eigen::Dynamic>::Zero(x.size());
		for(int64_t i = 0; i < out.size(); ++i)
		{
			x[i] -= epsilon;
			fvalue left = fn(x);
			x[i] += 2*epsilon;
			fvalue right = fn(x);
			x[i] -= epsilon;
			out[i] = (right-left)/(2*epsilon);
		}
		return out;
	}

	fvalue operator()(Eigen::VectorX<fvalue>& x, Eigen::VectorX<fvalue>& grad)
	{
		grad = getGrad(std::bind(&RtFunct::function, this, std::placeholders::_1), x, epsilon);
		std::cout<<"grad:\n"<<grad<<std::endl;
		return function(x);
	}
};

static Eigen::Matrix<fvalue, Eigen::Dynamic, 2> calcBounds(Eigen::VectorX<std::complex<fvalue>>& impedanceSpectra, Eigen::VectorX<fvalue> startTensor)
{
	Eigen::VectorX<fvalue> lowerBounds = Eigen::VectorX<fvalue>::Zero(startTensor.size());
	Eigen::VectorX<fvalue> upperBounds = Eigen::VectorX<fvalue>::Ones(startTensor.size())*impedanceSpectra.cwiseAbs().maxCoeff();

	Eigen::Matrix<fvalue, Eigen::Dynamic, 2> out(lowerBounds.size(), 2);
	out.col(0) = lowerBounds;
	out.col(1) = upperBounds;
	return out;
}

Eigen::VectorX<fvalue> calcDrt(Eigen::VectorX<std::complex<fvalue>>& impedanceSpectra, Eigen::VectorX<fvalue>& omegaTensor, FitMetics& fm, const FitParameters& fp)
{
	Eigen::Matrix<fvalue, Eigen::Dynamic, Eigen::Dynamic> aMatrixImag = aImag(omegaTensor);
	Eigen::Matrix<fvalue, Eigen::Dynamic, Eigen::Dynamic> aMatrixReal = aReal(omegaTensor);

	LBFGSpp::LBFGSBParam<fvalue> fitParam;
	fitParam.epsilon = fp.epsilon;
	fitParam.max_iterations = fp.maxIter;
	fitParam.max_linesearch = fp.maxIter*10;

	LBFGSpp::LBFGSBSolver<fvalue> solver(fitParam);
	RtFunct funct(impedanceSpectra, aMatrixImag, aMatrixReal, 0.01, fp.step);

	Eigen::VectorX<fvalue> x = guesStartingPoint(omegaTensor, impedanceSpectra);
	std::cout<<"StartingPoint\n"<<x<<std::endl;
	Eigen::Matrix<fvalue, Eigen::Dynamic, 2> bounds = calcBounds(impedanceSpectra, x);
	Eigen::VectorX<fvalue> lowerBounds = bounds.col(0);
	Eigen::VectorX<fvalue> upperBounds = bounds.col(1);

	fm.iterations = solver.minimize(funct, x, fm.fx, lowerBounds, upperBounds);

	return x;
}

Eigen::VectorX<fvalue> calcDrt(const std::vector<eis::DataPoint>& data, const std::vector<fvalue>& omegaVector, FitMetics& fm, const FitParameters& fp)
{
	Eigen::VectorX<std::complex<fvalue>> impedanceSpectra = eistoeigen(data);
	Eigen::VectorX<fvalue> omega = Eigen::VectorX<fvalue>::Map(omegaVector.data(), omegaVector.size());
	return calcDrt(impedanceSpectra, omega, fm, fp);
}

Eigen::VectorX<fvalue> calcDrt(const std::vector<eis::DataPoint>& data, FitMetics& fm,  const FitParameters& fp)
{
	Eigen::VectorX<fvalue> omega;
	Eigen::VectorX<std::complex<fvalue>> impedanceSpectra = eistoeigen(data, &omega);
	return calcDrt(impedanceSpectra, omega, fm, fp);
}


