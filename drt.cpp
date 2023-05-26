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

#include "eisdrt/eigendrt.h"
#include "eisdrt/types.h"
#include <stdexcept>

#ifdef USE_EISGEN
#include "eisdrt/eisdrt.h"
#include "eistoeigen.h"
#endif

#include <Eigen/Core>
#include <Eigen/StdVector>
#include <eisgenerator/eistype.h>
#include <iostream>

#include "Eigen/src/Core/Matrix.h"
#include "LBFG/LBFGSB.h"

template<typename fv>
static Eigen::Vector<fv, Eigen::Dynamic>
guesStartingPoint(Eigen::Vector<fv, Eigen::Dynamic>& omega, Eigen::Vector<std::complex<fv>, Eigen::Dynamic>& impedanceSpectra)
{
	Eigen::Vector<fv, Eigen::Dynamic> startingPoint = Eigen::Vector<fv, Eigen::Dynamic>::Zero(omega.size()+1);
	startingPoint[startingPoint.size()-1] = std::abs(impedanceSpectra[impedanceSpectra.size()-1]);
	return startingPoint;
}

template<typename fv>
static Eigen::Matrix<fv, Eigen::Dynamic, Eigen::Dynamic> aImag(Eigen::Vector<fv, Eigen::Dynamic>& omega)
{
	Eigen::Vector<fv, Eigen::Dynamic> tau = (omega * 1/static_cast<fv>(2*M_PI)).cwiseInverse();
	Eigen::Matrix<fv, Eigen::Dynamic, Eigen::Dynamic> out =
		Eigen::Matrix<fv, Eigen::Dynamic, Eigen::Dynamic>::Zero(omega.size(), omega.size());

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

template<typename fv>
static Eigen::Matrix<fv, Eigen::Dynamic, Eigen::Dynamic> aReal(Eigen::Vector<fv, Eigen::Dynamic>& omega)
{
	Eigen::Vector<fv, Eigen::Dynamic> tau = (omega * 1/static_cast<fv>(2*M_PI)).cwiseInverse();
	Eigen::Matrix<fv, Eigen::Dynamic, Eigen::Dynamic> out =
		Eigen::Matrix<fv, Eigen::Dynamic, Eigen::Dynamic>::Zero(omega.size(), omega.size());

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

template<typename fv>
class RtFunct
{
private:
	Eigen::Vector<std::complex<fv>, Eigen::Dynamic> impedanceSpectra;
	Eigen::Matrix<fv, Eigen::Dynamic, Eigen::Dynamic> aMatrixImag;
	Eigen::Matrix<fv, Eigen::Dynamic, Eigen::Dynamic> aMatrixReal;
	fv el;
	fv epsilon;

public:
	RtFunct(Eigen::Vector<std::complex<fv>, Eigen::Dynamic> impedanceSpectraI,
		Eigen::Matrix<fv, Eigen::Dynamic, Eigen::Dynamic> aMatrixImagI,
		Eigen::Matrix<fv, Eigen::Dynamic, Eigen::Dynamic> aMatrixRealI,
		fv elI, fv epsilonI):
	impedanceSpectra(impedanceSpectraI),
	aMatrixImag(aMatrixImagI),
	aMatrixReal(aMatrixRealI),
	el(elI),
	epsilon(epsilonI)
	{

	}

	fv function(const Eigen::Vector<fv, Eigen::Dynamic>& x)
	{
		int64_t size = x.size();
		Eigen::Vector<fv, Eigen::Dynamic> xLeft = x.head(x.size()-1);

		Eigen::Vector<fv, Eigen::Dynamic> t = aMatrixReal*xLeft;
		t = t - impedanceSpectra.real();
		t = t.array() + x[size-1];
		t = t.array().pow(2);
		fv MSE_re = t.sum();

		t = (aMatrixImag*xLeft - impedanceSpectra.imag()).array().pow(2);
		fv MSE_im = t.sum();
		fv reg_term = el/2*xLeft.array().pow(2).sum();
		fv obj = MSE_re + MSE_im + reg_term;
		return obj;
	}

	static Eigen::Vector<fv, Eigen::Dynamic> getGrad(std::function<fv(const Eigen::Vector<fv, Eigen::Dynamic>& x)> fn,
													Eigen::Vector<fv, Eigen::Dynamic>& x, fv epsilon)
	{
		Eigen::Vector<fv, Eigen::Dynamic> out = Eigen::Vector<fv, Eigen::Dynamic>::Zero(x.size());
		for(int64_t i = 0; i < out.size(); ++i)
		{
			x[i] -= epsilon;
			fv left = fn(x);
			x[i] += 2*epsilon;
			fv right = fn(x);
			x[i] -= epsilon;
			out[i] = (right-left)/(2*epsilon);
		}
		return out;
	}

	fv operator()(Eigen::VectorX<fv>& x, Eigen::VectorX<fv>& grad)
	{
		grad = getGrad(std::bind(&RtFunct::function, this, std::placeholders::_1), x, epsilon);
		return function(x);
	}
};

template<typename fv>
static Eigen::Matrix<fv, Eigen::Dynamic, 2> calcBounds(Eigen::VectorX<std::complex<fv>>& impedanceSpectra, Eigen::VectorX<fv> startTensor)
{
	Eigen::VectorX<fv> lowerBounds = Eigen::VectorX<fv>::Zero(startTensor.size());
	Eigen::VectorX<fv> upperBounds = Eigen::VectorX<fv>::Ones(startTensor.size())*impedanceSpectra.cwiseAbs().maxCoeff();

	Eigen::Matrix<fv, Eigen::Dynamic, 2> out(lowerBounds.size(), 2);
	out.col(0) = lowerBounds;
	out.col(1) = upperBounds;
	return out;
}

template<typename fv>
Eigen::VectorX<fv> calcDrt(Eigen::VectorX<std::complex<fv>>& impedanceSpectra, Eigen::VectorX<fv>& omegaTensor, FitMetics& fm, const FitParameters& fp)
{
	Eigen::Matrix<fv, Eigen::Dynamic, Eigen::Dynamic> aMatrixImag = aImag<fv>(omegaTensor);
	Eigen::Matrix<fv, Eigen::Dynamic, Eigen::Dynamic> aMatrixReal = aReal<fv>(omegaTensor);

	LBFGSpp::LBFGSBParam<fv> fitParam;
	fitParam.epsilon = fp.epsilon;
	fitParam.max_iterations = fp.maxIter;
	fitParam.max_linesearch = fp.maxIter*10;

	LBFGSpp::LBFGSBSolver<fv> solver(fitParam);
	RtFunct<fv> funct(impedanceSpectra, aMatrixImag, aMatrixReal, 0.01, fp.step);

	Eigen::VectorX<fv> x = guesStartingPoint(omegaTensor, impedanceSpectra);
	Eigen::Matrix<fv, Eigen::Dynamic, 2> bounds = calcBounds(impedanceSpectra, x);
	Eigen::VectorX<fv> lowerBounds = bounds.col(0);
	Eigen::VectorX<fv> upperBounds = bounds.col(1);

	fv fx;
	try
	{
		fm.iterations = solver.minimize(funct, x, fx, lowerBounds, upperBounds);
		fm.fx = fx;
	}
	catch(const std::invalid_argument& ex)
	{
		throw drt_errror(std::string(ex.what()));
	}
	catch(const std::runtime_error& ex)
	{
		throw drt_errror(std::string(ex.what()));
	}

	return x;
}

template Eigen::VectorX<double> calcDrt<double>(Eigen::VectorX<std::complex<double>>&,
	Eigen::VectorX<double>&, FitMetics& fm, const FitParameters& fp);

template Eigen::VectorX<float> calcDrt<float>(Eigen::VectorX<std::complex<float>>&,
	Eigen::VectorX<float>&, FitMetics& fm, const FitParameters& fp);

template<typename fv>
fv testFn()
{
	fv value = 0.001;
	return value;
}

template double testFn<double>();

#ifdef USE_EISGEN
std::vector<fvalue> calcDrt(const std::vector<eis::DataPoint>& data, const std::vector<fvalue>& omegaVector, FitMetics& fm, const FitParameters& fp)
{
	Eigen::VectorX<std::complex<fvalue>> impedanceSpectra = eistoeigen(data);
	Eigen::VectorX<fvalue> omega = Eigen::VectorX<fvalue>::Map(omegaVector.data(), omegaVector.size());

	Eigen::VectorX<fvalue> drt = calcDrt<fvalue>(impedanceSpectra, omega, fm, fp);
	std::vector<fvalue> stdvector(drt.data(), drt.data()+drt.size());
	return stdvector;
}

std::vector<fvalue> calcDrt(const std::vector<eis::DataPoint>& data, FitMetics& fm,  const FitParameters& fp)
{
	Eigen::VectorX<fvalue> omega;
	Eigen::VectorX<std::complex<fvalue>> impedanceSpectra = eistoeigen(data, &omega);
	Eigen::VectorX<fvalue> drt = calcDrt<fvalue>(impedanceSpectra, omega, fm, fp);
	std::vector<fvalue> stdvector(drt.data(), drt.data()+drt.size());
	return stdvector;
}
#endif

