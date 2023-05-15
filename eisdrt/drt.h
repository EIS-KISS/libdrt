#pragma once
#include <eisgenerator/eistype.h>
#include <Eigen/Core>

struct FitMetics
{
	int iterations;
	fvalue fx;
	bool compleated;
};

struct FitParameters
{
	int maxIter;
	double epsilon;
	double step;
	FitParameters(int maxIterI, double epsilonI = 1e-2, double stepI = 0.001): maxIter(maxIterI), epsilon(epsilonI), step(stepI){}
};

Eigen::VectorX<fvalue> calcDrt(Eigen::VectorX<std::complex<fvalue>>& impedanceSpectra, Eigen::VectorX<fvalue>& omegaTensor, FitMetics& fm, const FitParameters& fp);

Eigen::VectorX<fvalue> calcDrt(const std::vector<eis::DataPoint>& data, const std::vector<fvalue>& omegaVector, FitMetics& fm, const FitParameters& fp);

Eigen::VectorX<fvalue> calcDrt(const std::vector<eis::DataPoint>& data, FitMetics& fm, const FitParameters& fp);
