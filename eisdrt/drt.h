#pragma once
#include <eisgenerator/eistype.h>
#include <torch/torch.h>

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

torch::Tensor calcDrt(torch::Tensor& impedanceSpectra, torch::Tensor& omegaTensor, FitMetics& fm, const FitParameters& fp);

torch::Tensor calcDrt(const std::vector<eis::DataPoint>& data, const std::vector<fvalue>& omegaVector, FitMetics& fm,  const FitParameters& fp);

torch::Tensor calcDrt(const std::vector<eis::DataPoint>& data, FitMetics& fm,  const FitParameters& fp);
