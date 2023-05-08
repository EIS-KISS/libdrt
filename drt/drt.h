#pragma once
#include <eisgenerator/eistype.h>
#include <torch/torch.h>

struct FitMetics
{
	int iterations;
	fvalue fx;
};

torch::Tensor calcDrt(torch::Tensor& impedanceSpectra, torch::Tensor& omegaTensor, FitMetics& fm);

torch::Tensor calcDrt(const std::vector<eis::DataPoint>& data, const std::vector<fvalue>& omegaVector, FitMetics& fm);
