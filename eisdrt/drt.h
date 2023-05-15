#pragma once
#include <eisgenerator/eistype.h>
#include <Eigen/Core>

#include "types.h"

Eigen::VectorX<fvalue> calcDrt(Eigen::VectorX<std::complex<fvalue>>& impedanceSpectra, Eigen::VectorX<fvalue>& omegaTensor, FitMetics& fm, const FitParameters& fp);

Eigen::VectorX<fvalue> calcDrt(const std::vector<eis::DataPoint>& data, const std::vector<fvalue>& omegaVector, FitMetics& fm, const FitParameters& fp);

Eigen::VectorX<fvalue> calcDrt(const std::vector<eis::DataPoint>& data, FitMetics& fm, const FitParameters& fp);
