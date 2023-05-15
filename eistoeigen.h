#pragma once

#include <Eigen/Core>
#include <eisgenerator/eistype.h>
#include <vector>

Eigen::VectorX<std::complex<fvalue>> eistoeigen(const std::vector<eis::DataPoint>& data, Eigen::Vector<fvalue, Eigen::Dynamic>* omega = nullptr)
{
	Eigen::VectorX<std::complex<fvalue>> out(data.size());

	if(omega)
		*omega = Eigen::VectorX<fvalue>(data.size());

	for(size_t i = 0; i < data.size(); ++i)
	{
		out[i] = data[i].im;
		if(omega)
			(*omega)[i] = data[i].omega;
	}
	return out;
}
