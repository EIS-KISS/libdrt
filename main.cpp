#include <iostream>
#include <eisgenerator/model.h>
#include <eisgenerator/eistype.h>

#include "eisdrt/drt.h"
#include "eistoeigen.h"

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

int main(int argc, char** argv)
{
	std::cout<<std::scientific;

	eis::Range omega(1, 1e6, 3, true);
	eis::Model model("r{10}-r{50}p{0.02, 0.8}");

	std::vector<eis::DataPoint> data = model.executeSweep(omega);
	printImpedance(data);

	FitMetics fm = {};
	Eigen::VectorX<fvalue> omegaVector;
	Eigen::VectorX<std::complex<fvalue>> impedanceSpectra = eistoeigen(data, &omegaVector);
	Eigen::VectorX<fvalue> x = calcDrt(impedanceSpectra, omegaVector, fm, FitParameters(1000));

	std::cout<<"Iterations: "<<fm.iterations<<'\n';
	std::cout<<"fx "<<fm.fx<<'\n';
	std::cout<<"xVect\n"<<x<<'\n';

	return 0;
}
