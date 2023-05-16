#include <iostream>
#include <eisgenerator/model.h>
#include <eisgenerator/eistype.h>

#include "eisdrt/eisdrt.h"

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

void printFvalueVector(const std::vector<fvalue>& data)
{
	std::cout<<'[';
	size_t colcount = 0;
	for(fvalue point : data)
	{
		std::cout<<point;
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

	// specify the angular frequency range of the spectra to be
	// simulated to be 1-1*10^6 Hz with 3 steps and log10 distrobution
	eis::Range omega(1, 1e6, 3, true);

	// specify circut to be simulated
	eis::Model model("r{10}-r{50}p{0.02, 0.8}");

	// execute a simulation
	std::vector<eis::DataPoint> data = model.executeSweep(omega);

	// print the specrum
	printImpedance(data);

	// allocate a FitMetics struct on the stack
	FitMetics fm;

	// calculate the drt for this spectrum
	std::vector<fvalue> x = calcDrt(data, fm, FitParameters(1000));

	// print some info on the drt
	std::cout<<"Iterations: "<<fm.iterations<<'\n';
	std::cout<<"fx "<<fm.fx<<"\ndrt: ";
	printFvalueVector(x);

	return 0;
}
