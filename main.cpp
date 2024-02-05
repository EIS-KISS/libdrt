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

#include <iostream>
#include <eisgenerator/model.h>
#include <eisgenerator/eistype.h>

#include "eisdrt/eisdrt.h"

static void printImpedance(const std::vector<eis::DataPoint>& data)
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

static void printFvalueVector(const std::vector<fvalue>& data)
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
	eis::Range omega(1, 1e6, 10, true);

	// specify circut to be simulated
	eis::Model model("r{10}-r{50}p{0.02, 0.8}");

	// execute a simulation
	std::vector<eis::DataPoint> data = model.executeSweep(omega);

	// print the specrum
	printImpedance(data);

	// allocate a FitMetrics struct on the stack
	FitMetrics fm;

	// calculate the drt for this spectrum
	fvalue rSeries;
	std::vector<fvalue> x = calcDrt(data, fm, FitParameters(1000), &rSeries);

	assert(x.size() == data.size());

	std::vector<eis::DataPoint> recalculatedSpectra = calcImpedance(x, rSeries, omega);
	fvalue dist = eisNyquistDistance(data, recalculatedSpectra);

	// print some info on the drt
	std::cout<<"Iterations: "<<fm.iterations<<'\n';
	std::cout<<"fx "<<fm.fx<<"\ndrt: ";
	printFvalueVector(x);
	std::cout<<"r series: "<<rSeries<<'\n';
	std::cout<<"dist: "<<dist<<'\n';
	std::cout<<"recalculatedSpectra:\n";
	printImpedance(recalculatedSpectra);

	return dist < 2 ? 0 : 1;
}
