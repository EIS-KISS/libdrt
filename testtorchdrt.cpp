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

#include "eistotorch.h"
#include "eisdrt/torchdrt.h"
#include "eisdrt/eistorchdrt.h"

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

	FitMetrics fm = {};
	torch::Tensor drt = calcDrtTorch(data, fm, FitParameters(1000));

	std::cout<<"Iterations: "<<fm.iterations<<'\n';
	std::cout<<"fx "<<fm.fx<<'\n';
	std::cout<<"x "<<drt<<'\n';
	return 0;
}
