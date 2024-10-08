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
#include <kisstype/type.h>
#include <kisstype/spectra.h>

#include "eisdrt/eisdrt.h"

static void print_drt(const std::vector<fvalue>& data)
{
	std::cout<<'[';
	size_t colcount = 0;
	for(const fvalue point : data)
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

static eis::Spectra transform_to_drt_spectra(const std::vector<fvalue>& drt, const eis::Spectra& spectra)
{
	assert(spectra.data.size() == drt.size());
	eis::Spectra drtSpectra = spectra;
	drtSpectra.data.clear();
	drtSpectra.data.reserve(drt.size());

	for(size_t i = 0; i < drt.size(); ++i)
		drtSpectra.data.push_back(eis::DataPoint(std::complex<fvalue>(drt[i], 0), spectra.data[i].omega));

	return drtSpectra;
}

static void print_help(char* name)
{
	std::cout<<"Usage "<<name<<" [INPUT_SPECTRA_FILENAME] [OUTPUT_FILE_FILENAME]\n";
	std::cout<<"The input spectra is expected to be in eisgenerator format\n";
}

int main(int argc, char** argv)
{
	if(argc != 3 && argc > 0)
		print_help(argv[0]);
	if(argc != 3)
		return 1;

	bool toStdout = std::string(argv[2]) == "-";

	try
	{
		eis::Spectra spectra;
		if(std::string(argv[1]) != "-")
		{
			if(!toStdout)
				std::cout<<"Loading spectra\n";
			spectra = eis::Spectra::loadFromDisk(argv[1]);
		}
		else
		{
			if(!toStdout)
				std::cout<<"Waiting for spectra on stdin\n";
			spectra = eis::Spectra::loadFromStream(std::cin);
		}

		if(!toStdout)
			std::cout<<"Calculateing Drt\n";
		FitMetrics fm;
		std::vector<fvalue> drt = calcDrt(spectra.data, fm, FitParameters(1000));

		if(!toStdout)
		{
			std::cout<<"Calculated Drt:\n";
			print_drt(drt);
		}

		eis::Spectra drtSpectra = transform_to_drt_spectra(drt, spectra);
		if(!toStdout)
			drtSpectra.saveToDisk(argv[2]);
		else
			drtSpectra.saveToStream(std::cout);
	}
	catch(const eis::file_error& err)
	{
		std::cerr<<"Could not read spectra from "<<argv[1]<<' '<<err.what()<<'\n';
		return 1;
	}

	return 0;
}
