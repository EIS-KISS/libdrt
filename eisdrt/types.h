#pragma once
#include <eisgenerator/eistype.h>

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
