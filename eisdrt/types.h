#pragma once

/**
Types for use with all eisdrt apis
* @defgroup TYPES types
* Types for use with all eisdrt apis
* @{
*/

/**
 * @brief Returned information on a fit
 */
struct FitMetics
{
	int iterations;	/**< how many itterations where used */
	double fx;		/**< error function value remaining after fit */
	bool compleated;	/**< true if fit compleated sucessfully */
};

struct FitParameters
{
	int maxIter;
	double epsilon;
	double step;
	FitParameters(int maxIterI, double epsilonI = 1e-2, double stepI = 0.001): maxIter(maxIterI), epsilon(epsilonI), step(stepI){}
};

/**
....
* @}
*/
