#include <eisgenerator/eistype.h>
#include <vector>

#include "types.h"

/**
Api for use with eisgenerator applications
* @defgroup EISAPI eisgenerator API
* calculates drts with eisgenerator datatypes
* @{
*/

/**
 * @brief calculate a drt on eisgenerator types
 *
 * @param data a vector of eisgenerator datapoints with the values to your expirament, embedded omega values are ignored
 * @param omegaVector vector with the omega values that the impedances where mesured at
 * @param fm a fit metrics struct where this function returns information on the fit aquired
 * @param fp a struct with fit parameters
 * @return a vector with the drt values
 */
std::vector<fvalue> calcDrt(const std::vector<eis::DataPoint>& data, const std::vector<fvalue>& omegaVector, FitMetics& fm, const FitParameters& fp);

/**
 * @brief calculate a drt on eisgenerator types
 *
 * @param data a vector of eisgenerator datapoints with the values to your expirament, embedded omega values are used
 * @param fm a fit metrics struct where this function returns information on the fit aquired
 * @param fp a struct with fit parameters
 * @return a vector with the drt values
 */
std::vector<fvalue> calcDrt(const std::vector<eis::DataPoint>& data, FitMetics& fm, const FitParameters& fp);

/**
....
* @}
*/
