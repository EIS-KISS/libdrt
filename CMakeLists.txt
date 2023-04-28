cmake_minimum_required(VERSION 3.19)

project(drtwrap LANGUAGES CXX)

link_directories(${CMAKE_CURRENT_BINARY_DIR})

set (CMAKE_CXX_STANDARD 20)

find_package(Torch REQUIRED)
find_package(Eigen3 REQUIRED)

function(dump_variables)
	get_cmake_property(_variableNames VARIABLES)
	list (SORT _variableNames)
	foreach (_variableName ${_variableNames})
		message(STATUS "${_variableName}=${${_variableName}}")
	endforeach()
endfunction()

add_executable(${PROJECT_NAME} main.cpp )
target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES} ${EIGEN3_LIBRARIES} eisgenerator)
target_include_directories(${PROJECT_NAME} PRIVATE ${TORCH_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS} ./LBFG)
target_compile_options(${PROJECT_NAME} PRIVATE "-Wall" "-O2" "-g" "-fno-strict-aliasing" "-Wfatal-errors" "-Wno-reorder")

install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)
