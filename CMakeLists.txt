cmake_minimum_required(VERSION 3.19)

project(eisdrt LANGUAGES CXX)

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

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
	set(CMAKE_INSTALL_PREFIX "/usr" CACHE PATH "..." FORCE)
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

add_library(${PROJECT_NAME} SHARED eistotorch.cpp drt.cpp)
target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES} ${EIGEN3_LIBRARIES} eisgenerator)
target_include_directories(${PROJECT_NAME} PUBLIC ${TORCH_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS} ./LBFG)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-Wall -O2 -march=native -g" LINK_FLAGS "-flto")
install(TARGETS ${PROJECT_NAME} DESTINATION lib)

link_directories(${CMAKE_CURRENT_BINARY_DIR})
add_executable(${PROJECT_NAME}_test main.cpp)
add_dependencies(${PROJECT_NAME}_test ${PROJECT_NAME})
target_link_libraries(${PROJECT_NAME}_test -l${PROJECT_NAME} ${TORCH_LIBRARIES} eisgenerator)
target_include_directories(${PROJECT_NAME}_test PRIVATE . ${TORCH_INCLUDE_DIRS})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-Wall -O2 -march=native -g" LINK_FLAGS "-flto")
install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)

set(API_HEADERS_DIR eisdrt/)
set(API_HEADERS
	${API_HEADERS_DIR}/drt.h
)
install(FILES ${API_HEADERS} DESTINATION include/${PROJECT_NAME})
