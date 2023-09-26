cmake_minimum_required(VERSION 3.19)

project(eisdrt LANGUAGES CXX)

link_directories(${CMAKE_CURRENT_BINARY_DIR})

set (CMAKE_CXX_STANDARD 20)

find_package(Doxygen)
find_package(Eigen3 REQUIRED)
find_package(PkgConfig)
find_package(Torch)

if(DEFINED PKGCONFIG_FOUND)
	pkg_search_module(EIS libeisgenerator)
endif(DEFINED PKGCONFIG_FOUND)

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

if(DEFINED EIS_FOUND)
	message("eisgenerator support enabled")
	add_definitions(-DUSE_EISGEN)
else()
	message("eisgenerator support disabled")
endif(DEFINED EIS_FOUND)

if (DOXYGEN_FOUND)
	set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/doc/libeisdrt.doxygen.in)
	set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/doc/libeisdrt.doxygen)
	configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
	message("Doxygen support enabled, use target: doc")

	add_custom_target(doc
		COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
		WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
		COMMENT "Generating API documentation with Doxygen"
		VERBATIM )
else (DOXYGEN_FOUND)
	message("Doxygen needs to be installed to generate the doxygen documentation")
endif (DOXYGEN_FOUND)

add_library(${PROJECT_NAME} SHARED drt.cpp)
target_link_libraries(${PROJECT_NAME} ${EIGEN3_LIBRARIES} ${EIS_LIBRARIES} -leisgenerator)
target_include_directories(${PROJECT_NAME} PUBLIC ${EIGEN3_INCLUDE_DIRS} ${EIS_INCLUDE_DIRS} ./LBFG)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-Wall -O2 -march=native -g" LINK_FLAGS "-flto")
install(TARGETS ${PROJECT_NAME} DESTINATION lib)

if(DEFINED TORCH_LIBRARIES)
	message("torch support enabled")
	add_library(${PROJECT_NAME}_torch SHARED torchdrt.cpp)
	target_link_libraries(${PROJECT_NAME}_torch  ${TORCH_LIBRARIES} ${EIGEN3_LIBRARIES} ${PROJECT_NAME} ${EIS_LIBRARIES})
	target_include_directories(${PROJECT_NAME}_torch  PUBLIC ${TORCH_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS} ${EIS_INCLUDE_DIRS} ./LBFG)
	set_target_properties(${PROJECT_NAME}_torch  PROPERTIES COMPILE_FLAGS "-Wall -O2 -march=native -g" LINK_FLAGS "-flto")
	install(TARGETS ${PROJECT_NAME}_torch  DESTINATION lib)
else()
	message("torch support disabled")
endif(DEFINED TORCH_LIBRARIES)


if(DEFINED EIS_FOUND)
	link_directories(${CMAKE_CURRENT_BINARY_DIR})
	add_executable(${PROJECT_NAME}_test main.cpp)
	add_dependencies(${PROJECT_NAME}_test ${PROJECT_NAME})
	target_link_libraries(${PROJECT_NAME}_test -l${PROJECT_NAME} ${EIGEN3_LIBRARIES} ${EIS_LIBRARIES})
	target_include_directories(${PROJECT_NAME}_test PRIVATE . ${EIGEN3_INCLUDE_DIRS} ${EIS_INCLUDE_DIRS})
	set_target_properties(${PROJECT_NAME}_test PROPERTIES COMPILE_FLAGS "-Wall -O2 -march=native -g" LINK_FLAGS "-flto")

	if(DEFINED TORCH_LIBRARIES)
		link_directories(${CMAKE_CURRENT_BINARY_DIR})
		add_executable(${PROJECT_NAME}_test_torch testtorchdrt.cpp eistotorch.cpp)
		add_dependencies(${PROJECT_NAME}_test_torch ${PROJECT_NAME})
		target_link_libraries(${PROJECT_NAME}_test_torch ${PROJECT_NAME}_torch ${EIGEN3_LIBRARIES} ${EIS_LIBRARIES})
		target_include_directories(${PROJECT_NAME}_test_torch PRIVATE . ${EIGEN3_INCLUDE_DIRS} ${EIS_INCLUDE_DIRS})
		set_target_properties(${PROJECT_NAME}_test_torch PROPERTIES COMPILE_FLAGS "-Wall -O2 -march=native -g" LINK_FLAGS "-flto")
	endif(DEFINED TORCH_LIBRARIES)
endif(DEFINED EIS_FOUND)

if(DEFINED PKGCONFIG_FOUND)
	configure_file(pkgconfig/libeisdrt_torch.pc.in pkgconfig/libeisdrt_torch.pc @ONLY)
	configure_file(pkgconfig/libeisdrt.pc.in pkgconfig/libeisdrt.pc @ONLY)
	install(FILES ${CMAKE_CURRENT_BINARY_DIR}/pkgconfig/libeisdrt_torch.pc DESTINATION lib/pkgconfig)
	install(FILES ${CMAKE_CURRENT_BINARY_DIR}/pkgconfig/libeisdrt.pc DESTINATION lib/pkgconfig)
endif(DEFINED PKGCONFIG_FOUND)

set(API_HEADERS_DIR eisdrt/)
set(API_HEADERS
	${API_HEADERS_DIR}/eigendrt.h
	${API_HEADERS_DIR}/eisdrt.h
	${API_HEADERS_DIR}/eistorchdrt.h
	${API_HEADERS_DIR}/types.h
	${API_HEADERS_DIR}/torchdrt.h
)
install(FILES ${API_HEADERS} DESTINATION include/${PROJECT_NAME})
