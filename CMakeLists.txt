cmake_minimum_required(VERSION 3.26.2)
project(DisplayEdge)

# PyTorch configuration 
set(CMAKE_PREFIX_PATH libtorch)

set(OPENCV_EXTRA_MODULES_PATH opencv/build)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

set(CMAKE_PREFIX_PATH opencv/build)


find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(${PROJECT_NAME} main.cpp)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_range_for)

target_link_libraries(${PROJECT_NAME} "${OpenCV_LIBRARIES}")
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")
                            if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET ${PROJECT_NAME}
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:${PROJECT_NAME}>)
endif (MSVC)