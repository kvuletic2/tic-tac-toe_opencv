cmake_minimum_required(VERSION 3.23.0)
project(tic-tac-toe VERSION 0.1.0)

include(CTest)
enable_testing()

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(tic-tac-toe main.cpp)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)


target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )

target_compile_definitions(tic-tac-toe PUBLIC SOURCE_PATH="${CMAKE_SOURCE_DIR}/pictures/")