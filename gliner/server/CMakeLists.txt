cmake_minimum_required(VERSION 3.18)
project(gliner_server LANGUAGES CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT GLINER_CPP_ROOT)
  message(FATAL_ERROR "Set -DGLINER_CPP_ROOT=path/to/GLiNER.cpp vendor tree")
endif()
if(NOT ONNXRUNTIME_ROOTDIR)
  message(FATAL_ERROR "Set -DONNXRUNTIME_ROOTDIR=path/to/onnxruntime")
endif()

# Build upstream gliner lib (sets ONNXRUNTIME_* from ROOTDIR).
set(ONNXRUNTIME_ROOTDIR "${ONNXRUNTIME_ROOTDIR}" CACHE PATH "" FORCE)
set(UPDATE_SUBMODULES ON CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
# WHY static: tokenizers_cpp.a is often built without -fPIC; shared libgliner.so fails link.
set(GLINER_BUILD_SHARED OFF CACHE BOOL "" FORCE)
add_subdirectory("${GLINER_CPP_ROOT}" "${CMAKE_BINARY_DIR}/gliner_cpp" EXCLUDE_FROM_ALL)

set(THIRD_PARTY "${CMAKE_CURRENT_SOURCE_DIR}/third_party")
if(NOT EXISTS "${THIRD_PARTY}/httplib.h")
  message(FATAL_ERROR "missing ${THIRD_PARTY}/httplib.h — run scripts/vendor/ensure_gliner_server_deps.sh")
endif()
if(NOT EXISTS "${THIRD_PARTY}/nlohmann/json.hpp")
  message(FATAL_ERROR "missing ${THIRD_PARTY}/nlohmann/json.hpp — run scripts/vendor/ensure_gliner_server_deps.sh")
endif()

add_executable(gliner-server gliner-server.cpp)
target_include_directories(gliner-server PRIVATE
  "${GLINER_CPP_ROOT}/include"
  "${THIRD_PARTY}"
)
target_link_libraries(gliner-server PRIVATE gliner)
# Linux: httplib may need pthread
find_package(Threads REQUIRED)
target_link_libraries(gliner-server PRIVATE Threads::Threads)

install(TARGETS gliner-server RUNTIME DESTINATION bin)
