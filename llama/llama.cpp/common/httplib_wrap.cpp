// b9509 download.cpp / hf-cache.cpp link cpp-httplib from vendor/. CGO has no CMake
// dependency fetch; include httplib.cpp in a dedicated translation unit.
#include "../vendor/cpp-httplib/httplib.cpp"
