// CGO stub for --license in common/arg.cpp. CMake llama-server builds generate
// this from licenses/LICENSE-* via cmake/license.cmake; zerollama CGO links common
// without running that step.

const char * LICENSES[] = {
    "MIT License (llama.cpp)\n"
    "See llama/llama.cpp/LICENSE and vendor elizaOS/llama.cpp licenses/.\n",
    nullptr,
};
