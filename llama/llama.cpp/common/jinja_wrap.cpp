// b9509 added jinja under common/jinja/ for chat templates. CGO compiles only *.cpp
// directly under common/ — not subdirs — so we unity-include sources here.
#include "jinja/lexer.cpp"
#undef FILENAME
#include "jinja/parser.cpp"
#undef FILENAME
#include "jinja/runtime.cpp"
#undef FILENAME
#include "jinja/value.cpp"
#undef FILENAME
#include "jinja/string.cpp"
#include "jinja/caps.cpp"
#undef FILENAME
