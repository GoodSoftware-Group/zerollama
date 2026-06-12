// b9509 added jinja under common/jinja/ for chat templates. CGO compiles only *.cpp
// directly under common/ — not subdirs — so we unity-include sources here.
#include "jinja/lexer.cpp"
#include "jinja/parser.cpp"
#include "jinja/runtime.cpp"
#include "jinja/value.cpp"
#include "jinja/string.cpp"
#include "jinja/caps.cpp"
