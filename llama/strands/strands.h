#pragma once
#include "json.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// Pointer head for Strands Decider. Backbone hidden states come from
// llama-server; this head reads option token states + the final query token.
class strands_head {
  public:
    explicit strands_head(const std::string & path);
    ~strands_head();

    // pointers: token indices into hidden (option ends). Query = last prompt token.
    // rows JSON: array of { "type": 0|1|2, "pointers": [i,…] }
    std::vector<std::vector<float>> score(const std::vector<std::vector<float>> & hidden,
                                          const std::vector<int32_t> & tokens,
                                          const common_json & rows);

  private:
    struct impl;
    std::unique_ptr<impl> p;
};
