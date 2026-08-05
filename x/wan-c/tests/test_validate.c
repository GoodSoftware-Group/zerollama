#include "wan.h"
#include "wan_config.h"

#include <stdio.h>
#include <string.h>

int main(void) {
  char err[256];

  wan_gen_params good = {
      .prompt = "a cat",
      .negative_prompt = "",
      .width = 832,
      .height = 480,
      .frames = 49,
      .steps = 25,
      .cfg_scale = 5.0f,
      .shift = 5.0f,
      .seed = 42,
      .solver = WAN_SOLVER_UNIPC,
      .dtype = WAN_DTYPE_F16,
      .fps = 16,
  };
  if (wan_validate_params(&good, err, sizeof(err)) != 0) {
    fprintf(stderr, "good params rejected: %s\n", err);
    return 1;
  }

  wan_gen_params bad = good;
  bad.width = 833;
  if (wan_validate_params(&bad, err, sizeof(err)) == 0) {
    fprintf(stderr, "bad width accepted\n");
    return 1;
  }

  bad = good;
  bad.frames = 50;
  if (wan_validate_params(&bad, err, sizeof(err)) == 0) {
    fprintf(stderr, "bad frames accepted\n");
    return 1;
  }

  if (wan_validate_resolution(832, 480, 49, err, sizeof(err)) != 0) {
    fprintf(stderr, "resolution check failed: %s\n", err);
    return 1;
  }

  fprintf(stderr, "test_validate OK\n");
  return 0;
}
