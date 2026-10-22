// DI3 host unit: verified accept_by_pos EWMA + survival expected accept.
// Compile via scripts/phase/di3_accept_by_pos_unit.sh (no GPU / llama-server).

#include "../server-adaptive-dm.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>

static void expect_near(float got, float want, float eps, const char * label) {
    if (std::fabs(got - want) > eps) {
        std::fprintf(stderr, "FAIL %s: got=%g want=%g\n", label, got, want);
        std::exit(1);
    }
}

int main() {
    // Clean case: all accept then none (alpha=1 → last sample wins).
    server_adaptive_dm_state st;
    st.dm_profit_ewma_alpha = 1.0f;
    st.dm_profit_min_samples = 1;
    st.observe_profit_acceptance(4, 4);
    float out[8] = {};
    assert(st.export_verified_accept_by_pos(out, 8) == 4);
    for (int i = 0; i < 4; ++i) {
        expect_near(out[i], 1.0f, 1e-5f, "all-accept ewma");
    }

    st.observe_profit_acceptance(4, 0);
    assert(st.export_verified_accept_by_pos(out, 8) == 4);
    for (int i = 0; i < 4; ++i) {
        expect_near(out[i], 0.0f, 1e-5f, "none-accept ewma");
    }

    // Survival expected accept needs min_samples.
    server_adaptive_dm_state st2;
    st2.dm_profit_ewma_alpha = 0.5f;
    st2.dm_profit_min_samples = 3;
    for (int i = 0; i < 5; ++i) {
        st2.observe_profit_acceptance(3, 2);
    }
    bool ready = false;
    const float expected = server_adaptive_dm_survival_expected_accept(
            st2.profit_pos_accept_ewma,
            st2.profit_pos_samples,
            server_adaptive_dm_state::PROFIT_POSITIONS,
            /*depth=*/2,
            st2.dm_profit_min_samples,
            &ready);
    assert(ready);
    assert(expected > 0.0f);
    assert(expected <= 2.0f + 1e-5f);

    float ewma[4];
    const int n_export = st2.export_verified_accept_by_pos(ewma, 4);
    assert(n_export == 3);

    std::printf("ok di3 accept_by_pos unit (export=%d expected=%.3f)\n", n_export, expected);
    return 0;
}
