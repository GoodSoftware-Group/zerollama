// Unit/lab smoke for ane_ffn_policy — no ANE compile, no serve.
// JSON: ok, mode=ane_ffn_policy, cases_passed, …

#import <Foundation/Foundation.h>
#include "ane_ffn_policy.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void emit(BOOL ok, const char *err, NSDictionary *fields) {
    NSMutableString *json = [NSMutableString stringWithFormat:@"{\"ok\":%@", ok ? @"true" : @"false"];
    for (NSString *key in fields) {
        id val = fields[key];
        if ([val isKindOfClass:[NSNumber class]]) {
            [json appendFormat:@",\"%@\":%@", key, val];
        } else if ([val isKindOfClass:[NSString class]]) {
            NSString *s = [(NSString *)val stringByReplacingOccurrencesOfString:@"\\" withString:@"\\\\"];
            s = [s stringByReplacingOccurrencesOfString:@"\"" withString:@"\\\""];
            [json appendFormat:@",\"%@\":\"%@\"", key, s];
        }
    }
    if (err && err[0]) {
        NSString *escaped = [[NSString stringWithUTF8String:err]
            stringByReplacingOccurrencesOfString:@"\\" withString:@"\\\\"];
        escaped = [escaped stringByReplacingOccurrencesOfString:@"\"" withString:@"\\\""];
        [json appendFormat:@",\"error\":\"%@\"", escaped];
    }
    [json appendString:@"}\n"];
    printf("%s", [json UTF8String]);
}

static void clear_ffn_env(void) {
    unsetenv("ZEROLLAMA_ANE_FFN");
    unsetenv("ZEROLLAMA_ANE_FFN_MODE");
    unsetenv("ZEROLLAMA_ANE_FFN_IC");
    unsetenv("ZEROLLAMA_ANE_FFN_OC");
    unsetenv("ZEROLLAMA_ANE_FFN_SEQ_MAX");
    unsetenv("ZEROLLAMA_ANE_FFN_LAB_PORT");
    unsetenv("ZEROLLAMA_ANE_FFN_PORT");
    unsetenv("ZEROLLAMA_ANE_FFN_TELEMETRY");
    unsetenv("ZEROLLAMA_ANE_FFN_NAME");
    unsetenv("ZEROLLAMA_ANE_FFN_FORCE_ENABLE");
}

static int fail_count = 0;

static void expect(const char *name, bool cond) {
    if (!cond) {
        fprintf(stderr, "FAIL: %s\n", name);
        fail_count++;
    }
}

int main(int argc, const char *argv[]) {
    (void)argc;
    (void)argv;
    @autoreleasepool {
        clear_ffn_env();

        // 1) Default off
        {
            ane_ffn_policy_t p;
            expect("default_disabled", !ane_ffn_policy_load(&p));
            ane_ffn_verdict_t v = ane_ffn_policy_decide(&p, ANE_FFN_OP_MUL_MAT, 2048, 512, 256, 11435, NULL);
            expect("default_no_allow", !v.allow);
        }

        // 2) Shadow match on lab port
        {
            setenv("ZEROLLAMA_ANE_FFN", "1", 1);
            setenv("ZEROLLAMA_ANE_FFN_MODE", "shadow", 1);
            setenv("ZEROLLAMA_ANE_FFN_IC", "2048", 1);
            setenv("ZEROLLAMA_ANE_FFN_OC", "512", 1);
            setenv("ZEROLLAMA_ANE_FFN_SEQ_MAX", "512", 1);
            setenv("ZEROLLAMA_ANE_FFN_LAB_PORT", "11435", 1);
            ane_ffn_policy_t p;
            expect("shadow_enabled", ane_ffn_policy_load(&p));
            expect("shadow_mode", p.mode == ANE_FFN_MODE_SHADOW);
            ane_ffn_verdict_t v = ane_ffn_policy_decide(&p, ANE_FFN_OP_MUL_MAT, 2048, 512, 256, 11435, NULL);
            expect("shadow_allow", v.allow);
            expect("shadow_reason", strcmp(v.reason, "shadow_match") == 0);
        }

        // 3) Refuse production 11434 even in force
        {
            setenv("ZEROLLAMA_ANE_FFN_MODE", "force", 1);
            setenv("ZEROLLAMA_ANE_FFN_LAB_PORT", "11434", 1);
            ane_ffn_policy_t p;
            ane_ffn_policy_load(&p);
            ane_ffn_verdict_t v = ane_ffn_policy_decide(&p, ANE_FFN_OP_MUL_MAT, 2048, 512, 256, 11434, NULL);
            expect("prod_refused", !v.allow);
            expect("prod_reason", strcmp(v.reason, "production_port_refused") == 0);
        }

        // 4) Refuse 8081
        {
            setenv("ZEROLLAMA_ANE_FFN_LAB_PORT", "18081", 1);
            ane_ffn_policy_t p;
            ane_ffn_policy_load(&p);
            ane_ffn_verdict_t v = ane_ffn_policy_decide(&p, ANE_FFN_OP_MUL_MAT, 2048, 512, 256, 8081, NULL);
            expect("sidecar_refused", !v.allow);
        }

        // 5) MUL_MAT_ID never
        {
            setenv("ZEROLLAMA_ANE_FFN_LAB_PORT", "11435", 1);
            ane_ffn_policy_t p;
            ane_ffn_policy_load(&p);
            ane_ffn_verdict_t v = ane_ffn_policy_decide(&p, ANE_FFN_OP_MUL_MAT_ID, 2048, 512, 256, 11435, NULL);
            expect("no_mul_mat_id", !v.allow);
        }

        // 6) Geometry filter
        {
            ane_ffn_policy_t p;
            ane_ffn_policy_load(&p);
            ane_ffn_verdict_t v = ane_ffn_policy_decide(&p, ANE_FFN_OP_MUL_MAT, 1024, 512, 256, 11435, NULL);
            expect("ic_mismatch", !v.allow && strcmp(v.reason, "ic_mismatch") == 0);
        }

        // 7) Parse host port
        expect("parse_host", ane_ffn_policy_parse_host_port("127.0.0.1:11435") == 11435);
        expect("parse_prod", ane_ffn_policy_is_production_port(11434));

        // 8) Shadow note increments match counter (Metal-path helper)
        {
            clear_ffn_env();
            setenv("ZEROLLAMA_ANE_FFN", "1", 1);
            setenv("ZEROLLAMA_ANE_FFN_MODE", "shadow", 1);
            setenv("ZEROLLAMA_ANE_FFN_IC", "2048", 1);
            setenv("ZEROLLAMA_ANE_FFN_OC", "512", 1);
            setenv("ZEROLLAMA_ANE_FFN_LAB_PORT", "11435", 1);
            setenv("OLLAMA_HOST", "127.0.0.1:11435", 1);
            uint64_t before = ane_ffn_shadow_match_count();
            ane_ffn_shadow_note_mul_mat(2048, 512, 128, "blk.0.ffn_up_shexp.weight");
            expect("shadow_note_match", ane_ffn_shadow_match_count() == before + 1);
            ane_ffn_shadow_note_mul_mat(1024, 512, 128, "blk.0.ffn_up_shexp.weight"); // ic mismatch
            expect("shadow_note_no_false", ane_ffn_shadow_match_count() == before + 1);
        }

        // 9) Force try is fail-closed until ANE replace is wired
        {
            setenv("ZEROLLAMA_ANE_FFN_MODE", "force", 1);
            unsetenv("ZEROLLAMA_ANE_FFN_FORCE_ENABLE");
            expect("force_no_enable", !ane_ffn_force_try_mul_mat(2048, 512, 128, NULL));
            setenv("ZEROLLAMA_ANE_FFN_FORCE_ENABLE", "1", 1);
            uint64_t before = ane_ffn_force_deferred_count();
            expect("force_still_false", !ane_ffn_force_try_mul_mat(2048, 512, 128, NULL));
            expect("force_deferred", ane_ffn_force_deferred_count() == before + 1);
        }

        // 10) shexp name preset
        {
            clear_ffn_env();
            setenv("ZEROLLAMA_ANE_FFN", "1", 1);
            setenv("ZEROLLAMA_ANE_FFN_MODE", "shadow", 1);
            setenv("ZEROLLAMA_ANE_FFN_NAME", "shexp", 1);
            setenv("ZEROLLAMA_ANE_FFN_LAB_PORT", "11435", 1);
            ane_ffn_policy_t p;
            expect("shexp_load", ane_ffn_policy_load(&p) && p.n_name_pats == 3);
            ane_ffn_verdict_t ok = ane_ffn_policy_decide(
                &p, ANE_FFN_OP_MUL_MAT, 2048, 512, 128, 11435, "blk.3.ffn_up_shexp.weight");
            expect("shexp_match", ok.allow && ok.name_match);
            ane_ffn_verdict_t bad = ane_ffn_policy_decide(
                &p, ANE_FFN_OP_MUL_MAT, 2048, 512, 128, 11435, "blk.3.ffn_up.weight");
            expect("shexp_reject_dense", !bad.allow && strcmp(bad.reason, "name_mismatch") == 0);
            ane_ffn_verdict_t exps = ane_ffn_policy_decide(
                &p, ANE_FFN_OP_MUL_MAT, 2048, 512, 128, 11435, "blk.3.ffn_up_exps.weight");
            expect("shexp_reject_exps", !exps.allow);
            ane_ffn_verdict_t miss = ane_ffn_policy_decide(
                &p, ANE_FFN_OP_MUL_MAT, 2048, 512, 128, 11435, NULL);
            expect("shexp_reject_missing", !miss.allow && strcmp(miss.reason, "name_missing") == 0);
        }

        // 11) dense ffn preset rejects shexp
        {
            setenv("ZEROLLAMA_ANE_FFN_NAME", "ffn", 1);
            ane_ffn_policy_t p;
            ane_ffn_policy_load(&p);
            ane_ffn_verdict_t dens = ane_ffn_policy_decide(
                &p, ANE_FFN_OP_MUL_MAT, 2048, 512, 128, 11435, "blk.0.ffn_gate.weight");
            expect("ffn_match_dense", dens.allow);
            ane_ffn_verdict_t she = ane_ffn_policy_decide(
                &p, ANE_FFN_OP_MUL_MAT, 2048, 512, 128, 11435, "blk.0.ffn_gate_shexp.weight");
            expect("ffn_reject_shexp", !she.allow);
        }

        clear_ffn_env();
        unsetenv("OLLAMA_HOST");
        if (fail_count > 0) {
            emit(NO, "policy unit failures", @{
                @"mode": @"ane_ffn_policy",
                @"fail_count": @(fail_count),
            });
            return 1;
        }
        emit(YES, NULL, @{
            @"mode": @"ane_ffn_policy",
            @"fail_count": @0,
            @"shadow_match_count": @(ane_ffn_shadow_match_count()),
            @"force_deferred_count": @(ane_ffn_force_deferred_count()),
            @"note": @"fail-closed; MUL_MAT only; shexp/ffn name filter; prod ports refused",
            @"source": @"zerollama/ane_ffn_policy",
        });
        return 0;
    }
}
