from runtime.gpu_lanes import fits_on_one_gpu, pin_talk_lane, plan_talk_lane


def test_pin_drops_tensor_split_and_hides_other_gpus():
    args = ["-sm", "layer", "-mg", "0", "-np", "4", "-ts", "1.0,1.0", "-fit", "off", "-c", "8192"]
    out, env = pin_talk_lane(args, fits=True, talk_gpu=1)
    assert "-ts" not in out
    assert "1.0,1.0" not in out
    assert out[out.index("-mg") + 1] == "0"
    assert "-c" in out
    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    assert env["LLAMA_ARG_DEVICE"] == "CUDA0"


def test_oversized_model_keeps_split():
    args = ["-ts", "1,1", "-mg", "0"]
    out, env = plan_talk_lane(args, talk_gpu=1, estimated_bytes=40 << 30, total_bytes=24 << 30)
    assert out == args
    assert env == {}


def test_fourteen_b_fits_on_a_4090():
    assert fits_on_one_gpu(21 << 30, 24 << 30)
    out, env = plan_talk_lane(
        ["-mg", "0", "-ts", "1,1"],
        talk_gpu=1,
        estimated_bytes=21 << 30,
        total_bytes=24 << 30,
    )
    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    assert "-ts" not in out


def test_candidates_drop_slots_before_context():
    from runtime.gpu_lanes import talk_lane_candidates

    args = ["-np", "4", "-c", "32768", "-ts", "1,1"]
    cands = talk_lane_candidates(args)
    assert cands[0] == args
    assert cands[1][cands[1].index("-np") + 1] == "1"
    assert cands[1][cands[1].index("-c") + 1] == "32768"
    assert cands[-1][cands[-1].index("-c") + 1] == "8192"
