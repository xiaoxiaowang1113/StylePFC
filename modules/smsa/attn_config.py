import math


SMSA_MODES = (
    "smsa_full",
    "smsa_no_k_align",
    "smsa_no_v_align",
    "smsa_no_kv_align",
    "smsa_no_dual_t",
    "direct_replacement",
    "direct_addition",
)


def apply_style_mixing_self_attention_mode_settings(opt):
    direct_add_lambda = float(getattr(opt, "direct_add_lambda", 0.5))
    if not math.isfinite(direct_add_lambda) or direct_add_lambda < 0.0:
        raise ValueError("direct_add_lambda must be a finite nonnegative value")
    opt.direct_add_lambda_effective = direct_add_lambda
    opt.gamma_effective = opt.gamma

    if opt.smsa_mode == "smsa_full":
        opt.k_align_effective = True
        opt.v_align_effective = True
        opt.tau_s_effective = opt.tau_s
        opt.tau_c_effective = opt.tau_c
    elif opt.smsa_mode == "smsa_no_k_align":
        opt.k_align_effective = False
        opt.v_align_effective = True
        opt.tau_s_effective = opt.tau_s
        opt.tau_c_effective = opt.tau_c
    elif opt.smsa_mode == "smsa_no_v_align":
        opt.k_align_effective = True
        opt.v_align_effective = False
        opt.tau_s_effective = opt.tau_s
        opt.tau_c_effective = opt.tau_c
    elif opt.smsa_mode == "smsa_no_kv_align":
        opt.k_align_effective = False
        opt.v_align_effective = False
        opt.tau_s_effective = opt.tau_s
        opt.tau_c_effective = opt.tau_c
    elif opt.smsa_mode == "smsa_no_dual_t":
        opt.k_align_effective = True
        opt.v_align_effective = True
        opt.tau_s_effective = opt.T
        opt.tau_c_effective = opt.T
    elif opt.smsa_mode in {"direct_replacement", "direct_addition"}:
        # The direct baselines follow the paper definitions exactly:
        # pure cached-content query, raw K/V banks, and standard 1/sqrt(d)
        # scaling without SMSA alignment or dual-temperature routing.
        opt.gamma_effective = 1.0
        opt.k_align_effective = False
        opt.v_align_effective = False
        opt.tau_s_effective = 1.0
        opt.tau_c_effective = 1.0
    else:
        raise ValueError(f"Unsupported smsa_mode: {opt.smsa_mode}")
    return opt


def build_injection_config(opt, step_idx):
    return {
        "gamma": opt.gamma_effective,
        "T": getattr(opt, "T", opt.tau_s_effective),
        "timestep": step_idx,
        "tau_s": opt.tau_s_effective,
        "tau_c": opt.tau_c_effective,
        "smsa_mode": opt.smsa_mode,
        "k_align": opt.k_align_effective,
        "v_align": opt.v_align_effective,
        "direct_add_lambda": getattr(opt, "direct_add_lambda_effective", 0.5),
        "adain_eps": opt.adain_eps,
    }


# Style-Mixing Self-Attention (SMSA)
# Build the two attention branches used by SMSA: the style branch supplies raw
# style K/V banks, while the content branch supplies content K/V banks that are
# aligned inside attention before both branches enter one shared softmax.
def build_style_mixing_self_attention_features(opt, cnt_feat, sty_feat, step_idx):
    merged = {"config": build_injection_config(opt, step_idx)}
    q_keys = sorted(key for key in sty_feat.keys() if key.endswith("_q"))
    for q_key in q_keys:
        feature_prefix = q_key.rsplit("_", 1)[0]
        source_k_key = f"{feature_prefix}_k"
        source_v_key = f"{feature_prefix}_v"

        merged[q_key] = cnt_feat[q_key]
        merged[f"{feature_prefix}_k_style"] = sty_feat[source_k_key]
        merged[f"{feature_prefix}_k_content"] = cnt_feat[source_k_key]
        merged[f"{feature_prefix}_v_style_branch"] = sty_feat[source_v_key]
        merged[f"{feature_prefix}_v_content"] = cnt_feat[source_v_key]

    if "z_enc" in cnt_feat:
        merged["z_enc"] = cnt_feat["z_enc"]
    return merged


def prepare_style_mixing_self_attention_features(
    opt, cnt_feats, sty_feats, num_steps, start_step=0
):
    """Prepare timestep-indexed feature banks for Style-Mixing Self-Attention."""
    merged = [{"config": build_injection_config(opt, step_idx)} for step_idx in range(num_steps)]
    for step_idx in range(num_steps):
        if step_idx < (num_steps - start_step):
            continue
        merged[step_idx] = build_style_mixing_self_attention_features(
            opt, cnt_feats[step_idx], sty_feats[step_idx], step_idx
        )
    return merged
