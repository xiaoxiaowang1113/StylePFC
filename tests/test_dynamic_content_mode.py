import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUN = (ROOT / "run_stylepfc.py").read_text(encoding="utf-8")
DDIM = (ROOT / "ldm/models/diffusion/ddim.py").read_text(encoding="utf-8")
PHASE_FUSION = (ROOT / "modules/pfc/phase_fusion.py").read_text(
    encoding="utf-8"
)


class DynamicContentModeContractTest(unittest.TestCase):
    def test_mode_registration_preserves_comparison_modes(self):
        tree = ast.parse(PHASE_FUSION)
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "PFC_MODES"
                for target in node.targets
            ):
                self.assertEqual(
                    ("paper", "dynamic_content", "clean_triplet"),
                    ast.literal_eval(node.value),
                )
                break
        else:
            self.fail("PFC_MODES assignment not found")

    def test_public_runner_is_fixed_to_dynamic_content(self):
        self.assertIn(
            'PHASE_PRESERVING_FOURIER_CORRECTION_SAMPLER_MODE = "dynamic_content"',
            RUN,
        )
        self.assertNotIn('"--fpd_mode"', RUN)
        self.assertNotIn('"--ablation_case"', RUN)
        self.assertNotIn('"--without_fpd"', RUN)
        self.assertNotIn('"--without_attn_injection"', RUN)
        self.assertNotIn('"--without_init_adain"', RUN)
        self.assertIn('save_feature_map(pred_x0, "pred_x0", timestep)', RUN)

    def test_checkpoint_resolution_is_remote_first_with_local_fallback(self):
        self.assertIn("DEFAULT_CHECKPOINT_URL", RUN)
        self.assertIn("REMOTE_CHECKPOINT_CACHE_ROOT = PROJECT_ROOT", RUN)
        self.assertIn('os.environ.get("HF_TOKEN")', RUN)
        resolver = RUN[RUN.index("def resolve_model_checkpoint"):RUN.index("def parse_pfc_steps")]
        self.assertLess(
            resolver.index("download_remote_checkpoint(checkpoint_url)"),
            resolver.index("is_usable_checkpoint(local_checkpoint_path)"),
        )

    def test_runner_builds_only_the_content_clean_trajectory(self):
        self.assertIn(
            "Phase-Preserving Fourier Correction requires content inversion", RUN
        )
        self.assertIn("content_clean_latents=content_clean_latents_seq", RUN)
        self.assertIn("style_clean_latents=None", RUN)
        self.assertNotIn("style_clean_latents_seq", RUN)

    def test_sampler_uses_the_matching_loop_index(self):
        self.assertIn('fpd_mode == "dynamic_content"', DDIM)
        self.assertIn("content_clean_latents[i]", DDIM)
        self.assertIn("step_id = i + 1", DDIM)
        self.assertIn("total_steps - i - 1 < start_step", DDIM)
        self.assertIn("i + 1 in fpd_steps_set", DDIM)

    def test_dynamic_content_decodes_corrects_encodes_and_writes_back(self):
        self.assertIn("phase_preserving_fourier_correction", DDIM)
        self.assertIn("self.model.decode_first_stage(x_c_clean)", DDIM)
        self.assertIn("self.model.decode_first_stage(pred_x0)", DDIM)
        self.assertIn("self.model.encode_first_stage(aligned_image)", DDIM)
        self.assertIn("weighted_writeback(", DDIM)

    def test_interrupted_images_are_quarantined_not_deleted(self):
        self.assertNotIn(".unlink(", RUN)
        self.assertIn("quarantine_stale_part", RUN)
        self.assertIn("path.replace(target)", RUN)

    def test_legacy_diagnostic_hook_remains_available(self):
        self.assertIn("legacy_latent_mode", DDIM)
        self.assertIn("self._latent_phase_fusion(", DDIM)


if __name__ == "__main__":
    unittest.main()
