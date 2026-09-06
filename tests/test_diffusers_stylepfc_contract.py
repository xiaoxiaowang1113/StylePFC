import ast
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIFFUSERS_ROOT = PROJECT_ROOT / "diffusers_implementation"


def argparse_defaults(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    defaults = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        option = node.args[0].value
        for keyword in node.keywords:
            if keyword.arg == "default":
                try:
                    defaults[option] = ast.literal_eval(keyword.value)
                except (ValueError, TypeError):
                    # Some unrelated runner defaults are module-level variables.
                    # Only literal method defaults are compared by this contract.
                    pass
    return defaults


class DiffusersStylePFCContractTests(unittest.TestCase):
    def test_public_runners_and_dataset_utilities_exist(self):
        self.assertTrue((PROJECT_ROOT / "run_stylepfc.py").is_file())
        self.assertTrue((DIFFUSERS_ROOT / "run_stylepfc_diffusers.py").is_file())
        self.assertFalse((DIFFUSERS_ROOT / "run_styleid_diffusers.py").exists())
        for name in (
            "copy_inputs.py",
            "copy_inputs_flat.py",
            "prepare_flat_eval_refs.py",
            "eval_shared_category.py",
        ):
            self.assertTrue((PROJECT_ROOT / "util" / name).is_file(), name)

    def test_method_defaults_match_between_frameworks(self):
        compvis = argparse_defaults(PROJECT_ROOT / "run_stylepfc.py")
        diffusers = argparse_defaults(DIFFUSERS_ROOT / "config.py")
        pairs = {
            "--pfc_steps": "--pfc_steps",
            "--lambda_amp": "--lambda_amp",
            "--fusion_alpha": "--fusion_alpha",
            "--fusion_beta": "--fusion_beta",
            "--gamma": "--gamma",
            "--tau_c": "--tau_c",
            "--tau_s": "--tau_s",
            "--content_weight": "--content_weight",
            "--style_weight": "--style_weight",
        }
        for compvis_name, diffusers_name in pairs.items():
            self.assertEqual(
                compvis[compvis_name],
                diffusers[diffusers_name],
                f"default mismatch: {compvis_name} vs {diffusers_name}",
            )
        self.assertEqual(diffusers["--sd_version"], "1.5")
        self.assertEqual(diffusers["--init_mode"], "ca_adain")
        self.assertEqual(diffusers["--attn_mode"], "smsa")

    def test_diffusers_runner_wires_all_three_components(self):
        source = (DIFFUSERS_ROOT / "run_stylepfc_diffusers.py").read_text(
            encoding="utf-8"
        )
        for required in (
            "ca_adain(",
            "style_mixing_self_attention(",
            "phase_preserving_fourier_correction(",
            "weighted_writeback(",
            "content_clean_by_timestep",
        ):
            self.assertIn(required, source)
        self.assertNotIn("stylized_image.jpg", source)

    def test_style_mixing_self_attention_uses_joint_normalization(self):
        source = (DIFFUSERS_ROOT / "stable_diffusion.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "torch.cat([style_logits, content_logits], dim=-1).softmax(dim=-1)",
            source,
        )
        self.assertIn("torch.cat([v_style, v_content], dim=1)", source)


if __name__ == "__main__":
    unittest.main()
