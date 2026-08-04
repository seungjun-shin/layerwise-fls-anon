import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "analyze_location_interaction.py"
SPEC = importlib.util.spec_from_file_location("analyze_location_interaction", SCRIPT_PATH)
assert SPEC is not None
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)

interaction_test = module.interaction_test
one_site_location_and_c = module.one_site_location_and_c


def test_one_site_location_and_c_for_global() -> None:
    config = {"fls": {"mode": "global", "global": {"output_multiplier": 0.25}}}
    assert one_site_location_and_c(config) == ("global", 0.25)


def test_one_site_location_and_c_for_single_block() -> None:
    config = {
        "fls": {
            "mode": "blockwise",
            "groups": {
                "early": {"output_multiplier": 1.0},
                "middle": {"output_multiplier": 0.5},
                "late": {"output_multiplier": 1.0},
                "head": {"output_multiplier": 1.0},
            },
        }
    }
    assert one_site_location_and_c(config) == ("middle", 0.5)


def test_one_site_location_and_c_for_single_boundary() -> None:
    config = {
        "fls": {
            "mode": "location",
            "boundaries": {
                "after_early": {"output_multiplier": 1.0},
                "after_middle": {"output_multiplier": 1.0},
                "after_late": {"output_multiplier": 0.0625},
            },
        }
    }
    assert one_site_location_and_c(config) == ("after_late", 0.0625)


def test_one_site_location_and_c_for_identity_boundary_config() -> None:
    config = {
        "fls": {
            "mode": "location",
            "boundaries": {
                "after_early": {"output_multiplier": 1.0},
                "after_middle": {"output_multiplier": 1.0},
                "after_late": {"output_multiplier": 1.0},
            },
        }
    }
    assert one_site_location_and_c(config) == ("identity", 1.0)


def test_one_site_location_and_c_rejects_multi_block_profile() -> None:
    config = {
        "fls": {
            "mode": "blockwise",
            "groups": {
                "early": {"output_multiplier": 0.5},
                "middle": {"output_multiplier": 1.0},
                "late": {"output_multiplier": 0.25},
                "head": {"output_multiplier": 1.0},
            },
        }
    }
    assert one_site_location_and_c(config) is None


def test_one_site_location_and_c_rejects_multi_boundary_profile() -> None:
    config = {
        "fls": {
            "mode": "location",
            "boundaries": {
                "after_early": {"output_multiplier": 0.5},
                "after_middle": {"output_multiplier": 1.0},
                "after_late": {"output_multiplier": 0.0625},
            },
        }
    }
    assert one_site_location_and_c(config) is None


def test_interaction_test_runs_on_synthetic_rows() -> None:
    rows = []
    for location in ["early", "late"]:
        for c in [0.25, 1.0]:
            for seed in [0, 1]:
                value = 0.5
                if location == "early" and c == 0.25:
                    value = 0.6
                if location == "late" and c == 1.0:
                    value = 0.7
                rows.append({"location": location, "c": c, "final_test_accuracy": value + seed * 0.001})
    result = interaction_test(rows, "final_test_accuracy")
    assert result is not None
    assert result["df_num"] > 0
    assert result["df_den"] > 0
