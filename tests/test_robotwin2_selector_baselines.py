import unittest

from umm_reward_evaluator.benchmarks.robotwin2_selector_baselines import (
    evaluate_trace_distance,
    evaluate_heuristic,
    evaluate_prototype,
    evaluate_random_expected,
    evaluate_rank0,
)
from umm_reward_evaluator.benchmarks.robotwin2_antitemplate_diagnostics import diagnose_manifest
from umm_reward_evaluator.benchmarks.robotwin2_gripper_aware_trace import build_candidates
from umm_reward_evaluator.benchmarks.robotwin2_selector_failure_analysis import run_analysis
from umm_reward_evaluator.benchmarks.robotwin2_rank_randomization_sweep import (
    parse_prototype_config,
    parse_trace_distance_config,
    run_sweep,
)
from umm_reward_evaluator.benchmarks.robotwin2_kshot_calibration_sweep import run_kshot_sweep


def make_row(task, case, candidate, rank, success, actions, left_gripper, right_gripper=None):
    if right_gripper is None:
        right_gripper = left_gripper
    return {
        "benchmark": "robotwin2",
        "suite": "unit",
        "task_name": task,
        "case_id": case,
        "candidate_id": candidate,
        "candidate_rank_by_planner": rank,
        "actions": actions,
        "oracle_success": success,
        "metadata": {
            "state_trace": [
                {
                    "joint_action_vector": action,
                    "left_gripper": [left],
                    "right_gripper": [right],
                }
                for action, left, right in zip(actions, left_gripper, right_gripper, strict=True)
            ],
        },
    }


class RoboTwin2SelectorBaselinesTest(unittest.TestCase):
    def setUp(self):
        self.rows = [
            make_row("stack", "seed=0", "rank0", 0, False, [[0.0], [0.0]], [0.0, 0.0]),
            make_row("stack", "seed=0", "full", 1, True, [[1.0], [1.0]], [1.0, 1.0]),
            make_row("stack", "seed=0", "reverse", 2, False, [[2.0], [2.0]], [0.4, 0.4]),
            make_row("stack", "seed=1", "rank0", 0, False, [[0.0], [0.0]], [0.0, 0.0]),
            make_row("stack", "seed=1", "full", 1, True, [[1.1], [1.1]], [1.0, 1.0]),
            make_row("stack", "seed=1", "reverse", 2, False, [[2.0], [2.0]], [0.4, 0.4]),
            make_row("stamp", "seed=0", "rank0", 0, False, [[0.0], [0.0]], [0.0, 0.0]),
            make_row("stamp", "seed=0", "full", 1, True, [[0.2], [0.2]], [0.7, 0.7]),
            make_row("stamp", "seed=0", "reverse", 2, False, [[1.0], [1.0]], [0.2, 0.2]),
            make_row("stamp", "seed=1", "rank0", 0, False, [[0.0], [0.0]], [0.0, 0.0]),
            make_row("stamp", "seed=1", "full", 1, True, [[0.3], [0.3]], [0.7, 0.7]),
            make_row("stamp", "seed=1", "reverse", 2, False, [[1.0], [1.0]], [0.2, 0.2]),
        ]

    def test_rank0_and_random_expected_summarize_case_headroom(self):
        rank0 = evaluate_rank0(self.rows)
        self.assertEqual(rank0["overall"]["cases"], 4)
        self.assertEqual(rank0["overall"]["rank0_success"], 0)
        self.assertEqual(rank0["overall"]["oracle_success"], 4)

        random_expected = evaluate_random_expected(self.rows)
        self.assertAlmostEqual(random_expected["overall"]["tie_expected_success"], 4 / 3)

    def test_action_heuristic_uses_tie_expected_success(self):
        heuristic = evaluate_heuristic(self.rows, "energy_mean_max")
        self.assertEqual(heuristic["overall"]["selector_success"], 0)
        self.assertEqual(heuristic["overall"]["oracle_success"], 4)

    def test_gripper_nearest_positive_recovers_leave_one_case_successes(self):
        prototype = evaluate_prototype(
            self.rows,
            feature_mode="gripper_distribution",
            scope="same_task",
            prototype_mode="nearest_positive",
        )
        self.assertEqual(prototype["overall"]["selector_success"], 4)
        self.assertEqual(prototype["overall"]["selector_oracle_match"], 4)
        self.assertEqual(len(prototype["scored_rows"]), len(self.rows))

    def test_phase_gripper_feature_is_fixed_width_across_trace_lengths(self):
        rows = list(self.rows)
        rows[1] = make_row("stack", "seed=0", "full", 1, True, [[1.0], [1.0], [1.0]], [1.0, 1.0, 1.0])
        prototype = evaluate_prototype(
            rows,
            feature_mode="phase_gripper_distribution",
            scope="same_task",
            prototype_mode="nearest_positive",
        )
        self.assertEqual(prototype["overall"]["selector_success"], 4)

    def test_dtw_trace_distance_is_a_strong_expert_similarity_control(self):
        selector = evaluate_trace_distance(
            self.rows,
            feature_mode="dtw_gripper",
            scope="same_task",
        )
        self.assertEqual(selector["overall"]["selector_success"], 4)
        self.assertEqual(selector["overall"]["selector_oracle_match"], 4)
        self.assertEqual(len(selector["scored_rows"]), len(self.rows))

    def test_antitemplate_diagnostics_find_non_full_success_and_matched_negative(self):
        rows = [
            make_row("stack", "seed=0", "rank0", 0, False, [[0.0], [0.0]], [0.0, 0.0]),
            make_row("stack", "seed=0", "full_gripper_aware", 1, True, [[1.0], [1.0]], [1.0, 1.0]),
            make_row("stack", "seed=0", "drop_last", 2, True, [[1.0], [0.9]], [1.0, 1.0]),
            make_row("stack", "seed=0", "late_grip_fail", 3, False, [[1.0], [1.0]], [1.0, 1.0]),
            make_row("stack", "seed=1", "rank0", 0, False, [[0.0], [0.0]], [0.0, 0.0]),
            make_row("stack", "seed=1", "full_gripper_aware", 1, True, [[1.1], [1.1]], [1.0, 1.0]),
        ]
        summary = diagnose_manifest(rows, feature_mode="dtw_joint_gripper")
        self.assertEqual(summary["overall"]["cases"], 2)
        self.assertEqual(summary["overall"]["non_full_success_cases"], 1)
        self.assertEqual(summary["overall"]["diverse_non_full_success_cases"], 1)
        self.assertEqual(summary["overall"]["matched_negative_cases"], 1)
        self.assertEqual(summary["by_task"]["stack"]["non_full_success_cases"], 1)

    def test_antitemplate_candidate_preset_adds_named_probe_sources(self):
        actions = [[float(step + dim) for dim in range(14)] for step in range(4)]
        candidates = build_candidates(actions, "anti_template")
        by_id = {candidate.candidate_id: candidate for candidate in candidates}
        self.assertIn("full_gripper_aware", by_id)
        self.assertIn("repeat_middle", by_id)
        self.assertIn("gripper_early_1", by_id)
        self.assertIn("gripper_late_1", by_id)
        self.assertIn("contact_joint_perturb", by_id)
        self.assertEqual(by_id["full_gripper_aware"].candidate_source, "full_expert_trace")
        self.assertEqual(by_id["gripper_late_1"].candidate_source, "matched_gripper_timing_negative_probe")
        self.assertGreater(len(by_id["repeat_middle"].actions), len(actions))
        self.assertLess(len(by_id["stride2_hold_endpoint"].actions), len(actions))

    def test_targeted_hard_candidate_preset_adds_near_neighbor_probes(self):
        actions = [[float(step + dim) for dim in range(14)] for step in range(6)]
        candidates = build_candidates(actions, "targeted_hard")
        by_id = {candidate.candidate_id: candidate for candidate in candidates}
        self.assertIn("repeat_contact_long", by_id)
        self.assertIn("repeat_middle_drop_final", by_id)
        self.assertIn("delete_contact_step", by_id)
        self.assertIn("contact_joint_perturb_strong", by_id)
        self.assertIn("gripper_contact_pulse", by_id)
        self.assertEqual(by_id["repeat_contact_long"].candidate_source, "targeted_time_warp_negative_probe")
        self.assertEqual(by_id["contact_joint_offset_small"].candidate_source, "targeted_contact_negative_probe")
        self.assertEqual(by_id["gripper_contact_pulse"].candidate_source, "targeted_gripper_contact_negative_probe")
        self.assertGreater(len(by_id["repeat_contact_long"].actions), len(actions))
        self.assertEqual(len(by_id["delete_contact_step"].actions), len(actions) - 1)
        self.assertEqual(len(by_id["gripper_contact_pulse"].actions), len(actions))

    def test_selector_failure_analysis_tracks_sources_after_remap(self):
        rows = []
        for row in self.rows:
            row = dict(row)
            metadata = dict(row.get("metadata") or {})
            metadata["candidate_source"] = "success_probe" if row["oracle_success"] else "failure_probe"
            row["metadata"] = metadata
            rows.append(row)
        summary = run_analysis(
            rows,
            seeds=[0],
            mode="failure_rank0_shuffle_rest",
            remap_candidate_ids=True,
        )
        self.assertIn("heuristic:smoothness_max", summary["by_selector"])
        self.assertGreater(summary["num_rows"], 0)
        sources = summary["by_selector"]["heuristic:smoothness_max"]["source_counts"]
        self.assertTrue(set(sources).issubset({"success_probe", "failure_probe"}))

    def test_rank_randomization_sweep_aggregates_multiple_seeds(self):
        summary = run_sweep(
            self.rows,
            seeds=[0, 1],
            mode="failure_rank0_shuffle_rest",
            remap_candidate_ids=True,
        )
        self.assertEqual(summary["num_seeds"], 2)
        aggregate = {row["selector"]: row for row in summary["aggregate"]["selectors"]}
        self.assertEqual(aggregate["rank0"]["mean_success"], 0.0)
        self.assertAlmostEqual(aggregate["random_expected"]["mean_success"], 4 / 3)
        self.assertEqual(aggregate["candidate_id:full_gripper_aware"]["mean_success"], 0.0)

    def test_rank_randomization_sweep_accepts_custom_selector_lists(self):
        summary = run_sweep(
            self.rows,
            seeds=[0],
            mode="failure_rank0_shuffle_rest",
            remap_candidate_ids=True,
            heuristics=("smoothness_max",),
            prototypes=(parse_prototype_config("gripper_distribution:same_task:nearest_positive"),),
            trace_distances=(parse_trace_distance_config("dtw_gripper:same_task"),),
        )
        selectors = {row["selector"] for row in summary["aggregate"]["selectors"]}
        self.assertIn("heuristic:smoothness_max", selectors)
        self.assertIn("prototype:gripper_distribution:same_task:nearest_positive", selectors)
        self.assertIn("trace_distance:dtw_gripper:same_task:nearest_positive", selectors)
        self.assertNotIn("heuristic:energy_sum_max", selectors)
        self.assertEqual(summary["heuristics"], ["smoothness_max"])
        self.assertEqual(summary["prototypes"], ["gripper_distribution:same_task:nearest_positive"])
        self.assertEqual(summary["trace_distances"], ["dtw_gripper:same_task"])

    def test_kshot_sweep_reports_source_plus_target_calibration(self):
        summary = run_kshot_sweep(
            self.rows,
            rank_seeds=[0],
            support_seeds=[0],
            k_values=[0, 1],
            feature_modes=("gripper_distribution",),
            mode="failure_rank0_shuffle_rest",
            remap_candidate_ids=True,
            include_source_tasks=True,
        )
        aggregate = {row["selector"]: row for row in summary["aggregate"]["selectors"]}
        self.assertIn("kshot:gripper_distribution:source_plus_target:k0", aggregate)
        self.assertIn("kshot:gripper_distribution:source_plus_target:k1", aggregate)
        self.assertEqual(summary["k_values"], [0, 1])
        self.assertEqual(summary["feature_modes"], ["gripper_distribution"])


if __name__ == "__main__":
    unittest.main()
