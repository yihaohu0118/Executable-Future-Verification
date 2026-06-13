import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from umm_reward_evaluator.benchmarks import robotwin2_gripper_aware_trace as trace
from umm_reward_evaluator.benchmarks.robotwin2_selector_baselines import (
    evaluate_linear_probe,
    evaluate_trace_distance,
    evaluate_heuristic,
    evaluate_prototype,
    evaluate_random_expected,
    evaluate_rank0,
    feature_coverage,
)
from umm_reward_evaluator.benchmarks.robotwin2_antitemplate_diagnostics import diagnose_manifest
from umm_reward_evaluator.benchmarks.robotwin2_gripper_aware_trace import (
    CandidateSpec,
    build_candidates,
    compact_scene_state,
    discover_scene_actors,
    parse_seed_list,
    run_one_seed,
)
from umm_reward_evaluator.benchmarks.robotwin2_selector_failure_analysis import run_analysis
from umm_reward_evaluator.benchmarks.robotwin2_rank_randomization_sweep import (
    parse_prototype_config,
    parse_linear_probe_config,
    parse_trace_distance_config,
    run_sweep,
)
from umm_reward_evaluator.benchmarks.robotwin2_kshot_calibration_sweep import run_kshot_sweep


def make_row(
    task,
    case,
    candidate,
    rank,
    success,
    actions,
    left_gripper,
    right_gripper=None,
    actor_pose=None,
    actor_pairwise=None,
):
    if right_gripper is None:
        right_gripper = left_gripper
    actor_pose = actor_pose or [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    actor_pairwise = actor_pairwise or [0.0]
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
                    "actor_pose_vector": actor_pose,
                    "actor_pairwise_distances": actor_pairwise,
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

    def test_linear_probe_is_a_learned_verifier_baseline(self):
        selector = evaluate_linear_probe(
            self.rows,
            feature_mode="gripper_distribution",
            scope="same_task",
            l2=1.0,
        )
        self.assertEqual(selector["overall"]["cases"], 4)
        self.assertEqual(selector["overall"]["selector_success"], 4)
        self.assertEqual(selector["feature_coverage"]["case_coverage_rate"], 1.0)
        self.assertEqual(len(selector["scored_rows"]), len(self.rows))

    def test_nearest_pos_neg_uses_negative_neighbors(self):
        rows = [
            make_row("stack", "seed=0", "rank0", 0, False, [[0.0], [0.0]], [0.0, 0.0]),
            make_row("stack", "seed=0", "success", 1, True, [[1.0], [1.0]], [1.0, 1.0]),
            make_row("stack", "seed=0", "near_positive_failure", 2, False, [[1.0], [1.0]], [0.9, 0.9]),
            make_row("stack", "seed=1", "rank0", 0, False, [[0.0], [0.0]], [0.0, 0.0]),
            make_row("stack", "seed=1", "success", 1, True, [[1.0], [1.0]], [1.0, 1.0]),
            make_row("stack", "seed=1", "near_positive_failure", 2, False, [[1.0], [1.0]], [0.9, 0.9]),
        ]
        nearest_positive = evaluate_prototype(
            rows,
            feature_mode="gripper_distribution",
            scope="same_task",
            prototype_mode="nearest_positive",
        )
        contrastive = evaluate_prototype(
            rows,
            feature_mode="gripper_distribution",
            scope="same_task",
            prototype_mode="nearest_pos_neg",
        )
        self.assertGreaterEqual(
            contrastive["overall"]["selector_success"],
            nearest_positive["overall"]["selector_success"],
        )
        self.assertEqual(contrastive["overall"]["selector_success"], 2)

    def test_object_distribution_feature_uses_actor_pose_trace(self):
        rows = [
            make_row("stack", "seed=0", "rank0", 0, False, [[0.0]], [0.0], actor_pose=[0.0, 0.0, 0.0, 1, 0, 0, 0]),
            make_row("stack", "seed=0", "success", 1, True, [[1.0]], [1.0], actor_pose=[1.0, 0.0, 0.0, 1, 0, 0, 0]),
            make_row("stack", "seed=1", "rank0", 0, False, [[0.0]], [0.0], actor_pose=[0.1, 0.0, 0.0, 1, 0, 0, 0]),
            make_row("stack", "seed=1", "success", 1, True, [[1.0]], [1.0], actor_pose=[1.1, 0.0, 0.0, 1, 0, 0, 0]),
        ]
        prototype = evaluate_prototype(
            rows,
            feature_mode="object_distribution",
            scope="same_task",
            prototype_mode="nearest_positive",
        )
        self.assertEqual(prototype["overall"]["selector_success"], 2)

    def test_object_relation_feature_separates_same_gripper_outcomes(self):
        rows = [
            make_row(
                "stack",
                "seed=0",
                "rank0",
                0,
                False,
                [[1.0], [1.0]],
                [1.0, 1.0],
                actor_pose=[0.0, 0.0, 0.0, 1, 0, 0, 0, 2.0, 0.0, 0.0, 1, 0, 0, 0],
                actor_pairwise=[2.0],
            ),
            make_row(
                "stack",
                "seed=0",
                "success",
                1,
                True,
                [[1.0], [1.0]],
                [1.0, 1.0],
                actor_pose=[0.0, 0.0, 0.0, 1, 0, 0, 0, 0.2, 0.0, 0.0, 1, 0, 0, 0],
                actor_pairwise=[0.2],
            ),
            make_row(
                "stack",
                "seed=1",
                "rank0",
                0,
                False,
                [[1.0], [1.0]],
                [1.0, 1.0],
                actor_pose=[0.0, 0.0, 0.0, 1, 0, 0, 0, 2.1, 0.0, 0.0, 1, 0, 0, 0],
                actor_pairwise=[2.1],
            ),
            make_row(
                "stack",
                "seed=1",
                "success",
                1,
                True,
                [[1.0], [1.0]],
                [1.0, 1.0],
                actor_pose=[0.0, 0.0, 0.0, 1, 0, 0, 0, 0.3, 0.0, 0.0, 1, 0, 0, 0],
                actor_pairwise=[0.3],
            ),
        ]
        gripper = evaluate_prototype(
            rows,
            feature_mode="gripper_distribution",
            scope="same_task",
            prototype_mode="nearest_positive",
        )
        relation = evaluate_prototype(
            rows,
            feature_mode="object_relation_distribution",
            scope="same_task",
            prototype_mode="nearest_positive",
        )
        phase_relation = evaluate_prototype(
            rows,
            feature_mode="phase_object_relation_distribution",
            scope="same_task",
            prototype_mode="nearest_positive",
        )
        relation_dtw = evaluate_trace_distance(
            rows,
            feature_mode="dtw_object_relation",
            scope="same_task",
        )
        self.assertEqual(gripper["overall"]["selector_success"], 0)
        self.assertEqual(relation["overall"]["selector_success"], 2)
        self.assertEqual(phase_relation["overall"]["selector_success"], 2)
        self.assertEqual(relation_dtw["overall"]["selector_success"], 2)
        self.assertEqual(relation["feature_coverage"]["case_coverage_rate"], 1.0)
        self.assertEqual(relation_dtw["feature_coverage"]["row_coverage_rate"], 1.0)

    def test_object_relation_feature_reports_missing_object_trace_coverage(self):
        rows = [
            make_row("stack", "seed=0", "rank0", 0, False, [[1.0]], [1.0]),
            make_row("stack", "seed=0", "success", 1, True, [[1.0]], [1.0]),
            make_row("stack", "seed=1", "rank0", 0, False, [[1.0]], [1.0]),
            make_row("stack", "seed=1", "success", 1, True, [[1.0]], [1.0]),
        ]
        for row in rows:
            for snapshot in row["metadata"]["state_trace"]:
                snapshot.pop("actor_pose_vector")
                snapshot.pop("actor_pairwise_distances")

        coverage = feature_coverage(rows, "object_relation_distribution")
        selector = evaluate_prototype(
            rows,
            feature_mode="object_relation_distribution",
            scope="same_task",
            prototype_mode="nearest_positive",
        )
        self.assertEqual(coverage["rows_with_required_trace_keys"], 0)
        self.assertEqual(coverage["cases_with_all_candidates_covered"], 0)
        self.assertEqual(selector["feature_coverage"]["row_coverage_rate"], 0.0)

    def test_compact_scene_state_records_named_actor_poses(self):
        class Pose:
            p = [1.0, 2.0, 3.0]
            q = [1.0, 0.0, 0.0, 0.0]

        class Actor:
            def get_pose(self):
                return Pose()

        class Env:
            def __init__(self):
                self.block1 = Actor()
                self.table = Actor()
                self.wall = Actor()
                self.robot = Actor()

        state = compact_scene_state(Env())
        self.assertEqual(state["actor_names"], ["block1"])
        self.assertEqual(len(state["actor_pose_vector"]), 7)

    def test_compact_scene_state_discovers_nested_movable_actors(self):
        class Pose:
            p = [1.0, 2.0, 3.0]
            q = [1.0, 0.0, 0.0, 0.0]

        class Actor:
            def get_pose(self):
                return Pose()

        class Env:
            def __init__(self):
                self.objects = {"block2": Actor(), "table": Actor()}
                self.extra = [Actor()]
                self.wall = Actor()
                self.robot = Actor()

        actors = discover_scene_actors(Env())
        names = [name for name, _pose in actors]
        self.assertEqual(names, ["extra[0]", "objects.block2"])
        self.assertNotIn("objects.table", names)
        self.assertNotIn("wall", names)
        state = compact_scene_state(Env())
        self.assertEqual(state["actor_names"], names)
        self.assertEqual(len(state["actor_pairwise_distances"]), 1)

    def test_parse_seed_list_accepts_ranges_and_deduplicates(self):
        self.assertEqual(parse_seed_list("0, 2-4, 3, 7"), [0, 2, 3, 4, 7])
        with self.assertRaises(ValueError):
            parse_seed_list("4-2")
        with self.assertRaises(ValueError):
            parse_seed_list(" , ")

    def test_run_one_seed_writes_completed_outputs_atomically(self):
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "seed_0.jsonl"
            with (
                mock.patch.object(
                    trace,
                    "record_expert_actions",
                    return_value=([[1.0] * 14], {"expert_success": True, "expert_info": {}, "num_expert_actions": 1}),
                ),
                mock.patch.object(
                    trace,
                    "build_candidates",
                    return_value=[CandidateSpec("full", 0, [[1.0] * 14], "full_expert_trace")],
                ),
                mock.patch.object(
                    trace,
                    "run_candidate",
                    return_value={
                        "candidate_id": "full",
                        "success": True,
                        "actions": [[1.0] * 14],
                        "metadata": {"candidate_source": "full_expert_trace"},
                    },
                ),
            ):
                run_one_seed(
                    task_name="unit_task",
                    task_config="unit_config",
                    seed=0,
                    instruction="unit",
                    output=output,
                    skip_existing=False,
                    candidate_preset="targeted_energy_matched",
                    skip_replay_planner=False,
                )

            self.assertTrue(output.exists())
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 1)
            self.assertEqual(list(Path(tmp).glob("*.tmp.*")), [])

    def test_run_one_seed_does_not_publish_interrupted_outputs(self):
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "seed_0.jsonl"
            with (
                mock.patch.object(
                    trace,
                    "record_expert_actions",
                    return_value=([[1.0] * 14], {"expert_success": True, "expert_info": {}, "num_expert_actions": 1}),
                ),
                mock.patch.object(
                    trace,
                    "build_candidates",
                    return_value=[CandidateSpec("full", 0, [[1.0] * 14], "full_expert_trace")],
                ),
                mock.patch.object(trace, "run_candidate", side_effect=KeyboardInterrupt),
            ):
                with self.assertRaises(KeyboardInterrupt):
                    run_one_seed(
                        task_name="unit_task",
                        task_config="unit_config",
                        seed=0,
                        instruction="unit",
                        output=output,
                        skip_existing=False,
                        candidate_preset="targeted_energy_matched",
                        skip_replay_planner=False,
                    )

            self.assertFalse(output.exists())
            self.assertEqual(len(list(Path(tmp).glob("*.jsonl"))), 0)

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

    def test_targeted_energy_matched_candidate_preset_adds_long_failure_probes(self):
        actions = [[float(step + dim) for dim in range(14)] for step in range(6)]
        candidates = build_candidates(actions, "targeted_energy_matched")
        by_id = {candidate.candidate_id: candidate for candidate in candidates}
        self.assertIn("repeat_contact_long", by_id)
        self.assertIn("long_gripper_contact_pulse", by_id)
        self.assertIn("long_contact_joint_perturb_strong", by_id)
        self.assertIn("long_gripper_late_1", by_id)
        self.assertIn("long_reverse_contact", by_id)
        self.assertEqual(
            by_id["long_gripper_contact_pulse"].candidate_source,
            "energy_matched_gripper_contact_negative_probe",
        )
        self.assertEqual(
            by_id["long_contact_joint_perturb_strong"].candidate_source,
            "energy_matched_contact_negative_probe",
        )
        self.assertGreater(len(by_id["long_gripper_contact_pulse"].actions), len(by_id["repeat_contact_long"].actions))
        self.assertGreater(len(by_id["long_reverse_contact"].actions), len(by_id["repeat_contact_long"].actions))

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
        self.assertIn("prototype:object_relation_distribution:same_task:nearest_positive", summary["by_selector"])
        self.assertIn("trace_distance:dtw_object_relation:same_task:nearest_positive", summary["by_selector"])
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
            prototypes=(parse_prototype_config("gripper_distribution:same_task:nearest_pos_neg"),),
            trace_distances=(parse_trace_distance_config("dtw_gripper:same_task"),),
            linear_probes=(parse_linear_probe_config("gripper_distribution:same_task"),),
        )
        selectors = {row["selector"] for row in summary["aggregate"]["selectors"]}
        self.assertIn("heuristic:smoothness_max", selectors)
        self.assertIn("prototype:gripper_distribution:same_task:nearest_pos_neg", selectors)
        self.assertIn("trace_distance:dtw_gripper:same_task:nearest_positive", selectors)
        self.assertIn("linear_probe:gripper_distribution:same_task:ridge_l2_1", selectors)
        self.assertNotIn("heuristic:energy_sum_max", selectors)
        self.assertEqual(summary["heuristics"], ["smoothness_max"])
        self.assertEqual(summary["prototypes"], ["gripper_distribution:same_task:nearest_pos_neg"])
        self.assertEqual(summary["trace_distances"], ["dtw_gripper:same_task"])
        self.assertEqual(summary["linear_probes"], ["gripper_distribution:same_task"])

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
