import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import json

from umm_reward_evaluator.benchmarks.common import annotate_oracle_best, summarize_headroom
from umm_reward_evaluator.benchmarks.randomize_planner_rank import randomize_manifest_rows
from umm_reward_evaluator.benchmarks.robotwin2_main_table_gate import evaluate_gate
from umm_reward_evaluator.benchmarks.robotwin2_readiness_report import collect_reports, render_markdown
from umm_reward_evaluator.benchmarks.robotwin2_selector_table import (
    collect_selector_rows,
    render_markdown as render_selector_table_markdown,
)
from umm_reward_evaluator.benchmarks.robotwin2_trace_field_audit import audit_rows
from umm_reward_evaluator.benchmarks.robotwin2_trace_to_manifest import (
    convert_records as convert_robotwin2,
    filter_records_by_case_size,
)
from umm_reward_evaluator.benchmarks.world_model_diagnostic_to_manifest import convert_records as convert_diagnostic


class BenchmarkAdaptersTest(unittest.TestCase):
    def test_common_groups_by_task_and_case(self):
        rows = [
            {
                "benchmark": "x",
                "suite": "s",
                "task_name": "task_a",
                "case_id": "seed=1",
                "candidate_id": "a0",
                "candidate_rank_by_planner": 0,
                "actions": [[0.0]],
                "oracle_success": False,
            },
            {
                "benchmark": "x",
                "suite": "s",
                "task_name": "task_a",
                "case_id": "seed=1",
                "candidate_id": "a1",
                "candidate_rank_by_planner": 1,
                "actions": [[1.0]],
                "oracle_success": True,
            },
            {
                "benchmark": "x",
                "suite": "s",
                "task_name": "task_b",
                "case_id": "seed=1",
                "candidate_id": "b0",
                "candidate_rank_by_planner": 0,
                "actions": [[0.0]],
                "oracle_success": True,
            },
        ]
        annotated = annotate_oracle_best(rows)
        best_by_task = {(row["task_name"], row["candidate_id"]): row["oracle_best_candidate_id"] for row in annotated}
        self.assertEqual(best_by_task[("task_a", "a0")], "a1")
        self.assertEqual(best_by_task[("task_b", "b0")], "b0")
        summary = summarize_headroom(annotated)
        self.assertEqual(summary["cases"], 2)
        self.assertEqual(summary["rank0_success"], 1)
        self.assertEqual(summary["oracle_success"], 2)

    def test_robotwin2_trace_conversion(self):
        rows = convert_robotwin2(
            [
                {
                    "task_name": "open_microwave",
                    "seed": 7,
                    "instruction": "open the microwave",
                    "policy_name": "DP",
                    "ckpt_setting": "baseline",
                    "candidate_seed": 0,
                    "actions": [[0, 1], [2, 3]],
                    "success": False,
                },
                {
                    "task_name": "open_microwave",
                    "seed": 7,
                    "instruction": "open the microwave",
                    "policy_name": "DP",
                    "ckpt_setting": "sampled",
                    "candidate_seed": 1,
                    "actions": [[1, 1], [3, 3]],
                    "success": True,
                },
            ],
            default_suite="demo_randomized",
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["benchmark"], "robotwin2")
        self.assertEqual(rows[0]["case_id"], "seed=7|instruction=open the microwave")
        self.assertEqual(rows[0]["metadata"]["future_source"], "policy_rollout")
        self.assertEqual(rows[0]["oracle_best_candidate_id"], rows[1]["candidate_id"])

    def test_robotwin2_trace_filter_drops_incomplete_cases(self):
        records = [
            {"task_name": "stamp_seal", "seed": 0, "candidate_seed": 0, "actions": [[0]], "success": False},
            {"task_name": "stamp_seal", "seed": 0, "candidate_seed": 1, "actions": [[1]], "success": True},
            {"task_name": "stamp_seal", "seed": 1, "candidate_seed": 0, "actions": [[0]], "success": False},
        ]
        filtered, dropped = filter_records_by_case_size(records, required_candidates_per_case=2)
        rows = convert_robotwin2(filtered, default_suite="demo_clean_k5")
        self.assertEqual(len(filtered), 2)
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(dropped), 1)
        self.assertEqual(dropped[0]["candidate_count"], 1)
        self.assertTrue(any(row["oracle_success"] for row in rows))
        self.assertEqual({row["case_id"] for row in rows}, {"seed=0|instruction="})

    def test_robotwin2_trace_filter_drops_candidate_error_cases(self):
        records = [
            {"task_name": "stack_blocks_two", "seed": 0, "candidate_seed": 0, "actions": [[0]], "success": False},
            {
                "task_name": "stack_blocks_two",
                "seed": 0,
                "candidate_seed": 1,
                "actions": [[1]],
                "success": False,
                "metadata": {"candidate_error": "OutOfMemoryError"},
            },
            {"task_name": "stack_blocks_two", "seed": 1, "candidate_seed": 0, "actions": [[0]], "success": False},
            {"task_name": "stack_blocks_two", "seed": 1, "candidate_seed": 1, "actions": [[1]], "success": True},
        ]
        filtered, dropped = filter_records_by_case_size(
            records,
            required_candidates_per_case=2,
            drop_cases_with_candidate_error=True,
        )
        rows = convert_robotwin2(filtered, default_suite="demo_clean_k5")
        self.assertEqual(len(filtered), 2)
        self.assertEqual({row["case_id"] for row in rows}, {"seed=1|instruction="})
        self.assertEqual(len(dropped), 1)
        self.assertEqual(dropped[0]["drop_reasons"], ["candidate_error"])
        self.assertEqual(dropped[0]["candidate_error_count"], 1)
        self.assertEqual(dropped[0]["candidate_error_candidate_ids"], ["policy:ckpt:1"])

    def test_robotwin2_main_table_gate_checks_errors_and_feature_coverage(self):
        rows = convert_robotwin2(
            [
                {
                    "task_name": "stack_blocks_two",
                    "seed": 0,
                    "candidate_id": "rank0",
                    "candidate_seed": 0,
                    "candidate_rank_by_planner": 0,
                    "actions": [[0]],
                    "success": False,
                    "metadata": {
                        "future_source": "unit",
                        "future_representation": "actions_and_state_trace",
                        "verification_target": "task_success",
                        "state_trace": [
                            {
                                "actor_pose_vector": [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
                                "actor_pairwise_distances": [1.0],
                                "left_gripper": [0.0],
                                "right_gripper": [0.0],
                                "joint_action_vector": [0.0],
                            }
                        ],
                    },
                },
                {
                    "task_name": "stack_blocks_two",
                    "seed": 0,
                    "candidate_id": "success",
                    "candidate_seed": 1,
                    "candidate_rank_by_planner": 1,
                    "actions": [[1]],
                    "success": True,
                    "metadata": {
                        "future_source": "unit",
                        "future_representation": "actions_and_state_trace",
                        "verification_target": "task_success",
                        "state_trace": [
                            {
                                "actor_pose_vector": [0, 0, 0, 1, 0, 0, 0, 0.2, 0, 0, 1, 0, 0, 0],
                                "actor_pairwise_distances": [0.2],
                                "left_gripper": [1.0],
                                "right_gripper": [1.0],
                                "joint_action_vector": [1.0],
                            }
                        ],
                    },
                },
            ],
            default_suite="demo_clean_k5",
        )
        passed = evaluate_gate(
            rows,
            required_candidates_per_case=2,
            min_cases=1,
            min_oracle_better_cases=1,
            required_feature_modes=["object_relation_distribution"],
        )
        self.assertTrue(passed["passed"])

        with_error = [dict(row) for row in rows]
        with_error[1]["metadata"] = dict(with_error[1]["metadata"])
        with_error[1]["metadata"]["candidate_error"] = "OutOfMemoryError"
        failed_error = evaluate_gate(with_error, required_candidates_per_case=2)
        self.assertFalse(failed_error["passed"])
        self.assertFalse({check["name"]: check["passed"] for check in failed_error["checks"]}["candidate_error_free"])

        missing_object = [dict(row) for row in rows]
        for row in missing_object:
            row["metadata"] = dict(row["metadata"])
            row["metadata"]["state_trace"] = [
                {key: value for key, value in snapshot.items() if not key.startswith("actor_")}
                for snapshot in row["metadata"]["state_trace"]
            ]
        failed_coverage = evaluate_gate(
            missing_object,
            required_candidates_per_case=2,
            required_feature_modes=["object_relation_distribution"],
        )
        self.assertFalse(failed_coverage["passed"])
        checks = {check["name"]: check for check in failed_coverage["checks"]}
        self.assertFalse(checks["feature_coverage:object_relation_distribution"]["passed"])
        self.assertEqual(checks["feature_coverage:object_relation_distribution"]["detail"]["case_coverage_rate"], 0.0)

    def test_robotwin2_readiness_report_summarizes_gate_outputs(self):
        with TemporaryDirectory() as tmp:
            selectors_dir = Path(tmp)
            base = {
                "passed": True,
                "summary": {"cases": 2, "rank0_success": 0, "oracle_success": 2, "oracle_better": 2},
                "checks": [{"name": "candidate_error_free", "passed": True, "detail": {}}],
                "feature_coverages": [],
            }
            relation = {
                "passed": False,
                "summary": {"cases": 2, "rank0_success": 0, "oracle_success": 2, "oracle_better": 2},
                "checks": [],
                "feature_coverages": [{"feature_mode": "object_relation_distribution", "case_coverage_rate": 0.0}],
            }
            (selectors_dir / "stack_blocks_two_targeted_energy_matched_main_table_gate.json").write_text(
                json.dumps(base),
                encoding="utf-8",
            )
            (selectors_dir / "stack_blocks_two_targeted_energy_matched_relation_gate.json").write_text(
                json.dumps(relation),
                encoding="utf-8",
            )

            rows = collect_reports(selectors_dir)
            markdown = render_markdown(rows)
            self.assertEqual(rows[0]["task_name"], "stack_blocks_two")
            self.assertTrue(rows[0]["base_gate_passed"])
            self.assertFalse(rows[0]["relation_gate_passed"])
            self.assertEqual(rows[0]["relation_min_case_coverage"], 0.0)
            self.assertIn("| stack_blocks_two | 2 | 0/2 | 2/2 | 2/2 | pass | fail | 0.00 |", markdown)

    def test_robotwin2_trace_field_audit_reports_actor_names_and_static_pollution(self):
        rows = [
            {
                "benchmark": "robotwin2",
                "suite": "unit",
                "task_name": "stack_blocks_two",
                "case_id": "seed=0",
                "candidate_id": "rank0",
                "candidate_rank_by_planner": 0,
                "actions": [[0.0]],
                "oracle_success": False,
                "metadata": {
                    "state_trace": [
                        {
                            "actor_names": ["block1", "table"],
                            "actor_pose_vector": [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
                            "actor_pairwise_distances": [1.0],
                        }
                    ]
                },
            },
            {
                "benchmark": "robotwin2",
                "suite": "unit",
                "task_name": "stack_blocks_two",
                "case_id": "seed=0",
                "candidate_id": "success",
                "candidate_rank_by_planner": 1,
                "actions": [[1.0]],
                "oracle_success": True,
                "metadata": {
                    "state_trace": [
                        {
                            "actor_names": ["block1", "table"],
                            "actor_pose_vector": [0, 0, 0, 1, 0, 0, 0, 0.2, 0, 0, 1, 0, 0, 0],
                            "actor_pairwise_distances": [0.2],
                        }
                    ]
                },
            },
        ]
        summary = audit_rows(rows)
        self.assertEqual(summary["actor_name_candidate_counts"]["block1"], 2)
        self.assertEqual(summary["actor_name_candidate_counts"]["table"], 2)
        self.assertEqual(summary["cases_with_actor_names"], 1)
        self.assertEqual(summary["cases_with_static_actor_pollution"], 1)
        self.assertEqual(summary["case_rows"][0]["static_actor_names"], ["table"])

    def test_robotwin2_selector_table_extracts_key_sweep_rows(self):
        with TemporaryDirectory() as tmp:
            selectors_dir = Path(tmp)
            sweep = {
                "seed_results": [{"selectors": [{"selector": "rank0", "cases": 2}]}],
                "aggregate": {
                    "selectors": [
                        {"selector": "rank0", "mean_success": 0.0},
                        {"selector": "random_expected", "mean_success": 0.9},
                        {"selector": "heuristic:energy_sum_max", "mean_success": 0.0},
                        {"selector": "prototype:gripper_distribution:same_task:nearest_positive", "mean_success": 1.0},
                        {
                            "selector": "prototype:object_relation_distribution:same_task:nearest_positive",
                            "mean_success": 2.0,
                            "min_feature_case_coverage": 1.0,
                        },
                    ]
                },
            }
            (selectors_dir / "stack_blocks_two_targeted_energy_matched_rankrand_sweep.json").write_text(
                json.dumps(sweep),
                encoding="utf-8",
            )

            rows = collect_selector_rows(selectors_dir)
            markdown = render_selector_table_markdown(rows)
            self.assertEqual(rows[0]["task_name"], "stack_blocks_two")
            self.assertEqual(rows[0]["cases"], 2)
            self.assertEqual(rows[0]["relation"], 2.0)
            self.assertEqual(rows[0]["relation_min_coverage"], 1.0)
            self.assertIn("| stack_blocks_two | 2 | 0.0/2 | 0.9/2 | 0.0/2", markdown)

    def test_world_model_diagnostic_label_and_score_conversion(self):
        rows = convert_diagnostic(
            [
                {
                    "benchmark": "mirabench",
                    "task_name": "push_button",
                    "case_id": "case0",
                    "candidate_id": "visually_best",
                    "label": "fail",
                    "video_path": "a.mp4",
                },
                {
                    "benchmark": "mirabench",
                    "task_name": "push_button",
                    "case_id": "case0",
                    "candidate_id": "action_faithful",
                    "scores": {"action_following": 0.9},
                    "video_path": "b.mp4",
                },
            ],
            default_benchmark="mirabench",
            default_suite="action_conditioned_reliability",
            default_verification_target="action_conditioned_reliability",
            score_key="scores.action_following",
            threshold=0.8,
        )
        self.assertFalse(rows[0]["oracle_success"])
        self.assertTrue(rows[1]["oracle_success"])
        self.assertEqual(rows[0]["metadata"]["future_representation"], "rgb_video")
        self.assertEqual(rows[0]["oracle_best_candidate_id"], "action_faithful")

    def test_rank_randomization_groups_by_task_and_case(self):
        rows = []
        for task in ("task_a", "task_b"):
            rows.extend(
                [
                    {
                        "benchmark": "robotwin2",
                        "suite": "unit",
                        "task_name": task,
                        "case_id": "seed=0",
                        "candidate_id": "rank0",
                        "candidate_rank_by_planner": 0,
                        "actions": [[0.0]],
                        "oracle_success": False,
                    },
                    {
                        "benchmark": "robotwin2",
                        "suite": "unit",
                        "task_name": task,
                        "case_id": "seed=0",
                        "candidate_id": "full",
                        "candidate_rank_by_planner": 1,
                        "actions": [[1.0]],
                        "oracle_success": True,
                    },
                ]
            )

        randomized = randomize_manifest_rows(rows, seed=0, mode="prefer_success", remap_candidate_ids=True)
        self.assertEqual(len(randomized), 4)
        self.assertEqual(
            sorted((row["task_name"], row["case_id"], row["candidate_rank_by_planner"]) for row in randomized),
            [
                ("task_a", "seed=0", 0),
                ("task_a", "seed=0", 1),
                ("task_b", "seed=0", 0),
                ("task_b", "seed=0", 1),
            ],
        )
        for row in randomized:
            if row["candidate_rank_by_planner"] == 0:
                self.assertTrue(row["oracle_success"])
                self.assertEqual(row["candidate_id"], "cand_00")
                self.assertEqual(row["oracle_best_candidate_id"], "cand_00")


if __name__ == "__main__":
    unittest.main()
