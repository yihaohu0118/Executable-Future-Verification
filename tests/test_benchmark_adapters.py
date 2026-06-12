import unittest

from umm_reward_evaluator.benchmarks.common import annotate_oracle_best, summarize_headroom
from umm_reward_evaluator.benchmarks.randomize_planner_rank import randomize_manifest_rows
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
