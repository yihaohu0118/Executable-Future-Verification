import unittest

from umm_reward_evaluator.hard_negatives import build_negatives
from umm_reward_evaluator.manifest import group_by_case, validate_row


class ManifestAndNegativesTest(unittest.TestCase):
    def test_validate_minimal_row(self):
        row = {
            "case_id": "case_0",
            "candidate_id": "cand_0",
            "task": "pusht",
            "instruction": "push to goal",
            "frame_paths": ["/tmp/a.png", "/tmp/b.png"],
        }
        validate_row(row)

    def test_build_hard_negatives(self):
        rows = [
            {
                "case_id": "case_0",
                "candidate_id": "cand_0",
                "task": "pusht",
                "instruction": "push to goal",
                "frame_paths": ["a.png", "b.png", "c.png"],
                "actions": [[1, 0], [2, 0], [3, 0]],
            },
            {
                "case_id": "case_1",
                "candidate_id": "cand_0",
                "task": "pusht",
                "instruction": "push away",
                "frame_paths": ["d.png", "e.png", "f.png"],
                "actions": [[0, 1], [0, 2], [0, 3]],
            },
        ]
        negatives = build_negatives(
            rows,
            ["temporal_reverse", "temporal_shuffle", "action_shuffle", "goal_swap"],
            seed=0,
        )
        self.assertEqual(len(negatives), 8)
        grouped = group_by_case(negatives)
        self.assertEqual(set(grouped), {"case_0", "case_1"})
        reverse = [r for r in negatives if r["negative_type"] == "temporal_reverse"][0]
        self.assertEqual(reverse["frame_paths"], ["c.png", "b.png", "a.png"])


if __name__ == "__main__":
    unittest.main()
