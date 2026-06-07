from __future__ import annotations


SYSTEM_PROMPT = "You are a strict world-model rollout evaluator. Return only valid JSON."


ROLL_OUT_REWARD_PROMPT = """Evaluate whether this action-conditioned world-model rollout is useful for the task.

Inputs:
- Task instruction: {instruction}
- Task/domain: {task}
- Candidate id: {candidate_id}
- Action summary: {action_summary}

Judge the sampled rollout frames in temporal order. Penalize visually plausible videos if they:
- do not follow the action sequence,
- only look successful in the final frame but have impossible intermediate dynamics,
- ignore contact or physical constraints,
- drift object identity, scene layout, or goal-relevant state,
- are temporally shuffled, reversed, or inconsistent.

Return JSON with exactly these fields:
{{
  "task_success_score": 0-5,
  "task_progress_score": 0-5,
  "action_outcome_consistency_score": 0-5,
  "temporal_consistency_score": 0-5,
  "physical_plausibility_score": 0-5,
  "identity_layout_consistency_score": 0-5,
  "overall_score": 0-100,
  "verdict": "pass|partial|fail",
  "reason": "short evidence grounded in the frames and task"
}}

Scoring guidance:
- overall_score should reflect task usefulness for planning, not only visual quality.
- If action information is missing, reduce confidence and focus on task/temporal consistency.
- Be strict. A smooth but task-wrong rollout should receive a low score.
"""
