# Templates

Use these as starter files when materializing the lab in a target repo.

## Goal Mission

Path:

```text
goals/<goal-slug>/MISSION.md
```

Template:

```md
# <Goal Title>

## Objective

<Human-written statement of the research objective.>

## Success Metric

- Metric: <metric name>
- Direction: <lower|higher>
- Target: <target value or qualitative finish line>

## Scope

- In scope: <work that is allowed>
- Out of scope: <work that is not allowed>

## Constraints

- Budget: <budget or compute ceiling>
- Deadline: <date or window>
- Safety or policy constraints: <rules>
```

## Goal Metadata

Path:

```text
goals/<goal-slug>/goal.json
```

Template:

```json
{
  "project": "<project-name>",
  "metric_key": "<metric-name>",
  "metric_direction": "min",
  "training_window_seconds": null,
  "training_window_anchor": "first_dispatch",
  "deadline_utc": null,
  "safety_margin_seconds": null,
  "calibration_required": true,
  "calibration_source": null,
  "reactive_set_only": true,
  "report_due_at": null
}
```

## Year Plan

Path:

```text
goals/<goal-slug>/plans/year.md
```

Template:

```md
# Year Plan

## Goal

<Restate the mission in operational terms.>

## Major Bottlenecks

- <bottleneck 1>
- <bottleneck 2>

## Quarter Themes

- Q1: <theme>
- Q2: <theme>
- Q3: <theme>
- Q4: <theme>

## Current Focus

<What matters now and why.>
```

## Week Plan

Path:

```text
goals/<goal-slug>/plans/<YYYY>_w<WW>.md
```

Template:

```md
# Week Plan

## Objective

<What the lab must learn or unlock this week.>

## Active Campaign

<campaign name>

## Active Set

- Set 1: <goal, hypothesis class, compute budget>

## Search Strategy

- Strategy: <wide_search|deep_validation|tiered_exploration>
- Why this strategy fits the current GPU, VRAM, and time budget: <rationale>

## Next Set Rule

- Design the next set only after Set 1 finishes and the results are reviewed.

## Decision Rules

- Promote if: <condition>
- Reject if: <condition>
- Pivot if: <condition>

## Immediate Next Action

- <the first concrete thing the AI should do after writing this plan>

## Deliverables

- <deliverable 1>
- <deliverable 2>
```

## Campaign File

Path:

```text
goals/<goal-slug>/campaigns/c01_<name>.md
```

Template:

```md
# Campaign: <name>

## Thesis

<What this campaign is trying to prove or disprove.>

## Why Now

<Why this is the right campaign at the current frontier.>

## Open Questions

- <question 1>
- <question 2>

## Exit Conditions

- Success: <condition>
- Failure: <condition>
- Pivot: <condition>
```

## Project Config

Path:

```text
projects/<project-name>.json
```

Template:

```json
{
  "name": "<project-name>",
  "repo_path": "<absolute-or-repo-relative-path>",
  "metric": "<metric-name>",
  "metric_direction": "lower",
  "target": null,
  "run_command": "<command>",
  "process_pattern": "<optional process matcher>",
  "log_path": "<path pattern>",
  "completion_indicator": "<path pattern>",
  "summary_json_path": "<optional path>",
  "stages": {
    "explore": {"steps": 500, "threshold": 0.01},
    "validate": {"steps": 4000, "threshold": 0.005},
    "full": {"steps": 0, "threshold": 0.0}
  }
}
```

## Experiment Metadata

Path:

```text
experiments/<project>/snapshots/<experiment>/meta.json
```

Template:

```json
{
  "name": "<experiment-name>",
  "project": "<project-name>",
  "hypothesis": "<what this test is trying to learn>",
  "first_principles_rationale": "<why this is the right test from mechanism, budget, and failure-mode reasoning>",
  "parent_base": "<base-id>",
  "search_strategy": "<wide_search|deep_validation|tiered_exploration>",
  "stage": "explore",
  "steps": 500,
  "baseline_metric": null,
  "promotion_threshold": 0.01,
  "expected_duration_seconds": null,
  "predicted_duration_seconds": null,
  "predicted_startup_and_initial_validation_seconds": null,
  "predicted_final_validation_seconds": null,
  "predicted_post_train_overhead_seconds": null,
  "prediction_source": null,
  "prediction_sample_count": 0,
  "remaining_budget_seconds": null,
  "actual_runtime_seconds": null,
  "prediction_error_seconds": null,
  "prediction_error_ratio": null,
  "changes_summary": "<what changed>"
}
```

## Handoff Note

Path:

```text
state/NOW.md
```

Template:

```md
# Current State

## Active Goal

<goal>

## Running Work

- <run name>: <status>

## Latest Findings

- <finding 1>
- <finding 2>

## Next Actions

1. <action 1>
2. <action 2>

## Blockers

- <blocker or none>
```

## Knowledge File

Path:

```text
knowledge/<project>/wins.md
```

Template:

```md
# Wins

## Current Durable Beliefs

- <belief 1>
- <belief 2>

## Evidence

- <experiment or report>: <what it showed>
```

## Cycle Report

Path:

```text
reports/<date>_<topic>.md
```

Template:

```md
# Cycle Report

## Scope

<What this cycle covered.>

## Checked

- <experiment 1>
- <experiment 2>

## Findings

- <finding 1>
- <finding 2>

## Promotions Or Rejections

- <decision 1>
- <decision 2>

## Next Wave

- <next experiment 1>
- <next experiment 2>
```
