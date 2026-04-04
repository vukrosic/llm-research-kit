# Autonomous Prompts

Use these prompts after copying the kit into the target repo.

## Bootstrap The Lab

```text
Read AGENTS.md, LAB.md, OPERATING_MODEL.md, PRODUCT_SPEC.md, INTAKE_PROMPT.md, PLANNING_PROMPT.md, FOLDER_BLUEPRINT.md, TEMPLATES.md, and SETUP.md.

We are installing a file-based autonomous research lab into this repo.

Your job:
- create the missing folders and starter files described in FOLDER_BLUEPRINT.md
- use TEMPLATES.md to materialize the initial mission, project config, plans, knowledge files, and handoff note
- keep all durable state in repo files
- ask only for critical missing facts that cannot be inferred safely
- once the minimum facts are known, write the first plan and take the first concrete action immediately

Operate in autonomous mode.
Do not wait for routine approval.
Stop only for mission changes, budget changes, destructive pivots, or policy conflicts.
```

## Start Or Continue The Loop

```text
Operate as the research agent for this repo.

Read AGENTS.md, LAB.md, OPERATING_MODEL.md, the active goal files, project configs, knowledge files, and state/NOW.md if it exists.

Then:
1. reconcile state
2. check running work
3. adjudicate completed work against the correct same-step baselines
4. update knowledge and reports
5. design the next wave from first principles, starting from the current bottleneck, mechanism, GPU/time budget, and whether the right move is wide search, deep validation, or tiered exploration
6. dispatch or stage the next work
7. update state/NOW.md

Proceed autonomously unless LAB.md says you must stop.
Do not stop at analysis when the next action is clear.
```

## Time-Boxed Research Sprint

```text
Run a time-boxed autonomous research sprint in this repo.

Before dispatching anything, create or update a goal with machine-readable timing metadata, define the training window, and ensure each proposed run fits the deadline.

If the human says something like `you have 2 hours`, write that down explicitly as:
- `training_window_seconds`
- `deadline_utc`
- `safety_margin_seconds`

Start with calibration or an explicit prior calibration source before planning the main set.

For every planned run, record:
- `predicted_duration_seconds`
- `predicted_startup_and_initial_validation_seconds` if available
- `predicted_final_validation_seconds` if available
- `predicted_post_train_overhead_seconds` if available
- `prediction_source`
- `prediction_sample_count`

For the batch as a whole, record:
- total budget seconds
- elapsed seconds
- remaining budget seconds
- predicted remaining sweep seconds
- predicted run count that still fits

After each completed run, update the record with:
- `actual_runtime_seconds`
- `prediction_error_seconds`
- `prediction_error_ratio`
- updated remaining budget seconds

Do not launch the next run if the calibrated prediction says it will miss the deadline or leave no safety margin.
If observed runtime drift exceeds the current prediction materially, recalibrate before dispatching more work.
Design one active set only. Do not design the next set until the current set has finished and been read.
Design the set from first principles, not by copying a generic sweep template.
Choose explicitly between quick elimination, wider screening, and longer validation based on the actual budget.

Use the operating docs in this repo as the authority.
Keep the sprint file-based and leave a complete handoff at the end.
```

## Recovery Prompt

```text
Recover this lab from files only.

Read the operating docs, then rebuild context from goals, plans, project configs, experiment records, knowledge files, and state/NOW.md.

Identify:
- what is active
- what is blocked
- what is missing
- the next concrete action

If critical records are missing, recreate them from TEMPLATES.md and document the recovery assumptions.
```
