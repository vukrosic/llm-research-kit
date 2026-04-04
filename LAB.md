# Lab Rules

This file defines authority and hard policy for the lab once it is installed into a target repo.

Execution mechanics live in `OPERATING_MODEL.md`.
The first user-facing workflow lives in `PRODUCT_SPEC.md`.

## Authority

- The human owns mission, scope, budget, and hard constraints.
- The AI owns planning and execution below the mission layer unless the user selects supervised mode.
- Human-written mission files are not rewritten without explicit approval.

## Hard Rules

- If the human gives a budget or deadline, write it into durable repo state before dispatching work.
- Time-boxed work must start from a calibration run or a documented calibration source.
- Track predicted versus actual runtime after every run.
- Recalibrate when runtime drift appears.
- Design one active set at a time.
- Do not pre-design future sets as if earlier results are already known.
- Do not start work that no longer fits the remaining budget with explicit margin.
- Do not spend compute without a named hypothesis and comparison target.
- Do not evaluate an experiment against a mismatched baseline.
- Every session must end with an updated handoff note.

## Source Of Truth

The lab must be file-based.

Primary records live in the target repo:

- goals and plans
- project configs
- experiment records
- knowledge files
- reports and handoff notes

Derived dashboards are useful, but primary records win when they disagree.

## Required Discipline

Every experiment must:

- have a name
- have a stated hypothesis
- record its parent base or baseline
- be judged against the correct same-step baseline
- leave behind durable results

Every adjudication must:

- record the verdict
- update knowledge
- update the handoff state

## Status Vocabulary

Use this status model for experiment records:

- `pending`
- `running`
- `done`
- `failed`
- `rejected`
- `stale_winner`
- `validated_winner`
- `promoted`
- `rollback_invalidated`

## Promotion Rule

An experiment is promotable only when:

- the result is valid
- the metric beats the required threshold
- the comparison used the correct same-step baseline
- the experiment was validated on the latest base

If a strong result was produced on an older base, mark it `stale_winner` and re-run it on the latest base before promotion.

## Rollback Rule

If a promoted result is later found invalid because of bad evaluation, broken logging, stale comparison, or implementation error:

- mark it `rollback_invalidated`
- restore the previous valid base
- record the failure mode in knowledge and the handoff note

## Operating Modes

Autonomous mode:

- the AI may plan, dispatch, adjudicate, and update files without routine approval

Supervised mode:

- the AI must stop at the agreed approval gates before dispatch, promotion, broad repo changes, or major pivots

## Minimal Knowledge Requirement

Do not let findings remain only in chat or commit history.

At minimum, maintain:

- wins
- failures
- training notes
- architecture or design notes

## Policy Rule

If a rule matters in practice, it should exist in durable repo files before the lab relies on it.
