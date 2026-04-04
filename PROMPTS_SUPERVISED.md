# Supervised Prompts

Use these prompts when the AI should do the work but stop at approval gates.

## Bootstrap In Supervised Mode

```text
Read AGENTS.md, LAB.md, OPERATING_MODEL.md, PRODUCT_SPEC.md, INTAKE_PROMPT.md, PLANNING_PROMPT.md, FOLDER_BLUEPRINT.md, TEMPLATES.md, and SETUP.md.

Install this file-based research lab into the current repo, but operate in supervised mode.

Create the missing folders and starter files.
Ask for approval before:
- dispatching experiments
- promoting a new base
- making broad code changes outside a scoped experiment
- changing plans above the current week level
```

## Propose The Next Wave

```text
Read the current lab state and propose the next research wave.

I want:
- a short summary of what the lab currently believes
- the next experiments you would run from first principles
- whether this should be wide search, deep validation, or tiered exploration
- the baseline each one compares against
- the expected cost and upside

Do not dispatch anything until I approve.
```

## Review Completed Results

```text
Read the latest completed experiments and adjudicate them against the correct same-step baselines.

Return:
- the verdict for each experiment
- any promotion candidates
- the knowledge updates that should be written
- the next best experiments to queue

Do not promote or dispatch until I approve.
```

## Weekly Planning

```text
Read the mission, current quarter plan, month plan, current knowledge, and recent results.

Draft the next week plan and campaign updates.
Keep it concrete, compute-aware, and tied to the current bottleneck.
Wait for approval before changing the active week plan.
```
