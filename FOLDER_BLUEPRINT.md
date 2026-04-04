# Folder Blueprint

This kit does not ship any runtime folders. The AI should create them inside the target repo as needed.

## Recommended Structure

```text
<target-repo>/
  AGENTS.md
  LAB.md
  OPERATING_MODEL.md
  FOLDER_BLUEPRINT.md
  TEMPLATES.md
  SETUP.md

  goals/
    ACTIVE.md
    <goal-slug>/
      MISSION.md
      goal.json
      queue.json
      plans/
      campaigns/
      progress.md
      resources.md
      reports/

  projects/
    <project>.json

  experiments/
    <project>/
      base/
      snapshots/
      current_best.json
      base_id.txt

  knowledge/
    <project>/
      wins.md
      failures.md
      training.md
      architecture.md

  state/
    NOW.md
    FRONTIER.md
    ACTIVE_RUNS.md
    ADJUDICATION_QUEUE.md

  reports/
    <date>_<topic>.md

  logs/
    ...
```

## Ownership

Human-owned:

- `goals/<goal>/MISSION.md`
- budget and policy changes

AI-owned:

- plans
- campaigns
- progress tracking
- reports
- knowledge files
- state dashboards
- experiment records

Mixed ownership:

- `projects/<project>.json`
  - humans may define the initial runtime contract
  - AI may extend it when the runtime contract becomes clearer

## Creation Order

On first install, the AI should usually create files in this order:

1. `goals/ACTIVE.md`
2. first `goals/<goal>/MISSION.md`
3. first goal plan files
4. `projects/<project>.json`
5. `knowledge/<project>/` files
6. `state/NOW.md`
7. `experiments/<project>/` skeleton

## Notes By Folder

`goals/`

- long-horizon intent and planning
- the stable place for mission and campaign history

`projects/`

- machine-readable run contract
- metric, launch command, parsing logic, stage budgets, and environments

`experiments/`

- base snapshot plus experiment snapshots
- the authoritative experiment ledger

`knowledge/`

- cumulative findings organized by topic instead of by date

`state/`

- derived operational dashboards and handoff files

`reports/`

- cycle reports, retrospectives, and sprint summaries

`logs/`

- machine-local runtime output
- often useful but usually not the best source of truth

## Optional Variants

You can collapse or expand this structure if the target repo is simpler or more complex. The important property is not the exact folder names. The important property is that plans, experiments, knowledge, and handoff state are explicit and durable.
