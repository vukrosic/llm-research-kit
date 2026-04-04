# Setup

Use this file when installing the kit into a real project repo.

## Preconditions

You need:

- a target codebase where experiments can actually run
- an AI coding agent working inside that repo
- a primary metric with a direction
- a way to launch training or evaluation
- a place where results can be read from logs, JSON, or artifacts

## Facts To Provide Early

Provide these facts up front when the AI cannot infer them safely:

- the research objective
- the active repo or subproject
- the primary metric and whether lower or higher is better
- the training or evaluation entrypoint
- how completion is detected
- the compute environment and budget
- the available GPU and approximate VRAM
- the time budget or deadline
- whether the AI should operate autonomously or under supervision

## Installation Flow

1. Copy these markdown files into the root of the target repo.
2. Start the AI in that repo.
3. Have it read:
   - `AGENTS.md`
   - `LAB.md`
   - `OPERATING_MODEL.md`
   - `PRODUCT_SPEC.md`
   - `INTAKE_PROMPT.md`
   - `PLANNING_PROMPT.md`
   - `FOLDER_BLUEPRINT.md`
   - `TEMPLATES.md`
4. Use one of the prompt files:
   - `PROMPTS_AUTONOMOUS.md`
   - `PROMPTS_SUPERVISED.md`
5. Let the AI create the working folders and starter files described in `FOLDER_BLUEPRINT.md`.
6. Expect it to ask only the missing critical questions, then write the plan and begin.

## Expected First Output In The Target Repo

On a clean install, the AI should create:

- at least one goal mission
- one project config
- starter knowledge files
- a first year plan and current week plan
- a current handoff note
- the experiment folder skeleton

## Secret Handling

Do not store credentials in these docs.

When the target repo needs machine credentials, API keys, or SSH details, place them in local-only files that are not committed.

## Git Policy

- track policy, goals, plans, knowledge, and reports
- decide deliberately whether to track experiment artifacts
- do not commit credentials
- treat derived dashboards as disposable if they can be regenerated
