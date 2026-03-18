# Goal

- Train a stable ~1B dense GPT-style model with `OneBConfig`.
- Use it as the first step toward a GPT-3-class training stack.
- Keep the first phase simple: make 1B training work reliably before chasing larger scale.

## Optimization Program

- Treat `optimization/` as a running research log, not just notes.
- For each knob:
  - define one sharp research question,
  - define the cheapest proxy experiment that could answer it,
  - define the transfer check to longer training,
  - record the decision and the next question.
- Start with learning rate because it most strongly affects both stability and final quality.
