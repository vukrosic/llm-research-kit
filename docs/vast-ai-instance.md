# Vast AI Instance

This repository can be used on the remote Vast AI box at:

```bash
ssh -L 8080:localhost:8080 -p 39751 root@171.101.230.15
```

The `-L 8080:localhost:8080` forward keeps any service you start on the remote
machine's port `8080` reachable from your local browser at `http://localhost:8080`.

## What Is Already Installed

The instance already has a working Python and CUDA stack:

- Ubuntu 24.04.4
- `/usr/bin/python3` available system-wide
- a ready-to-use environment at `/venv/main`
- PyTorch `2.11.0+cu130`
- CUDA available through PyTorch
- one NVIDIA GeForce RTX 3090

## How To Use It

Activate the environment after logging in:

```bash
source /venv/main/bin/activate
```

Then run Python normally:

```bash
python --version
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
PY
```

If you prefer not to activate the environment, use the interpreter directly:

```bash
/venv/main/bin/python --version
```

## Notes

- Do not reinstall Python, PyTorch, or CUDA.
- Use the existing `/venv/main` environment for training and experiments.
- Keep the SSH tunnel open while using forwarded services on port `8080`.
