import torch


DEVICE_CHOICES = ("auto", "cuda", "mps", "cpu")


def mps_is_available() -> bool:
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def resolve_device(requested: str = "auto") -> torch.device:
    requested = (requested or "auto").lower()
    if requested not in DEVICE_CHOICES:
        raise ValueError(f"Unknown device '{requested}'. Choose one of: {', '.join(DEVICE_CHOICES)}")

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if mps_is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available.")
    if requested == "mps" and not mps_is_available():
        raise RuntimeError("MPS was requested, but Apple Metal acceleration is not available.")

    return torch.device(requested)


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        return f"CUDA: {props.name} ({props.total_memory / 1e9:.1f} GB)"
    if device.type == "mps":
        return "MPS: Apple Metal GPU"
    return "CPU"
