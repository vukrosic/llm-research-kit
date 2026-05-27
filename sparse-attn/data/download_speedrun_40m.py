from pathlib import Path

from datasets import load_dataset


def main() -> None:
    out = Path("processed_data/speedrun_40M")
    if out.exists():
        print(f"Dataset already exists: {out}")
        return
    print("Downloading 40M Token Subset from vukrosic/blueberry-1B-pretrain...")
    ds = load_dataset("vukrosic/blueberry-1B-pretrain", split="train[:20000]")
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out))
    print(f"Speedrun data ready: {out}")


if __name__ == "__main__":
    main()

