"""Dataset loading for the probe (medical VQA).

The study runs on VQA-RAD (radiology visual question answering) pulled from the
HuggingFace Hub. A custom annotation file can also be supplied via `--data_path`.

Each sample is a dict:
    {"id", "task", "image" (path), "question", "answer", "answers" (list)}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple


def load_annotations(data_path: str | Path) -> List[Dict]:
    """Load a (custom) annotation file, mirroring the original `--data_path` flag."""
    data = json.loads(Path(data_path).read_text(encoding="utf-8"))
    return data["samples"] if isinstance(data, dict) else data


# --------------------------------------------------------------------------- #
# Real dataset from the HuggingFace Hub (requires `datasets`)
# --------------------------------------------------------------------------- #
HF_VQARAD_REPO = "flaviagiammarino/vqa-rad"


def load_hf_vqarad(
    out_dir: str | Path,
    num_samples: int = 8,
    split: str = "test",
    start_index: int = 0,
    repo_id: str = HF_VQARAD_REPO,
) -> Tuple[List[Dict], Path]:
    """Materialise VQA-RAD (radiology VQA) samples into the local sample format.

    Columns used: ``image`` (inline PIL), ``question`` (str), ``answer`` (str).
    Each image is saved to disk so samples use an ``image`` (path) convention.
    """
    from datasets import load_dataset  # imported lazily so the package stays optional

    out_dir = Path(out_dir)
    img_dir = out_dir / "images" / "vqarad"
    ann_dir = out_dir / "annotations"
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    end = start_index + num_samples
    ds = load_dataset(repo_id, split=f"{split}[:{end}]")

    samples: List[Dict] = []
    for i in range(start_index, min(end, len(ds))):
        row = ds[i]
        sid = f"vqarad_{split}_{i:04d}"
        path = img_dir / f"{sid}.png"
        row["image"].convert("RGB").save(path)
        answer = str(row["answer"])
        sample: Dict = {
            "id": sid,
            "task": "vqa_rad",
            "image": str(path),
            "question": row["question"],
            "answer": answer,
            "answers": [answer],
        }
        # pass through VQA-RAD metadata when present in the HF dataset
        for field in ("question_type", "answer_type", "image_organ"):
            if field in row and row[field] is not None:
                sample[field] = str(row[field])
        samples.append(sample)

    ann_path = ann_dir / f"vqarad_{split}.json"
    ann_path.write_text(json.dumps({"task": "vqa_rad", "samples": samples}, indent=2), encoding="utf-8")
    return samples, ann_path
