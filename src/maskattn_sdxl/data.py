"""COCO caption preparation and a minimal image-caption dataset for MaskAttn-SDXL training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def _require_spacy(model_name: str):
    try:
        import spacy
    except ImportError as exc:
        raise ImportError(
            "COCO noun-phrase filtering requires spaCy. Install `spacy` and its English model, e.g. "
            "`python -m spacy download en_core_web_sm`."
        ) from exc
    try:
        return spacy.load(model_name, disable=["ner", "textcat"])
    except OSError as exc:
        raise OSError(
            f"spaCy model `{model_name}` is unavailable. Install it with `python -m spacy download {model_name}`."
        ) from exc


def build_coco_caption_cache(
    annotations_path: str | Path,
    images_dir: str | Path,
    output_path: str | Path,
    *,
    min_noun_phrases: int = 2,
    spacy_model: str = "en_core_web_sm",
    limit: int | None = None,
) -> int:
    """Create JSONL examples with captions containing at least ``min_noun_phrases`` parsed noun chunks."""
    annotation_path = Path(annotations_path)
    image_root = Path(images_dir)
    if not annotation_path.is_file():
        raise FileNotFoundError(f"COCO caption annotations not found: {annotation_path}")
    if not image_root.is_dir():
        raise FileNotFoundError(f"COCO image directory not found: {image_root}")
    if min_noun_phrases < 1:
        raise ValueError("min_noun_phrases must be at least one")
    nlp = _require_spacy(spacy_model)
    with annotation_path.open("r", encoding="utf-8") as handle:
        source = json.load(handle)
    names = {image["id"]: image["file_name"] for image in source["images"]}
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    retained = 0
    with output.open("w", encoding="utf-8") as handle:
        for annotation in source["annotations"]:
            caption = annotation["caption"].strip()
            doc = nlp(caption)
            noun_phrases = [chunk.text for chunk in doc.noun_chunks]
            if len(noun_phrases) < min_noun_phrases:
                continue
            image_name = names.get(annotation["image_id"])
            if image_name is None:
                continue
            record = {
                "image_id": annotation["image_id"],
                "image_path": str(image_root / image_name),
                "caption": caption,
                "noun_phrases": noun_phrases,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            retained += 1
            if limit is not None and retained >= limit:
                break
    return retained


def read_jsonl(path: str | Path, *, limit: int | None = None) -> list[dict]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"JSONL file not found: {source}")
    records: list[dict] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {source}:{line_number}") from exc
            records.append(record)
            if limit is not None and len(records) >= limit:
                break
    if not records:
        raise ValueError(f"No records found in {source}")
    return records


class ImageCaptionDataset:
    def __init__(self, records: Iterable[dict], resolution: int):
        from PIL import Image
        from torch.utils.data import Dataset
        from torchvision import transforms

        self._dataset_base = Dataset
        self.records = list(records)
        self.image_type = Image.Image
        self.transform = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        from PIL import Image

        record = self.records[index]
        image_path = Path(record["image_path"])
        if not image_path.is_file():
            raise FileNotFoundError(f"Image referenced by caption cache is missing: {image_path}")
        with Image.open(image_path) as image:
            pixel_values = self.transform(image.convert("RGB"))
        return {"pixel_values": pixel_values, "caption": record["caption"], "image_id": record.get("image_id")}


def collate_captions(examples: list[dict]) -> dict:
    import torch

    return {
        "pixel_values": torch.stack([example["pixel_values"] for example in examples]),
        "captions": [example["caption"] for example in examples],
        "image_ids": [example.get("image_id") for example in examples],
    }
