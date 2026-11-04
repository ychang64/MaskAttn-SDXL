"""COCO and Flickr30k caption preparation and a minimal image-caption dataset for MaskAttn-SDXL."""

from __future__ import annotations

import json
import random
import shutil
from collections import defaultdict
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


def _load_coco_annotations(annotations_path: str | Path, images_dir: str | Path) -> tuple[dict, dict]:
    annotation_path = Path(annotations_path)
    if not annotation_path.is_file():
        raise FileNotFoundError(f"COCO caption annotations not found: {annotation_path}")
    if not Path(images_dir).is_dir():
        raise FileNotFoundError(f"COCO image directory not found: {images_dir}")
    with annotation_path.open("r", encoding="utf-8") as handle:
        source = json.load(handle)
    return source, {image["id"]: image["file_name"] for image in source["images"]}


def build_coco_caption_cache(
    annotations_path: str | Path,
    images_dir: str | Path,
    output_path: str | Path,
    *,
    min_noun_phrases: int = 2,
    spacy_model: str = "en_core_web_sm",
    limit: int | None = None,
    seed: int = 42,
) -> int:
    """Create JSONL examples with captions containing at least ``min_noun_phrases`` parsed noun chunks.

    Captions are visited in seeded random order, so ``limit`` yields a random subset rather than the first
    images in the annotation file.
    """
    if min_noun_phrases < 1:
        raise ValueError("min_noun_phrases must be at least one")
    source, names = _load_coco_annotations(annotations_path, images_dir)
    annotations = [annotation for annotation in source["annotations"] if annotation["image_id"] in names]
    random.Random(seed).shuffle(annotations)
    nlp = _require_spacy(spacy_model)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    retained = 0
    captions = (annotation["caption"].strip() for annotation in annotations)
    with output.open("w", encoding="utf-8") as handle:
        for annotation, doc in zip(annotations, nlp.pipe(captions, batch_size=256)):
            noun_phrases = [chunk.text for chunk in doc.noun_chunks]
            if len(noun_phrases) < min_noun_phrases:
                continue
            record = {
                "image_id": annotation["image_id"],
                "image_path": str(Path(images_dir) / names[annotation["image_id"]]),
                "caption": doc.text,
                "noun_phrases": noun_phrases,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            retained += 1
            if limit is not None and retained >= limit:
                break
    return retained


def _write_eval_split(
    captions_by_image: dict[str, list[str]],
    images_dir: str | Path,
    output_prompts: str | Path,
    output_images: str | Path,
    *,
    count: int,
    seed: int,
) -> int:
    """Pick ``count`` distinct images with one caption each; prompts and real images stay index-aligned."""
    names = sorted(captions_by_image)
    if len(names) < count:
        raise ValueError(f"Requested {count} images but only {len(names)} are available")
    generator = random.Random(seed)
    chosen = generator.sample(names, count)
    prompts_path = Path(output_prompts)
    images_path = Path(output_images)
    prompts_path.parent.mkdir(parents=True, exist_ok=True)
    images_path.mkdir(parents=True, exist_ok=True)
    with prompts_path.open("w", encoding="utf-8") as handle:
        for index, name in enumerate(chosen):
            source = Path(images_dir) / name
            if not source.is_file():
                raise FileNotFoundError(f"Image referenced by captions is missing: {source}")
            target = images_path / f"{index:05d}{source.suffix.lower()}"
            shutil.copy2(source, target)
            caption = generator.choice(captions_by_image[name])
            handle.write(json.dumps({"prompt": caption, "image": target.name}, ensure_ascii=False) + "\n")
    return count


def build_coco_eval_split(
    annotations_path: str | Path,
    images_dir: str | Path,
    output_prompts: str | Path,
    output_images: str | Path,
    *,
    count: int = 3000,
    seed: int = 42,
) -> int:
    source, names = _load_coco_annotations(annotations_path, images_dir)
    captions_by_image: dict[str, list[str]] = defaultdict(list)
    for annotation in source["annotations"]:
        if annotation["image_id"] in names:
            captions_by_image[names[annotation["image_id"]]].append(annotation["caption"].strip())
    return _write_eval_split(captions_by_image, images_dir, output_prompts, output_images, count=count, seed=seed)


def build_flickr30k_eval_split(
    captions_path: str | Path,
    images_dir: str | Path,
    output_prompts: str | Path,
    output_images: str | Path,
    *,
    count: int = 500,
    seed: int = 42,
) -> int:
    """Read the official ``results_20130124.token`` format: ``<image>.jpg#<n><TAB><caption>``."""
    path = Path(captions_path)
    if not path.is_file():
        raise FileNotFoundError(f"Flickr30k captions not found: {path}")
    captions_by_image: dict[str, list[str]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            identifier, separator, caption = line.rstrip("\n").partition("\t")
            if not separator or "#" not in identifier:
                raise ValueError(f"Expected `<image>#<n><TAB><caption>` at {path}:{line_number}")
            captions_by_image[identifier.split("#", 1)[0]].append(caption.strip())
    return _write_eval_split(captions_by_image, images_dir, output_prompts, output_images, count=count, seed=seed)


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
        from torchvision import transforms

        self.records = list(records)
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
