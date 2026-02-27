"""
Live2D model discovery and model_dict management.

Provides auto-scan of live2d-models directory for .model3.json files,
merge with model_dict.json, and automatic addition of discovered models.
"""

import json
import os
from pathlib import Path

from loguru import logger

LIVE2D_DIR = "live2d-models"
MODEL_DICT_PATH = "model_dict.json"

DEFAULT_EMOTION_MAP = {
    "neutral": 0,
    "anger": 2,
    "disgust": 2,
    "fear": 1,
    "joy": 3,
    "smirk": 3,
    "sadness": 1,
    "surprise": 3,
}

DEFAULT_TAP_MOTIONS = {
    "HitAreaHead": {"": 1},
    "HitAreaBody": {"": 1},
}


def scan_live2d_models_dir(live2d_dir: str = LIVE2D_DIR) -> list[dict]:
    """
    Scan live2d-models directory for .model3.json files.

    Supports structures: folder/model.model3.json, folder/runtime/model.model3.json,
    or any nested path. Uses folder name as model name.

    Args:
        live2d_dir: Path to live2d-models directory.

    Returns:
        List of dicts: name (folder), url (path like /live2d-models/...),
        model_path (local path), avatar (optional).
    """
    if not os.path.isdir(live2d_dir):
        return []

    discovered: list[dict] = []
    supported_avatar_exts = (".png", ".jpg", ".jpeg")

    for entry in os.scandir(live2d_dir):
        if not entry.is_dir():
            continue

        folder_name = entry.name.replace("\\", "/")
        found_models: list[tuple[str, str]] = []

        for root, _, files in os.walk(entry.path):
            for f in files:
                if f.endswith(".model3.json"):
                    abs_path = os.path.join(root, f)
                    rel_path = os.path.relpath(abs_path, live2d_dir)
                    url_path = "/" + os.path.join(live2d_dir, rel_path).replace(
                        "\\", "/"
                    )
                    found_models.append((abs_path, url_path))

        if not found_models:
            continue

        for abs_path, url_path in found_models:
            stem = Path(abs_path).stem.replace(".model3", "")
            model_name = stem if stem else folder_name

            avatar_path = None
            for ext in supported_avatar_exts:
                candidate = os.path.join(os.path.dirname(abs_path), f"{stem}{ext}")
                if os.path.isfile(candidate):
                    rel = os.path.relpath(candidate, live2d_dir)
                    avatar_path = f"/{live2d_dir}/{rel}".replace("\\", "/")
                    break
            if not avatar_path:
                for ext in supported_avatar_exts:
                    candidate = os.path.join(entry.path, f"{folder_name}{ext}")
                    if os.path.isfile(candidate):
                        rel = os.path.relpath(candidate, live2d_dir)
                        avatar_path = f"/{live2d_dir}/{rel}".replace("\\", "/")
                        break

            discovered.append(
                {
                    "name": model_name,
                    "url": url_path,
                    "model_path": abs_path,
                    "avatar": avatar_path,
                    "folder": folder_name,
                }
            )

    return discovered


def load_model_dict(path: str = MODEL_DICT_PATH) -> list[dict]:
    """
    Load model_dict.json.

    Returns:
        List of model entries, or empty list if file missing or invalid.
    """
    if not os.path.isfile(path):
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Could not load {path}: {e}")
        return []

    if not isinstance(data, list):
        return []

    return data


def create_default_model_entry(name: str, url: str) -> dict:
    """
    Create a minimal model_dict entry for an auto-discovered model.

    Args:
        name: Model name.
        url: URL path (e.g. /live2d-models/foo/runtime/foo.model3.json).

    Returns:
        Model entry dict compatible with model_dict.json.
    """
    return {
        "name": name,
        "description": "",
        "url": url,
        "kScale": 0.5,
        "initialXshift": 0,
        "initialYshift": 0,
        "kXOffset": 1150,
        "idleMotionGroupName": "Idle",
        "emotionMap": DEFAULT_EMOTION_MAP.copy(),
        "tapMotions": {k: v.copy() for k, v in DEFAULT_TAP_MOTIONS.items()},
    }


def merge_model_lists(
    model_dict_list: list[dict],
    scanned_list: list[dict],
) -> list[dict]:
    """
    Merge model_dict entries with scanned discoveries.

    Only models that exist on disk (in scanned_list) are returned.
    model_dict entries for deleted folders are excluded.
    Prefer model_dict for full info. Add scanned models not in model_dict
    with default config.

    Args:
        model_dict_list: Models from model_dict.json.
        scanned_list: Models from scan_live2d_models_dir (source of truth for disk).

    Returns:
        Merged list of model info dicts (model_info format for frontend).
    """
    scanned_by_name = {s.get("name"): s for s in scanned_list if s.get("name")}
    dict_by_name = {m.get("name"): m for m in model_dict_list if m.get("name")}
    result: list[dict] = []

    for name, s in scanned_by_name.items():
        if name in dict_by_name:
            entry = dict(dict_by_name[name])
            if "url" not in entry or not entry["url"]:
                entry["url"] = s.get("url", f"/live2d-models/{name}/unknown.model3.json")
            result.append(entry)
        else:
            entry = create_default_model_entry(
                name=name,
                url=s.get("url", f"/live2d-models/{name}/unknown.model3.json"),
            )
            result.append(entry)

    return result


def get_merged_model_list(
    live2d_dir: str = LIVE2D_DIR,
    model_dict_path: str = MODEL_DICT_PATH,
) -> list[dict]:
    """
    Get full list of available Live2D models (model_dict + scanned).

    Returns:
        List of model info dicts ready for API/frontend.
    """
    model_dict_list = load_model_dict(model_dict_path)
    scanned = scan_live2d_models_dir(live2d_dir)
    return merge_model_lists(model_dict_list, scanned)


def add_model_to_dict(
    model_entry: dict,
    model_dict_path: str = MODEL_DICT_PATH,
) -> bool:
    """
    Append a model entry to model_dict.json if not already present.

    Args:
        model_entry: Full model info dict (from create_default_model_entry or model_dict format).
        model_dict_path: Path to model_dict.json.

    Returns:
        True if added, False if already present or write failed.
    """
    current = load_model_dict(model_dict_path)
    name = model_entry.get("name")
    if not name:
        return False

    if any(m.get("name") == name for m in current):
        return False

    current.append(model_entry)

    try:
        Path(model_dict_path).parent.mkdir(parents=True, exist_ok=True)
        with open(model_dict_path, "w", encoding="utf-8") as f:
            json.dump(current, f, indent=2, ensure_ascii=False)
        logger.info(f"Added model {name} to {model_dict_path}")
        return True
    except OSError as e:
        logger.error(f"Failed to write {model_dict_path}: {e}")
        return False
