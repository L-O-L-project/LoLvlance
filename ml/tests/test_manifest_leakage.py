from __future__ import annotations

import unittest
from pathlib import Path

from ml.tests.integrity_utils import derive_weak_collision_key, load_manifest_entries, split_manifest_entries


class TestManifestLeakage(unittest.TestCase):
    def test_manifest_exists(self) -> None:
        entries = load_manifest_entries()
        self.assertGreater(len(entries), 0, "Manifest is empty. Leakage validation cannot run.")

    def test_no_track_group_overlap_between_train_and_validation(self) -> None:
        split_entries = split_manifest_entries(load_manifest_entries())
        train_track_groups = {entry["track_group_id"] for entry in split_entries.get("train", [])}
        val_track_groups = {entry["track_group_id"] for entry in split_entries.get("val", [])}
        overlap = sorted(train_track_groups & val_track_groups)

        self.assertFalse(
            overlap,
            "Track-group leakage detected between train and val splits: "
            + ", ".join(overlap[:20]),
        )

    def test_no_duplicate_audio_paths_between_train_and_validation(self) -> None:
        split_entries = split_manifest_entries(load_manifest_entries())
        train_paths = {Path(entry["audio_path"]).as_posix() for entry in split_entries.get("train", [])}
        val_paths = {Path(entry["audio_path"]).as_posix() for entry in split_entries.get("val", [])}
        overlap = sorted(train_paths & val_paths)

        self.assertFalse(
            overlap,
            "Duplicate audio files appear in both train and val splits: "
            + ", ".join(overlap[:20]),
        )

    def test_no_duplicate_clip_ids_between_train_and_validation(self) -> None:
        split_entries = split_manifest_entries(load_manifest_entries())
        train_clip_ids = {entry["clip_id"] for entry in split_entries.get("train", [])}
        val_clip_ids = {entry["clip_id"] for entry in split_entries.get("val", [])}
        overlap = sorted(train_clip_ids & val_clip_ids)

        self.assertFalse(
            overlap,
            "Duplicate clip_id values appear in both train and val splits: "
            + ", ".join(overlap[:20]),
        )

    def test_no_weak_naming_collisions_between_train_and_validation(self) -> None:
        split_entries = split_manifest_entries(load_manifest_entries())
        train_collision_keys = {
            derive_weak_collision_key(entry): Path(entry["audio_path"]).name
            for entry in split_entries.get("train", [])
        }
        val_collision_keys = {
            derive_weak_collision_key(entry): Path(entry["audio_path"]).name
            for entry in split_entries.get("val", [])
        }
        overlap = sorted(set(train_collision_keys) & set(val_collision_keys))

        self.assertFalse(
            overlap,
            "Weak naming collisions suggest potential split leakage between train and val: "
            + ", ".join(
                f"{key} ({train_collision_keys[key]} vs {val_collision_keys[key]})"
                for key in overlap[:20]
            ),
        )


if __name__ == "__main__":
    unittest.main()
