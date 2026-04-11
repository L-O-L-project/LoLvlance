from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from audio_separator.separator import Separator
from imageio_ffmpeg import get_ffmpeg_exe

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_MODEL_FILENAME = "htdemucs_6s.yaml"
DEFAULT_MODEL_DIR = Path("ml/.cache/audio-separator-models")
DEFAULT_OUTPUT_DIR = Path("ml/.cache/audio-separator-output")

STEM_TO_SOURCE = {
    "vocals": "vocal",
    "vocal": "vocal",
    "bass": "bass",
    "drums": "drums",
    "guitar": "guitar",
    "piano": "keys",
    "keys": "keys",
}


@dataclass
class StemStats:
    stem: str
    source: str | None
    rms: float
    peak: float
    energy: float
    sample_rate: int
    frames: int


class StemSeparationService:
    def __init__(
        self,
        model_filename: str,
        model_file_dir: Path,
        output_dir: Path,
        log_level: int = logging.INFO,
        preload_model: bool = False,
    ) -> None:
        self.logger = logging.getLogger("stem-service")
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            self.logger.addHandler(handler)

        self.model_filename = model_filename
        self.model_file_dir = model_file_dir
        self.output_dir = output_dir
        self.separator: Separator | None = None
        self.model_loaded = False

        self.model_file_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if preload_model:
            self.ensure_model_loaded()

    def ensure_model_loaded(self) -> None:
        if self.separator is not None and self.model_loaded:
            return

        configure_ffmpeg_path(self.logger)
        self.logger.info("Loading source separation model: %s", self.model_filename)
        start = time.perf_counter()
        self.separator = Separator(
            log_level=self.logger.level,
            model_file_dir=self.model_file_dir.as_posix(),
            output_dir=self.output_dir.as_posix(),
            output_format="WAV",
            output_bitrate=None,
            sample_rate=44100,
            use_soundfile=True,
        )
        self.separator.load_model(model_filename=self.model_filename)
        self.model_loaded = True
        self.logger.info("Model ready in %.2fs", time.perf_counter() - start)

    def analyze_wav_bytes(self, wav_bytes: bytes) -> dict[str, Any]:
        if not wav_bytes:
            raise ValueError("Empty request body")

        self.ensure_model_loaded()

        with tempfile.TemporaryDirectory(prefix="stem-service-") as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "input.wav"
            input_path.write_bytes(wav_bytes)

            stem_files = self._separate_file(input_path, temp_path)
            stem_stats = [self._read_stem_stats(path) for path in stem_files]
            detected_sources = self._aggregate_detected_sources(stem_stats)

            return {
                "engine": "audio-separator",
                "model": self.model_filename,
                "detectedSources": detected_sources,
                "stems": [
                    {
                        "stem": stem.stem,
                        "source": stem.source,
                        "confidence": round(self._energy_ratio(stem.energy, stem_stats), 4),
                        "rms": round(stem.rms, 6),
                        "peak": round(stem.peak, 6),
                        "sampleRate": stem.sample_rate,
                        "frames": stem.frames,
                    }
                    for stem in stem_stats
                ],
                "timestampMs": int(time.time() * 1000),
            }

    def _separate_file(self, input_path: Path, temp_dir: Path) -> list[Path]:
        assert self.separator is not None

        original_output_dir = self.separator.output_dir
        self.separator.output_dir = temp_dir.as_posix()

        original_model_output_dir = None
        if getattr(self.separator, "model_instance", None) is not None:
            original_model_output_dir = self.separator.model_instance.output_dir
            self.separator.model_instance.output_dir = temp_dir.as_posix()

        try:
            output_files = self.separator.separate(input_path.as_posix())
            normalized_paths: list[Path] = []

            for path in output_files:
                output_path = Path(path)

                if not output_path.is_absolute():
                    output_path = temp_dir / output_path

                if output_path.exists():
                    normalized_paths.append(output_path)

            return normalized_paths
        finally:
            self.separator.output_dir = original_output_dir
            if getattr(self.separator, "model_instance", None) is not None and original_model_output_dir is not None:
                self.separator.model_instance.output_dir = original_model_output_dir

    def _read_stem_stats(self, path: Path) -> StemStats:
        audio, sample_rate = sf.read(path.as_posix(), always_2d=True)
        mono = np.mean(audio, axis=1)
        energy = float(np.mean(np.square(mono))) if mono.size else 0.0
        rms = float(np.sqrt(energy))
        peak = float(np.max(np.abs(mono))) if mono.size else 0.0
        stem_name = infer_stem_name(path)
        source = STEM_TO_SOURCE.get(stem_name)

        return StemStats(
            stem=stem_name,
            source=source,
            rms=rms,
            peak=peak,
            energy=energy,
            sample_rate=int(sample_rate),
            frames=int(mono.shape[0]),
        )

    def _aggregate_detected_sources(self, stem_stats: list[StemStats]) -> list[dict[str, Any]]:
        total_energy = sum(stem.energy for stem in stem_stats if stem.source) or 1e-8
        by_source: dict[str, dict[str, Any]] = {}

        for stem in stem_stats:
            if not stem.source:
                continue

            entry = by_source.setdefault(
                stem.source,
                {"source": stem.source, "confidence": 0.0, "labels": []},
            )
            entry["confidence"] += stem.energy / total_energy
            if stem.stem not in entry["labels"]:
                entry["labels"].append(stem.stem)

        detected_sources = [
            {
                "source": entry["source"],
                "confidence": round(min(1.0, entry["confidence"]), 2),
                "labels": sorted(entry["labels"]),
            }
            for entry in by_source.values()
            if entry["confidence"] >= 0.08
        ]

        detected_sources.sort(key=lambda item: item["confidence"], reverse=True)
        return detected_sources

    def _energy_ratio(self, energy: float, stem_stats: list[StemStats]) -> float:
        total_energy = sum(stem.energy for stem in stem_stats) or 1e-8
        return energy / total_energy


class StemRequestHandler(BaseHTTPRequestHandler):
    service: StemSeparationService

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/health":
            self._send_json({"error": "Not found"}, status=404)
            return

        self._send_json(
            {
                "ok": True,
                "engine": "audio-separator",
                "model": self.service.model_filename,
                "modelLoaded": self.service.model_loaded,
            }
        )

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/analyze-stems":
            self._send_json({"error": "Not found"}, status=404)
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            content_length = 0

        if content_length <= 0:
            self._send_json({"error": "Request body is empty"}, status=400)
            return

        try:
            payload = self.rfile.read(content_length)
            result = self.service.analyze_wav_bytes(payload)
            self._send_json(result)
        except Exception as error:  # pragma: no cover - keep server resilient
            self.service.logger.exception("Stem separation request failed")
            self._send_json({"error": str(error)}, status=500)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        self.service.logger.info("%s - %s", self.address_string(), format % args)

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")


def infer_stem_name(path: Path) -> str:
    match = re.search(r"\(([^)]+)\)", path.stem)
    if match:
        return match.group(1).strip().lower()

    normalized = path.stem.lower()

    for stem_name in ("vocals", "drums", "bass", "guitar", "piano", "keys", "other"):
        if stem_name in normalized:
            return stem_name

    return normalized


def configure_ffmpeg_path(logger: logging.Logger) -> None:
    ffmpeg_path = Path(get_ffmpeg_exe()).resolve()
    shim_dir = Path(tempfile.gettempdir()) / "soundfix-ffmpeg-bin"
    shim_dir.mkdir(parents=True, exist_ok=True)
    shim_ffmpeg_path = shim_dir / "ffmpeg"

    if not shim_ffmpeg_path.exists():
        shim_ffmpeg_path.symlink_to(ffmpeg_path)

    ffmpeg_dir = shim_dir.as_posix()
    current_path = os.environ.get("PATH", "")

    if ffmpeg_dir not in current_path.split(os.pathsep):
        os.environ["PATH"] = f"{ffmpeg_dir}{os.pathsep}{current_path}" if current_path else ffmpeg_dir
        logger.info("Using bundled ffmpeg from %s", ffmpeg_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a local stem separation sidecar for the SoundFix UI."
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--model", default=DEFAULT_MODEL_FILENAME)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--preload-model", action="store_true")
    parser.add_argument("--clear-output-dir", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_level = getattr(logging, str(args.log_level).upper(), logging.INFO)

    if args.clear_output_dir and args.output_dir.exists():
        shutil.rmtree(args.output_dir, ignore_errors=True)

    service = StemSeparationService(
        model_filename=args.model,
        model_file_dir=args.model_dir,
        output_dir=args.output_dir,
        log_level=log_level,
        preload_model=args.preload_model,
    )

    StemRequestHandler.service = service
    server = ThreadingHTTPServer((args.host, args.port), StemRequestHandler)

    service.logger.info(
        "Stem separation service listening on http://%s:%s using model %s",
        args.host,
        args.port,
        args.model,
    )

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        service.logger.info("Shutting down stem separation service...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
