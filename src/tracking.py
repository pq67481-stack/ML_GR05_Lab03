import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .preprocess import ROOT


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def latest_run() -> dict:
    path = ROOT / "progress_results" / "latest.json"
    return json.loads(path.read_text(encoding="utf-8"))


class Run:
    def __init__(self) -> None:
        now = datetime.now(timezone.utc)
        self.id = f"{now:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
        self.paths = {
            name: ROOT / relative / self.id
            for name, relative in {
                "data": "data/processed",
                "reports": "reports",
                "model": "models",
                "images": "docs/images",
            }.items()
        }
        for path in self.paths.values():
            path.mkdir(parents=True)
        (ROOT / "logs").mkdir(exist_ok=True)
        (ROOT / "progress_results").mkdir(exist_ok=True)
        self.history = []

    def update(self, stage: str, message: str, status: str = "running") -> dict:
        event = {"time_utc": datetime.now(timezone.utc).isoformat(), "stage": stage, "message": message}
        self.history.append(event)
        with (ROOT / "logs" / f"{self.id}.log").open("a", encoding="utf-8") as log:
            log.write(f"{event['time_utc']} [{stage}] {message}\n")
        manifest = {
            "run_id": self.id,
            "status": status,
            "paths": {name: path.relative_to(ROOT).as_posix() for name, path in self.paths.items()},
            "history": self.history,
        }
        write_json(ROOT / "progress_results" / f"{self.id}.json", manifest)
        if status == "completed":
            write_json(ROOT / "progress_results" / "latest.json", manifest)
        return manifest
