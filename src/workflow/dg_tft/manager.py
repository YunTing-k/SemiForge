# -*- coding: utf-8 -*-
"""
Header information
---------
Shanghai Jiao Tong University, School of Integrated Circuits, SMIL Lab
Author: Yu Huang
Create Date: 2026.6.25
Description: WorkflowManager — thread-safe DG-TFT workflow run lifecycle management

Revision:
---------
2026.6.25      Yu Huang      1.0      Initial implementation
"""
from __future__ import annotations

import os
import json
import threading
import logging
from datetime import datetime
from pathlib import Path

from .schema import (
    WorkflowRun,
    WorkflowStatus,
    StageResult,
    DatasetProfile,
    StageName,
)
from src.constants import SESSION_PATH, WORKFLOW_RUNS_NAME

sys_log = logging.getLogger("logger")

_SESSION_PATH = SESSION_PATH


class PreparationTracker:
    """Manages the lifecycle of data preparation sessions.

    Each preparation session has a unique prep_id, a source directory,
    and an output directory under session/<uuid>/runs/run_NNNN/data-prep/attempt_N/.
    """

    def __init__(self):
        self._next_prep_id: int = 1
        self._lock = threading.Lock()
        self._preps: dict[int, dict] = {}

    def register_prep(self, raw_dataset_dir: str, output_dir: str) -> int:
        with self._lock:
            prep_id = self._next_prep_id
            self._next_prep_id += 1
            self._preps[prep_id] = {
                "prep_id": prep_id,
                "raw_dataset_dir": raw_dataset_dir,
                "output_dir": output_dir,
                "ok": False,
                "issues": [],
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            return prep_id

    def get_prep(self, prep_id: int) -> dict | None:
        with self._lock:
            return self._preps.get(prep_id)

    def update_prep(self, prep_id: int, ok: bool, issues: list[str]):
        with self._lock:
            prep = self._preps.get(prep_id)
            if prep is None:
                return
            prep["ok"] = ok
            prep["issues"] = issues

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "next_prep_id": self._next_prep_id,
                "preps": self._preps,
            }

    def from_dict(self, data: dict):
        with self._lock:
            self._next_prep_id = data.get("next_prep_id", 1)
            self._preps = data.get("preps", {})


class WorkflowManager:
    """Thread-safe DG-TFT workflow run manager.

    Manages the full lifecycle of workflow runs: creation, stage updates,
    artifact/metric attachment, status transitions, and persistence.
    """

    def __init__(self):
        self.session_uuid: str = ""
        self._lock = threading.Lock()
        self._runs: dict[int, WorkflowRun] = {}
        self._next_run_id: int = 1
        self._running_threads: dict[str, threading.Thread] = {}
        self.preparation_tracker = PreparationTracker()

    # ──────────────────── CRUD ────────────────────

    def create_run(
        self,
        dataset_dir: str,
        hysteresis_dir: str | None = None,
        output_root: str = "",
        device_name: str = "",
        prep_id: int | None = None,
    ) -> tuple[bool, int, str]:
        with self._lock:
            run_id = self._next_run_id
            self._next_run_id += 1
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            run = WorkflowRun(
                run_id=run_id,
                dataset_dir=dataset_dir,
                hysteresis_dir=hysteresis_dir,
                output_root=output_root,
                prep_id=prep_id,
                status=WorkflowStatus.PENDING,
                created_at=now,
                updated_at=now,
                extra={"device_name": device_name},
            )
            self._runs[run_id] = run
            self._save_runs()
            sys_log.info(
                "Workflow run %d created: dataset=%s output=%s",
                run_id, dataset_dir, output_root,
            )
            return True, run_id, now

    def get_run(self, run_id: int) -> WorkflowRun | None:
        with self._lock:
            return self._runs.get(run_id)

    def list_runs(self) -> list[dict]:
        with self._lock:
            result = []
            for run in self._runs.values():
                result.append({
                    "run_id": run.run_id,
                    "status": run.status.value,
                    "dataset_dir": run.dataset_dir,
                    "prep_id": run.prep_id,
                    "current_stage": run.current_stage,
                    "created_at": run.created_at,
                    "updated_at": run.updated_at,
                })
            return result

    # ──────────────────── Mutations ────────────────────

    def update_stage(
        self,
        run_id: int,
        stage_name: str,
        status: WorkflowStatus,
        stage_result: StageResult | None = None,
    ) -> bool:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return False
            run.current_stage = stage_name
            run.status = status
            run.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if stage_result is not None:
                run.stage_results[stage_name] = stage_result
            self._runs[run_id] = run
            self._save_runs()
            return True

    def attach_artifact(self, run_id: int, key: str, path: str) -> bool:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return False
            run.artifacts[key] = path
            run.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._runs[run_id] = run
            self._save_runs()
            return True

    def attach_metric(self, run_id: int, key: str, value: float) -> bool:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return False
            run.metrics[key] = value
            run.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._runs[run_id] = run
            self._save_runs()
            return True

    def mark_done(self, run_id: int) -> bool:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return False
            run.status = WorkflowStatus.DONE
            run.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._runs[run_id] = run
            self._save_runs()
            sys_log.info("Workflow run %d marked DONE", run_id)
            return True

    def mark_failed(self, run_id: int, reason: str = "") -> bool:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return False
            run.status = WorkflowStatus.FAILED
            if reason:
                run.errors.append(reason)
            run.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._runs[run_id] = run
            self._save_runs()
            sys_log.info("Workflow run %d marked FAILED: %s", run_id, reason)
            return True

    def cancel_run(self, run_id: int) -> bool:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return False
            run.status = WorkflowStatus.CANCELLED
            run.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._runs[run_id] = run
            self._save_runs()
            return True

    def add_warning(self, run_id: int, warning: str) -> bool:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return False
            run.warnings.append(warning)
            self._runs[run_id] = run
            self._save_runs()
            return True

    def set_dataset_profile(self, run_id: int, profile: DatasetProfile) -> bool:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return False
            run.temperature_enabled = profile.has_temperature_iv
            run.hysteresis_enabled = profile.has_hysteresis
            run.extra["dataset_profile"] = profile.to_dict()
            run.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._runs[run_id] = run
            self._save_runs()
            return True

    # ──────────────────── Persistence ────────────────────

    def _runs_file(self) -> Path:
        return Path(os.path.join(_SESSION_PATH, self.session_uuid, WORKFLOW_RUNS_NAME))

    def _save_runs(self) -> None:
        if not self.session_uuid:
            return
        file_path = self._runs_file()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "next_run_id": self._next_run_id,
            "runs": {str(k): v.to_dict() for k, v in self._runs.items()},
            "preparation_tracker": self.preparation_tracker.to_dict(),
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def save_runs(self) -> None:
        with self._lock:
            self._save_runs()

    def load_runs(self) -> None:
        if not self.session_uuid:
            return
        file_path = self._runs_file()
        if not file_path.exists():
            sys_log.info("No workflow_runs.json found, starting fresh")
            return
        try:
            with self._lock:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._next_run_id = data.get("next_run_id", 1)
                self._runs = {}
                for k, v in data.get("runs", {}).items():
                    self._runs[int(k)] = WorkflowRun.from_dict(v)
                prep_data = data.get("preparation_tracker", {})
                if prep_data:
                    self.preparation_tracker.from_dict(prep_data)
                sys_log.info(
                    "Loaded %d workflow runs, next_id=%d",
                    len(self._runs), self._next_run_id,
                )
        except Exception as e:
            sys_log.error("Failed to load workflow_runs.json: %s", e)

    # ──────────────────── Serialization ────────────────────

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "next_run_id": self._next_run_id,
                "runs": {str(k): v.to_dict() for k, v in self._runs.items()},
                "preparation_tracker": self.preparation_tracker.to_dict(),
            }

    def from_dict(self, data: dict) -> None:
        with self._lock:
            self._next_run_id = data.get("next_run_id", 1)
            self._runs = {
                int(k): WorkflowRun.from_dict(v)
                for k, v in data.get("runs", {}).items()
            }
            prep_data = data.get("preparation_tracker", {})
            if prep_data:
                self.preparation_tracker.from_dict(prep_data)
