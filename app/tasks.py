from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

from app.models import GenerationRequest, TaskRecord, utc_now_iso
from app.workflow import CapabilityFactoryService


class TaskManager:
    def __init__(self, service: CapabilityFactoryService, max_workers: int = 2):
        self.service = service
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._tasks: dict[str, TaskRecord] = {}
        self._lock = threading.Lock()

    def submit(self, request: GenerationRequest) -> TaskRecord:
        task_id = self.service.make_task_id()
        record = TaskRecord(
            task_id=task_id,
            status="queued",
            request=request,
            provider_name=self.service.provider_name,
        )
        with self._lock:
            self._tasks[task_id] = record
        self._executor.submit(self._run_task, task_id, request)
        return record

    def get(self, task_id: str) -> TaskRecord | None:
        with self._lock:
            return self._tasks.get(task_id)

    def _run_task(self, task_id: str, request: GenerationRequest) -> None:
        with self._lock:
            queued_record = self._tasks[task_id]
            self._tasks[task_id] = queued_record.model_copy(
                update={"status": "running", "updated_at": utc_now_iso()}
            )
        try:
            result = self.service.run_task(request=request, task_id=task_id)
        except Exception as exc:  # pragma: no cover - defensive API handling
            failed_record = TaskRecord(
                task_id=task_id,
                status="failed",
                request=request,
                provider_name=self.service.provider_name,
                error_message=str(exc),
            )
            failed_record.report_paths = self.service.report_writer.write(failed_record)
            with self._lock:
                self._tasks[task_id] = failed_record
            return

        with self._lock:
            self._tasks[task_id] = result
