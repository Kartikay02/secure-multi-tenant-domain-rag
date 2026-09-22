"""Evaluation run storage abstractions and file/memory implementations."""

import asyncio
import tempfile
from pathlib import Path

from app.rag.evaluation.domain import EvaluationRun, EvaluationRunSummary
from app.rag.evaluation.interfaces import EvaluationStoreProtocol


class InMemoryEvaluationStore(EvaluationStoreProtocol):
    """In-memory storage for evaluation runs suitable for unit testing."""

    def __init__(self) -> None:
        self._runs: dict[str, EvaluationRun] = {}

    async def save_run(self, run: EvaluationRun) -> str:
        self._runs[run.run_id] = run
        return run.run_id

    async def get_run(self, run_id: str) -> EvaluationRun | None:
        return self._runs.get(run_id)

    async def list_runs(self) -> list[EvaluationRunSummary]:
        summaries = [
            EvaluationRunSummary(
                run_id=r.run_id,
                run_name=r.run_name,
                dataset_name=r.dataset_name,
                dataset_version=r.dataset_version,
                created_at=r.started_at,
                sample_count=r.sample_count,
                failure_rate=r.failure_rate,
                mean_latency_ms=r.latency_summary.mean_ms,
                key_metrics={k: v.mean for k, v in r.aggregate_metrics.items()},
            )
            for r in self._runs.values()
        ]
        return sorted(summaries, key=lambda s: s.created_at, reverse=True)

    async def delete_run(self, run_id: str) -> bool:
        if run_id in self._runs:
            del self._runs[run_id]
            return True
        return False


class FileEvaluationStore(EvaluationStoreProtocol):
    """Filesystem-based persistent storage for evaluation runs as JSON files."""

    def __init__(self, base_directory: Path | str = "data/evaluations") -> None:
        self._base_dir = Path(base_directory)
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, run_id: str) -> Path:
        return self._base_dir / f"{run_id}.json"

    async def save_run(self, run: EvaluationRun) -> str:
        target_path = self._get_path(run.run_id)

        def _sync_save() -> None:
            with tempfile.NamedTemporaryFile(
                mode="w",
                dir=self._base_dir,
                delete=False,
                encoding="utf-8",
            ) as tmp:
                tmp.write(run.to_json(indent=2))
                tmp_path = Path(tmp.name)
            tmp_path.replace(target_path)

        await asyncio.to_thread(_sync_save)
        return run.run_id

    async def get_run(self, run_id: str) -> EvaluationRun | None:
        target_path = self._get_path(run_id)

        def _sync_get() -> EvaluationRun | None:
            if not target_path.exists():
                return None
            content = target_path.read_text(encoding="utf-8")
            return EvaluationRun.from_json(content)

        return await asyncio.to_thread(_sync_get)

    async def list_runs(self) -> list[EvaluationRunSummary]:
        def _sync_list() -> list[EvaluationRunSummary]:
            summaries: list[EvaluationRunSummary] = []
            for path in self._base_dir.glob("*.json"):
                try:
                    content = path.read_text(encoding="utf-8")
                    run = EvaluationRun.from_json(content)
                    summaries.append(
                        EvaluationRunSummary(
                            run_id=run.run_id,
                            run_name=run.run_name,
                            dataset_name=run.dataset_name,
                            dataset_version=run.dataset_version,
                            created_at=run.started_at,
                            sample_count=run.sample_count,
                            failure_rate=run.failure_rate,
                            mean_latency_ms=run.latency_summary.mean_ms,
                            key_metrics={k: v.mean for k, v in run.aggregate_metrics.items()},
                        )
                    )
                except Exception:
                    continue

            return sorted(summaries, key=lambda s: s.created_at, reverse=True)

        return await asyncio.to_thread(_sync_list)

    async def delete_run(self, run_id: str) -> bool:
        target_path = self._get_path(run_id)

        def _sync_delete() -> bool:
            if target_path.exists():
                target_path.unlink()
                return True
            return False

        return await asyncio.to_thread(_sync_delete)
