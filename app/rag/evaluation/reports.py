"""Report generator formatting evaluation runs and comparisons into Markdown summaries."""

from app.rag.evaluation.domain import EvaluationRun, MetricCategory, RunComparisonResult


class ReportGenerator:
    """Generates human-readable Markdown summary reports for evaluation runs and comparisons."""

    @classmethod
    def generate_run_report(cls, run: EvaluationRun) -> str:
        """Generate a complete Markdown report for an individual evaluation run."""
        lines: list[str] = []

        lines.append(f"# RAG Evaluation Report: {run.run_name}")
        lines.append("")

        if run.metadata.get("benchmark_type") == "synthetic_offline":
            lines.append("> [!NOTE]")
            lines.append(
                "> **SYNTHETIC OFFLINE BENCHMARK NOTICE**: This benchmark was executed using an offline synthetic"
            )
            lines.append(
                "> golden dataset and simulated rule-based/mock LLM judges (`MockLLMJudge`). Reported scores and"
            )
            lines.append(
                "> latencies reflect local CPU memory retriever execution and do NOT represent live production LLM performance."
            )
            lines.append("")

        lines.append("## Executive Summary")
        lines.append(f"- **Run ID**: `{run.run_id}`")
        lines.append(f"- **Dataset**: `{run.dataset_name}` (v{run.dataset_version})")
        lines.append(
            f"- **Execution Period**: {run.started_at.isoformat()} to {run.completed_at.isoformat()}"
        )
        lines.append(f"- **Duration**: {run.duration_seconds:.2f}s")
        lines.append(f"- **Total Samples**: {run.sample_count}")
        lines.append(f"- **Success Count**: {run.success_count}")
        lines.append(
            f"- **Failure Count**: {run.failure_count} (Failure Rate: {run.failure_rate * 100:.1f}%)"
        )
        lines.append("")

        # Metrics Table
        lines.append("## Benchmark Metrics")
        lines.append(
            "| Metric | Category | Mean | Median | Min | Max | p95 | Limitations & Blind Spots |"
        )
        lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

        for m_name, stat in sorted(run.aggregate_metrics.items()):
            cat_label = "Automated" if stat.category == MetricCategory.AUTOMATED else "LLM Judge"
            limitations_short = (
                stat.limitations.replace("\n", " ") if stat.limitations else "None specified."
            )
            lines.append(
                f"| **{m_name}** | {cat_label} | {stat.mean:.3f} | {stat.median:.3f} | "
                f"{stat.min:.3f} | {stat.max:.3f} | {stat.p95:.3f} | {limitations_short} |"
            )
        lines.append("")

        # Latency Summary Table
        lines.append("## Latency Breakdown")
        ls = run.latency_summary
        lines.append("| Metric | Value (ms) |")
        lines.append("| :--- | :--- |")
        lines.append(f"| **Mean Latency** | {ls.mean_ms:.2f} ms |")
        lines.append(f"| **Median / p50** | {ls.p50_ms:.2f} ms |")
        lines.append(f"| **p90 Latency** | {ls.p90_ms:.2f} ms |")
        lines.append(f"| **p95 Latency** | {ls.p95_ms:.2f} ms |")
        lines.append(f"| **p99 Latency** | {ls.p99_ms:.2f} ms |")
        lines.append(f"| **Min / Max** | {ls.min_ms:.2f} ms / {ls.max_ms:.2f} ms |")
        lines.append("")

        if ls.stage_averages:
            lines.append("### Pipeline Stage Averages")
            lines.append("| Stage | Mean Duration (ms) |")
            lines.append("| :--- | :--- |")
            for stage, ms in sorted(ls.stage_averages.items()):
                lines.append(f"| `{stage}` | {ms:.2f} ms |")
            lines.append("")

        # Failures section if applicable
        failures = [s for s in run.sample_results if s.error]
        if failures:
            lines.append("## Failed Queries")
            lines.append("| Sample ID | Question | Error |")
            lines.append("| :--- | :--- | :--- |")
            for f in failures:
                q_snippet = f.question[:60] + "..." if len(f.question) > 60 else f.question
                lines.append(f"| `{f.sample_id[:8]}` | {q_snippet} | `{f.error}` |")
            lines.append("")

        return "\n".join(lines)

    @classmethod
    def generate_comparison_report(cls, comparison: RunComparisonResult) -> str:
        """Generate a complete Markdown comparison diff report between two runs."""
        lines: list[str] = []

        lines.append("# RAG Evaluation Comparison Report")
        lines.append("")
        lines.append("## Overview")
        lines.append(f"- **Baseline Run ID**: `{comparison.baseline_run_id}`")
        lines.append(f"- **Candidate Run ID**: `{comparison.candidate_run_id}`")
        lines.append(f"- **Dataset Evaluated**: `{comparison.dataset_name}`")
        lines.append(f"- **Mean Latency Delta**: {comparison.mean_latency_delta_ms:+.2f} ms")
        lines.append(f"- **p95 Latency Delta**: {comparison.p95_latency_delta_ms:+.2f} ms")
        lines.append(f"- **Failure Rate Delta**: {comparison.failure_rate_delta * 100:+.1f}%")
        lines.append("")

        # Metric Deltas Table
        lines.append("## Metric Performance Comparison")
        lines.append("| Metric | Category | Baseline | Candidate | Delta (Δ) | % Change | Status |")
        lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

        status_icons = {
            "IMPROVED": "🟢 IMPROVED",
            "REGRESSED": "🔴 REGRESSED",
            "UNCHANGED": "⚪ UNCHANGED",
        }

        for m_name, delta in sorted(comparison.metric_deltas.items()):
            cat_label = "Automated" if delta.category == MetricCategory.AUTOMATED else "LLM Judge"
            icon = status_icons.get(delta.status, delta.status)
            lines.append(
                f"| **{m_name}** | {cat_label} | {delta.baseline_mean:.3f} | {delta.candidate_mean:.3f} | "
                f"{delta.delta:+.3f} | {delta.percent_change:+.1f}% | {icon} |"
            )
        lines.append("")

        # Sample Regressions
        if comparison.regressions:
            lines.append("## Detected Sample Regressions")
            lines.append("| Sample ID | Question Snippet | Metric | Baseline | Candidate | Delta |")
            lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
            for reg in comparison.regressions[:20]:
                q_snippet = reg.question[:60] + "..." if len(reg.question) > 60 else reg.question
                lines.append(
                    f"| `{reg.sample_id[:8]}` | {q_snippet} | {reg.metric_name} | "
                    f"{reg.baseline_score:.3f} | {reg.candidate_score:.3f} | {reg.delta:+.3f} |"
                )
            lines.append("")

        # Sample Improvements
        if comparison.improvements:
            lines.append("## Detected Sample Improvements")
            lines.append("| Sample ID | Question Snippet | Metric | Baseline | Candidate | Delta |")
            lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
            for imp in comparison.improvements[:20]:
                q_snippet = imp.question[:60] + "..." if len(imp.question) > 60 else imp.question
                lines.append(
                    f"| `{imp.sample_id[:8]}` | {q_snippet} | {imp.metric_name} | "
                    f"{imp.baseline_score:.3f} | {imp.candidate_score:.3f} | {imp.delta:+.3f} |"
                )
            lines.append("")

        return "\n".join(lines)
