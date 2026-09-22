"""Evaluation run comparison and regression detection engine."""

from app.rag.evaluation.domain import (
    EvaluationRun,
    MetricCategory,
    MetricDelta,
    RunComparisonResult,
    SampleRegression,
)


class RunComparator:
    """Compares baseline and candidate evaluation runs, detecting metric regressions and improvements."""

    def __init__(
        self,
        metric_tolerance: float = 0.005,
        sample_regression_threshold: float = 0.10,
    ) -> None:
        self.metric_tolerance = metric_tolerance
        self.sample_regression_threshold = sample_regression_threshold

    def compare(
        self,
        baseline: EvaluationRun,
        candidate: EvaluationRun,
    ) -> RunComparisonResult:
        """Compare a baseline run against a candidate run."""
        metric_deltas: dict[str, MetricDelta] = {}

        # Collect all metric keys across both runs
        all_metrics = set(baseline.aggregate_metrics.keys()).union(
            candidate.aggregate_metrics.keys()
        )

        for m_name in sorted(list(all_metrics)):
            base_stat = baseline.aggregate_metrics.get(m_name)
            cand_stat = candidate.aggregate_metrics.get(m_name)

            base_mean = base_stat.mean if base_stat else 0.0
            cand_mean = cand_stat.mean if cand_stat else 0.0
            category = (
                cand_stat.category
                if cand_stat
                else (base_stat.category if base_stat else MetricCategory.AUTOMATED)
            )

            delta = round(cand_mean - base_mean, 4)
            if base_mean > 0.0:
                pct_change = round((delta / base_mean) * 100.0, 2)
            else:
                pct_change = 100.0 if cand_mean > 0.0 else 0.0

            if delta > self.metric_tolerance:
                status = "IMPROVED"
            elif delta < -self.metric_tolerance:
                status = "REGRESSED"
            else:
                status = "UNCHANGED"

            metric_deltas[m_name] = MetricDelta(
                metric_name=m_name,
                category=category,
                baseline_mean=round(base_mean, 4),
                candidate_mean=round(cand_mean, 4),
                delta=delta,
                percent_change=pct_change,
                status=status,
            )

        # Latency delta
        mean_latency_delta = round(
            candidate.latency_summary.mean_ms - baseline.latency_summary.mean_ms, 2
        )
        p95_latency_delta = round(
            candidate.latency_summary.p95_ms - baseline.latency_summary.p95_ms, 2
        )

        # Failure rate delta
        failure_rate_delta = round(candidate.failure_rate - baseline.failure_rate, 4)

        # Sample-level regressions and improvements
        base_samples = {s.sample_id: s for s in baseline.sample_results}
        cand_samples = {s.sample_id: s for s in candidate.sample_results}

        regressions: list[SampleRegression] = []
        improvements: list[SampleRegression] = []

        common_ids = set(base_samples.keys()).intersection(cand_samples.keys())
        for s_id in common_ids:
            b_sample = base_samples[s_id]
            c_sample = cand_samples[s_id]

            for m_name in b_sample.metrics:
                if m_name in c_sample.metrics:
                    b_score = b_sample.metrics[m_name].score
                    c_score = c_sample.metrics[m_name].score
                    score_diff = round(c_score - b_score, 4)

                    if score_diff <= -self.sample_regression_threshold:
                        regressions.append(
                            SampleRegression(
                                sample_id=s_id,
                                question=c_sample.question,
                                metric_name=m_name,
                                baseline_score=b_score,
                                candidate_score=c_score,
                                delta=score_diff,
                            )
                        )
                    elif score_diff >= self.sample_regression_threshold:
                        improvements.append(
                            SampleRegression(
                                sample_id=s_id,
                                question=c_sample.question,
                                metric_name=m_name,
                                baseline_score=b_score,
                                candidate_score=c_score,
                                delta=score_diff,
                            )
                        )

        # Sort regressions by worst delta
        regressions.sort(key=lambda r: r.delta)
        improvements.sort(key=lambda r: r.delta, reverse=True)

        return RunComparisonResult(
            baseline_run_id=baseline.run_id,
            candidate_run_id=candidate.run_id,
            dataset_name=candidate.dataset_name,
            metric_deltas=metric_deltas,
            mean_latency_delta_ms=mean_latency_delta,
            p95_latency_delta_ms=p95_latency_delta,
            failure_rate_delta=failure_rate_delta,
            regressions=regressions,
            improvements=improvements,
            metadata={
                "baseline_run_name": baseline.run_name,
                "candidate_run_name": candidate.run_name,
                "dataset_version": candidate.dataset_version,
            },
        )
