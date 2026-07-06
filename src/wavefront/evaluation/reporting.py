from __future__ import annotations

from dataclasses import asdict

from wavefront.evaluation.config import EvalConfig


def create_report(
        cfg: EvalConfig,
) -> dict:
    """
    Create an empty evaluation and inference-benchmark report.

    Args:
        cfg: Shared evaluation configuration used to generate the report.

    Returns:
        Dictionary with two entries:

            cfg:
                JSON-compatible dictionary representation of the evaluation
                configuration.

            results:
                Empty dictionary to be populated with per-operator metrics.
    """
    return {
        "cfg": asdict(cfg),
        "results": {},
    }


def add_result(
        report: dict,
        operator_name: str,
        result: dict,
) -> None:
    """
    Add or replace the evaluation result for one reconstruction operator.

    Args:
        report: Benchmark report created by ``create_report``.
        operator_name: Display name and dictionary key for the evaluated
            operator, such as ``"DeepONet"``, ``"FNO"``, or ``"Poisson"``.
        result: Metrics dictionary produced by an evaluation helper. It
            typically contains ``mean_rel_l2``, ``median_rel_l2``, sample
            count, and optional inference benchmark data.

    Side Effects:
        Updates ``report["results"][operator_name]`` in place.
    """
    report["results"][operator_name] = result


def print_report(
        report: dict,
) -> None:
    """
    Print a compact, human-readable comparison of reconstruction operators.

    The summary displays the evaluation configuration followed by relative L2
    error statistics and, when available, inference throughput metrics for
    each recorded operator.

    Args:
        report: Benchmark report created by ``create_report`` and populated
            with one or more calls to ``add_result``.

    Notes:
        Missing metric values are displayed as ``nan`` rather than raising an
        exception. This allows partially completed reports to be printed for
        debugging or incremental evaluation workflows.
    """
    cfg = report.get("cfg", {})

    print("=== Fair comparison report ===")

    print(
        "mode: "
        f"{cfg.get('mode', '?')} | "
        f"n_train={cfg.get('n_train')} | "
        f"n_test={cfg.get('n_test')} | "
        f"n_eval={cfg.get('n_eval')}"
    )

    print()

    # Print one result block per evaluated reconstruction operator.
    for name, result in report.get("results", {}).items():
        print(f"--- {name} ---")

        print(
            "  mean relative L2: "
            f"{result.get('mean_rel_l2', float('nan')):.4f}"
        )

        print(
            "  median relative L2: "
            f"{result.get('median_rel_l2', float('nan')):.4f}"
        )

        # Timing metrics are optional because an evaluation may be completed
        # without running an inference benchmark.
        benchmark = result.get("bench")

        if benchmark is not None:
            print(
                "  ms per function: "
                f"{benchmark.get('ms_per_function', float('nan')):.3f}"
            )

            print(
                "  functions per second: "
                f"{benchmark.get('functions_per_s', float('nan')):.1f}"
            )

        print()
