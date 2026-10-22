"""
ChemLLMBench – Main Entry Point
================================
Unified CLI to run one or more chemistry LLM benchmarks.

Quick start (run all tasks):
    python -m ChemLLMBench.main --model gpt-4 --backend api --tasks all

Run specific tasks:
    python -m ChemLLMBench.main --model gpt-4 --backend api \
        --tasks name_prediction property_prediction

Programmatic usage:
    from ChemLLMBench.main import run_all_benchmarks
    results = run_all_benchmarks(model_name="gpt-4", backend="api")
"""

import argparse
import json
import os
import datetime


# ── Registry of available benchmark tasks ────────────────────────────────────
# Each entry maps a short task key to:
#   (display_name, module_import_path, run_function_name, extra_kwargs)
# extra_kwargs are merged into the common kwargs passed to run_benchmark.

TASK_REGISTRY = {
    # ── Name Prediction (4 sub-tasks) ────────────────────────────────────
    "smiles_to_iupac": {
        "display": "Name Prediction – SMILES → IUPAC",
        "module": "Name_Prediction.name_pred",
        "function": "run_benchmark",
        "kwargs": {"task_name": "SMILES_TO_IUPAC"},
    },
    "iupac_to_smiles": {
        "display": "Name Prediction – IUPAC → SMILES",
        "module": "Name_Prediction.name_pred",
        "function": "run_benchmark",
        "kwargs": {"task_name": "IUPAC_TO_SMILES"},
    },
    "smiles_to_formula": {
        "display": "Name Prediction – SMILES → Formula",
        "module": "Name_Prediction.name_pred",
        "function": "run_benchmark",
        "kwargs": {"task_name": "SMILES_TO_FORMULA"},
    },
    "formula_to_smiles": {
        "display": "Name Prediction – Formula → SMILES",
        "module": "Name_Prediction.name_pred",
        "function": "run_benchmark",
        "kwargs": {"task_name": "FORMULA_TO_SMILES"},
    },

    # ── Property Prediction (5 datasets) ─────────────────────────────────
    "bbbp": {
        "display": "Property Prediction – BBBP",
        "module": "Property_Prediction.Property_prediction",
        "function": "run_benchmark",
        "kwargs": {"dataset_name": "BBBP"},
    },
    "bace": {
        "display": "Property Prediction – BACE",
        "module": "Property_Prediction.Property_prediction",
        "function": "run_benchmark",
        "kwargs": {"dataset_name": "BACE"},
    },
    "tox21": {
        "display": "Property Prediction – Tox21",
        "module": "Property_Prediction.Property_prediction",
        "function": "run_benchmark",
        "kwargs": {"dataset_name": "Tox21"},
    },
    "hiv": {
        "display": "Property Prediction – HIV",
        "module": "Property_Prediction.Property_prediction",
        "function": "run_benchmark",
        "kwargs": {"dataset_name": "HIV"},
    },
    "clintox": {
        "display": "Property Prediction – ClinTox",
        "module": "Property_Prediction.Property_prediction",
        "function": "run_benchmark",
        "kwargs": {"dataset_name": "ClinTox"},
    },

    # ── Reaction Prediction ──────────────────────────────────────────────
    "reaction_prediction": {
        "display": "Forward Reaction Prediction",
        "module": "Reaction_prediction.reaction_pred",
        "function": "run_benchmark",
        "kwargs": {},
    },

    # ── Retrosynthesis ───────────────────────────────────────────────────
    "retrosynthesis": {
        "display": "Retrosynthesis",
        "module": "Retrosynthesis.retro",
        "function": "run_benchmark",
        "kwargs": {},
    },

    # ── Molecule Design (2 sub-tasks) ────────────────────────────────────
    "mol_caption": {
        "display": "Molecule Caption (SMILES → Description)",
        "module": "Molecule_Design.Mol_design_and_mol_caption",
        "function": "run_benchmark",
        "kwargs": {"task_name": "MOL_CAPTION"},
    },
    "mol_design": {
        "display": "Molecule Design (Description → SMILES)",
        "module": "Molecule_Design.Mol_design_and_mol_caption",
        "function": "run_benchmark",
        "kwargs": {"task_name": "MOL_DESIGN"},
    },
}

# Convenience groups
TASK_GROUPS = {
    "all": list(TASK_REGISTRY.keys()),
    "name_prediction": [
        "smiles_to_iupac", "iupac_to_smiles",
        "smiles_to_formula", "formula_to_smiles",
    ],
    "property_prediction": ["bbbp", "bace", "tox21", "hiv", "clintox"],
    "reaction_prediction": ["reaction_prediction"],
    "retrosynthesis": ["retrosynthesis"],
    "molecule_design": ["mol_caption", "mol_design"],
}


def _import_function(module_path: str, function_name: str):
    """Dynamically import a function from a dotted module path."""
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, function_name)


def _run_single_task(task_key, model_name, backend, sample_strategy, k,
                     temperature, max_new_tokens, output_dir):
    """Run a single benchmark task and return its results dict."""
    entry = TASK_REGISTRY[task_key]
    print(f"  BENCHMARK: {entry['display']}")

    fn = _import_function(entry["module"], entry["function"])

    # Build kwargs — merge task-specific kwargs with common ones
    kwargs = dict(entry["kwargs"])
    kwargs["model_name"] = model_name
    kwargs["backend"] = backend
    kwargs["sample_strategy"] = sample_strategy
    kwargs["k"] = k
    kwargs["temperature"] = temperature
    kwargs["max_new_tokens"] = max_new_tokens
    kwargs["output_dir"] = output_dir

    try:
        result = fn(**kwargs)
        return {"status": "success", "result": result}
    except Exception as e:
        print(f"\n[✗] Task {task_key} failed: {e}\n")
        return {"status": "error", "error": str(e)}


def run_all_benchmarks(
    model_name: str,
    backend: str = "api",
    tasks: list[str] | None = None,
    sample_strategy: str = "random",
    k: int = 8,
    temperature: float = 0.0,
    max_new_tokens: int = 256,
    output_dir: str = "results",
):
    """
    Run one or more ChemLLMBench tasks.

    Parameters
    ----------
    model_name : str
        Model identifier (e.g. "gpt-4", "claude-3-opus-20240229").
    backend : str
        "api" for LiteLLM or "local" for vLLM.
    tasks : list[str] or None
        Task keys or group names (see TASK_REGISTRY / TASK_GROUPS).
        None or ["all"] runs every task.
    sample_strategy : str
        "random" or "scaffold" for few-shot example selection.
    k : int
        Number of few-shot examples (0 for zero-shot).
    temperature : float
        Sampling temperature.
    max_new_tokens : int
        Max tokens to generate.
    output_dir : str
        Directory to save per-task results.

    Returns
    -------
    dict[str, dict]
        Mapping from task_key to its results dict.
    """
    # Resolve task list
    if tasks is None:
        tasks = ["all"]

    task_keys = []
    for t in tasks:
        t_lower = t.lower()
        if t_lower in TASK_GROUPS:
            task_keys.extend(TASK_GROUPS[t_lower])
        elif t_lower in TASK_REGISTRY:
            task_keys.append(t_lower)
        else:
            raise ValueError(
                f"Unknown task or group: {t!r}. "
                f"Available tasks: {list(TASK_REGISTRY.keys())}. "
                f"Available groups: {list(TASK_GROUPS.keys())}."
            )

    # Deduplicate while preserving order
    seen = set()
    unique_keys = []
    for k_ in task_keys:
        if k_ not in seen:
            seen.add(k_)
            unique_keys.append(k_)
    task_keys = unique_keys

    print(f"{'═' * 70}")
    print(f"  ChemLLMBench — Running {len(task_keys)} benchmark(s)")
    print(f"  Model    : {model_name}")
    print(f"  Backend  : {backend}")
    print(f"  Strategy : {sample_strategy} (k={k})")
    print(f"  Tasks    : {', '.join(task_keys)}")
    print(f"{'═' * 70}")

    all_results = {}
    for task_key in task_keys:
        result = _run_single_task(
            task_key=task_key,
            model_name=model_name,
            backend=backend,
            sample_strategy=sample_strategy,
            k=k,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            output_dir=output_dir,
        )
        all_results[task_key] = result

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'═' * 70}")
    print("  BENCHMARK SUMMARY")
    print(f"{'─' * 70}")
    for task_key, res in all_results.items():
        display = TASK_REGISTRY[task_key]["display"]
        if res["status"] == "success":
            metrics = res["result"].get("metrics", {})
            # Try to find the primary metric
            primary = (
                metrics.get("accuracy")
                or metrics.get("exact_match")
                or metrics.get("top1_accuracy")
            )
            metric_str = f"{primary:.4f}" if primary is not None else "done"
            print(f"  ✓ {display:<45s}  {metric_str}")
        else:
            print(f"  ✗ {display:<45s}  FAILED: {res['error'][:40]}")
    print(f"{'═' * 70}\n")

    # Save combined summary
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(output_dir, f"benchmark_summary_{timestamp}.json")
    summary = {
        "model": model_name,
        "backend": backend,
        "sample_strategy": sample_strategy,
        "k": k,
        "temperature": temperature,
        "timestamp": timestamp,
        "tasks": {
            key: {
                "display": TASK_REGISTRY[key]["display"],
                "status": res["status"],
                "metrics": res["result"].get("metrics") if res["status"] == "success" else None,
                "error": res.get("error"),
            }
            for key, res in all_results.items()
        },
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[✓] Summary saved to: {summary_path}")

    return all_results


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    available = sorted(list(TASK_REGISTRY.keys()) + list(TASK_GROUPS.keys()))

    parser = argparse.ArgumentParser(
        description="ChemLLMBench – Unified Chemistry LLM Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Available tasks:\n"
            + "\n".join(f"  {k:<25s} {TASK_REGISTRY[k]['display']}" for k in TASK_REGISTRY)
            + "\n\nAvailable groups:\n"
            + "\n".join(f"  {k:<25s} → {', '.join(v)}" for k, v in TASK_GROUPS.items())
        ),
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Model identifier (e.g. gpt-4, claude-3-opus-20240229).",
    )
    parser.add_argument(
        "--backend", type=str, default="api", choices=["api", "local"],
        help="'api' for LiteLLM (OpenAI/Anthropic/etc.), 'local' for vLLM.",
    )
    parser.add_argument(
        "--tasks", type=str, nargs="+", default=["all"],
        help=f"Task keys or group names to run. Choices: {available}",
    )
    parser.add_argument(
        "--strategy", type=str, default="random", choices=["random", "scaffold"],
        help="Few-shot example sampling strategy.",
    )
    parser.add_argument("--k", type=int, default=8, help="Number of few-shot examples (0 = zero-shot).")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max new tokens to generate.")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory.")

    args = parser.parse_args()

    run_all_benchmarks(
        model_name=args.model,
        backend=args.backend,
        tasks=args.tasks,
        sample_strategy=args.strategy,
        k=args.k,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
