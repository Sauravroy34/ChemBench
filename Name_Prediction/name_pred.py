"""
Name Prediction Benchmark
============================
Modular benchmark for molecular naming and identifier prediction using LLMs.
Supports IUPAC name generation, SMILES prediction, and molecular formula tasks.

Usage (as a script):
    # API model (OpenAI, Anthropic, etc.)
    python -m Name_Prediction.name_pred \
        --task SMILES_TO_IUPAC --model gpt-4 --backend api \
        --strategy random --k 8 --n 1 --temperature 0.0

    # Local HuggingFace model via vLLM
    python -m Name_Prediction.name_pred \
        --task IUPAC_TO_SMILES --model meta-llama/Llama-2-7b-chat-hf --backend local \
        --strategy scaffold --k 8 --n 1 --temperature 0.0

Usage (as a library):
    from ChemLLMBench.Name_Prediction.name_pred import run_benchmark
    results = run_benchmark(
        task_name="SMILES_TO_IUPAC", model_name="gpt-4", backend="api",
        sample_strategy="random", k=8,
    )
"""

import os
import re
import json
import argparse
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm

from ChemLLMBench.Name_Prediction.data import load_dataset, TASK_COL_MAP
from ChemLLMBench.utils import LiteLLMModel, HuggingFaceModel


# ── Supported tasks ─────────────────────────────────────────────────────────
SUPPORTED_TASKS = list(TASK_COL_MAP.keys())


# ── Helper: parse the generated answer from raw LLM output ──────────────────
def _parse_prediction(raw_text, task_name):
    """Extract the predicted value from the raw LLM response.

    The model is prompted to reply in the format ``<col>: <value>``.
    This function tries to extract ``<value>`` from that pattern,
    falling back to the full stripped text if no pattern is found.

    Returns an empty string on parse failure.
    """
    if not raw_text:
        return ""

    text = raw_text.strip()
    _, to_col = TASK_COL_MAP[task_name]

    # Try structured pattern  "<to_col>: <answer>"
    pattern = re.compile(
        rf"{re.escape(to_col)}\s*:\s*(.+)", re.IGNORECASE
    )
    match = pattern.search(text)
    if match:
        return match.group(1).strip()

    # Fallback: return the first non-empty line
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line

    return ""


# ── Helper: validate a SMILES string ────────────────────────────────────────
def _is_valid_smiles(s):
    """Return True if *s* can be parsed into a valid RDKit molecule."""
    from rdkit import Chem
    if not s:
        return False
    mol = Chem.MolFromSmiles(s)
    return mol is not None


# ── Helper: canonicalise SMILES for exact comparison ────────────────────────
def _canonical(s):
    """Return canonical SMILES or empty string if invalid."""
    from rdkit import Chem
    if not s:
        return ""
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol)


# ── Helper: Tanimoto similarity between two SMILES ─────────────────────────
def _tanimoto_similarity(smi_a, smi_b):
    """Morgan-fingerprint Tanimoto similarity between two SMILES strings.
    Returns 0.0 if either SMILES is invalid."""
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    mol_a = Chem.MolFromSmiles(smi_a) if smi_a else None
    mol_b = Chem.MolFromSmiles(smi_b) if smi_b else None
    if mol_a is None or mol_b is None:
        return 0.0

    fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp_a = fp_gen.GetFingerprint(mol_a)
    fp_b = fp_gen.GetFingerprint(mol_b)
    return DataStructs.TanimotoSimilarity(fp_a, fp_b)


# ── Helper: compute metrics ─────────────────────────────────────────────────
def _compute_metrics(y_true, y_pred, task_name):
    """Compute task-appropriate metrics for name-prediction.

    Metrics computed:
    - **exact_match**: fraction of predictions that exactly match ground truth
      (case-insensitive for text, canonical SMILES for SMILES outputs).
    - **valid_rate**: fraction of non-empty predictions.

    For tasks that output SMILES (IUPAC_TO_SMILES, FORMULA_TO_SMILES):
    - **valid_smiles_rate**: fraction of predictions that are valid SMILES.
    - **mean_tanimoto**: mean Morgan-FP Tanimoto similarity with ground truth.

    For tasks that output formulas (SMILES_TO_FORMULA):
    - **formula_match**: fraction of formulas that match (case-insensitive).
    """
    _, to_col = TASK_COL_MAP[task_name]
    total = len(y_true)

    # ── basic counts ──
    n_valid = sum(1 for p in y_pred if p.strip())
    n_invalid = total - n_valid

    # ── exact match (normalised) ──
    exact = 0
    for gt, pred in zip(y_true, y_pred):
        gt_s = str(gt).strip()
        pred_s = str(pred).strip()
        if to_col == "smiles":
            # Compare canonical forms
            if _canonical(pred_s) and _canonical(pred_s) == _canonical(gt_s):
                exact += 1
        else:
            # Case-insensitive text comparison
            if pred_s.lower() == gt_s.lower():
                exact += 1

    metrics = {
        "exact_match": exact / total if total else 0.0,
        "valid_rate": n_valid / total if total else 0.0,
        "total": total,
        "valid_predictions": n_valid,
        "invalid_predictions": n_invalid,
    }

    # ── SMILES-output tasks: validity & Tanimoto ──
    if to_col == "smiles":
        valid_smiles = sum(1 for p in y_pred if _is_valid_smiles(p.strip()))
        tanimoto_scores = [
            _tanimoto_similarity(str(gt).strip(), str(p).strip())
            for gt, p in zip(y_true, y_pred)
        ]
        metrics["valid_smiles_rate"] = valid_smiles / total if total else 0.0
        metrics["mean_tanimoto"] = float(np.mean(tanimoto_scores)) if tanimoto_scores else 0.0

    # ── Formula-output task ──
    if to_col == "formula":
        formula_match = sum(
            1 for gt, p in zip(y_true, y_pred)
            if str(p).strip().lower() == str(gt).strip().lower()
        )
        metrics["formula_match"] = formula_match / total if total else 0.0

    return metrics


# ── Helper: build model ─────────────────────────────────────────────────────
def _build_model(backend, model_name, temperature, max_new_tokens=256):
    """Instantiate the right ModelInterface subclass.

    Uses a longer default *max_new_tokens* (256) than Property Prediction
    because IUPAC names can be lengthy.
    """
    if backend == "api":
        return LiteLLMModel(
            model_name=model_name,
            max_new_tokens=max_new_tokens,
        )
    elif backend == "local":
        return HuggingFaceModel(
            temperature=temperature,
            model_name=model_name,
            max_new_tokens=max_new_tokens,
        )
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Use 'api' or 'local'.")


# ── Main benchmark runner ────────────────────────────────────────────────────
def run_benchmark(
    task_name: str,
    model_name: str,
    backend: str = "api",
    sample_strategy: str = "random",
    k: int = 8,
    n: int = 1,
    temperature: float = 0.0,
    max_new_tokens: int = 256,
    output_dir: str = "results",
):
    """
    Run the name-prediction benchmark on one task.

    Parameters
    ----------
    task_name : str
        One of SMILES_TO_IUPAC, IUPAC_TO_SMILES, SMILES_TO_FORMULA,
        FORMULA_TO_SMILES.
    model_name : str
        Model identifier (e.g. "gpt-4", "meta-llama/Llama-2-7b-chat-hf").
    backend : str
        "api" for LiteLLM-based calls, "local" for vLLM-based local inference.
    sample_strategy : str
        "random" or "scaffold" for few-shot example selection.
    k : int
        Number of few-shot examples (0 for zero-shot).
    n : int
        Number of completions to sample per molecule.
    temperature : float
        Sampling temperature.
    max_new_tokens : int
        Max tokens to generate.
    output_dir : str
        Directory to save results.

    Returns
    -------
    dict  with keys: "metrics", "predictions", "raw_outputs"
    """
    if task_name not in SUPPORTED_TASKS:
        raise ValueError(
            f"Unknown task: {task_name!r}. Choose from {SUPPORTED_TASKS}"
        )

    # ── 1. Load data & prompts ───────────────────────────────────────────
    print(f"[*] Loading task: {task_name}  (strategy={sample_strategy}, k={k})")
    system_prompts, user_prompts, df_test = load_dataset(
        name=task_name, sample_strategy=sample_strategy, k=k,
    )

    _, to_col = TASK_COL_MAP[task_name]
    y_true = df_test[to_col].astype(str).tolist()

    # ── 2. Build model ───────────────────────────────────────────────────
    print(f"[*] Initialising model: {model_name}  (backend={backend})")
    model = _build_model(
        backend=backend,
        model_name=model_name,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )

    # ── 3. Run inference ─────────────────────────────────────────────────
    raw_outputs = []
    predictions = []

    print(f"[*] Running inference on {len(user_prompts)} molecules …")
    for idx in tqdm(range(len(user_prompts)), desc=task_name):
        sys_prompt = system_prompts[idx]
        usr_prompt = user_prompts[idx]

        # Generate model response
        outputs = model.generate(
            user_prompt=usr_prompt, sys_prompt=sys_prompt,
            n=n,
        )

        raw_text = outputs[0] if outputs else ""
        raw_outputs.append(raw_text)

        pred = _parse_prediction(raw_text, task_name)
        predictions.append(pred)

    # ── 4. Evaluate ──────────────────────────────────────────────────────
    print("[*] Computing metrics …")
    metrics = _compute_metrics(y_true, predictions, task_name)

    print(f"\n{'═' * 55}")
    print(f"  Task    : {task_name}")
    print(f"  Model   : {model_name}")
    print(f"  Strategy: {sample_strategy} (k={k})")
    print(f"{'─' * 55}")
    print(f"  Exact Match  : {metrics['exact_match']:.4f}")
    print(f"  Valid Rate   : {metrics['valid_rate']:.4f}")
    if "valid_smiles_rate" in metrics:
        print(f"  Valid SMILES  : {metrics['valid_smiles_rate']:.4f}")
        print(f"  Mean Tanimoto: {metrics['mean_tanimoto']:.4f}")
    if "formula_match" in metrics:
        print(f"  Formula Match: {metrics['formula_match']:.4f}")
    print(f"  Valid        : {metrics['valid_predictions']}/{metrics['total']}")
    print(f"{'═' * 55}\n")

    # ── 5. Save results ──────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = model_name.replace("/", "_").replace("\\", "_")
    run_tag = f"{task_name}_{safe_model}_{sample_strategy}_k{k}_{timestamp}"

    # Save predictions CSV
    results_df = df_test.copy().reset_index(drop=True)
    results_df["raw_output"] = raw_outputs
    results_df["prediction"] = predictions
    results_df["ground_truth"] = y_true
    csv_path = os.path.join(output_dir, f"{run_tag}_predictions.csv")
    results_df.to_csv(csv_path, index=False)

    # Save metrics JSON
    meta = {
        "task": task_name,
        "model": model_name,
        "backend": backend,
        "sample_strategy": sample_strategy,
        "k": k,
        "n": n,
        "temperature": temperature,
        "timestamp": timestamp,
        "metrics": metrics,
    }
    json_path = os.path.join(output_dir, f"{run_tag}_metrics.json")
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"[✓] Predictions saved to: {csv_path}")
    print(f"[✓] Metrics saved to:     {json_path}")

    return {"metrics": metrics, "predictions": predictions, "raw_outputs": raw_outputs}


# ── CLI entry-point ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="ChemLLMBench – Molecular Name Prediction Benchmark"
    )
    parser.add_argument(
        "--task", type=str, required=True,
        choices=SUPPORTED_TASKS,
        help="Task name (e.g. SMILES_TO_IUPAC, IUPAC_TO_SMILES).",
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Model identifier (e.g. gpt-4, claude-3-opus-20240229, meta-llama/Llama-2-7b-chat-hf).",
    )
    parser.add_argument(
        "--backend", type=str, default="api", choices=["api", "local"],
        help="'api' for LiteLLM (OpenAI/Anthropic/etc.), 'local' for vLLM.",
    )
    parser.add_argument(
        "--strategy", type=str, default="random", choices=["random", "scaffold"],
        help="Few-shot example sampling strategy.",
    )
    parser.add_argument("--k", type=int, default=8, help="Number of few-shot examples (0 = zero-shot).")
    parser.add_argument("--n", type=int, default=1, help="Number of completions per molecule.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max new tokens to generate.")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory.")

    args = parser.parse_args()

    run_benchmark(
        task_name=args.task,
        model_name=args.model,
        backend=args.backend,
        sample_strategy=args.strategy,
        k=args.k,
        n=args.n,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()