"""
Multi-objective flow-shop scheduling example using pymoo.

The script loads processing times from a tab-separated file, converts decimal
commas to dots, sets up a three-objective flow-shop problem (minimize makespan,
sum of completion times, and total idle time) and solves it with three
pymoo algorithms (NSGA-II, NSGA-III, MOEA/D). A Gantt chart of the best
NSGA-II permutation is saved to ``flowshop_gantt.png`` in the current
directory.
"""

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.operators.crossover.pmx import PMX
from pymoo.operators.mutation.inversion import InversionMutation
from pymoo.operators.sampling.rnd import PermutationRandomSampling
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions

# Default order of machines for all jobs in the sample data.
MACHINE_ORDER = [
    "DO-TEHTANJE",
    "DO-KAPSULIRANJE",
    "DO-ČIŠČENJE",
    "DO-MEŠANJE",
]


def load_jobs(path: Path) -> pd.DataFrame:
    """Load the job routing table from a TSV file and normalize decimals."""

    df = pd.read_csv(path, sep="\t")
    df["anTotalTime"] = (
        df["anTotalTime"].astype(str).str.replace(",", ".", regex=False).astype(float)
    )
    return df


def build_processing_matrix(
    df: pd.DataFrame, machine_order: Sequence[str]
) -> Tuple[List[str], np.ndarray]:
    """Return job identifiers and processing matrix ordered by ``machine_order``."""

    job_ids = list(df["acKey"].unique())
    processing = np.zeros((len(job_ids), len(machine_order)), dtype=float)

    machine_index = {name: idx for idx, name in enumerate(machine_order)}
    for row in df.itertuples(index=False):
        job_idx = job_ids.index(row.acKey)
        if row.acIdent not in machine_index:
            raise ValueError(f"Unknown machine '{row.acIdent}' in input data")
        processing[job_idx, machine_index[row.acIdent]] = row.anTotalTime

    return job_ids, processing


def evaluate_permutation(
    permutation: Sequence[int], processing_times: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    """
    Evaluate a job permutation on the flow-shop and return timing details.

    Returns start times, end times, makespan, sum of job completion times, and
    aggregate machine idle time.
    """

    n_jobs = len(permutation)
    n_machines = processing_times.shape[1]
    start = np.zeros((n_jobs, n_machines), dtype=float)
    end = np.zeros_like(start)

    for i, job_idx in enumerate(permutation):
        for m in range(n_machines):
            earliest = 0.0
            if i > 0:
                earliest = max(earliest, end[i - 1, m])
            if m > 0:
                earliest = max(earliest, end[i, m - 1])

            start[i, m] = earliest
            duration = processing_times[job_idx, m]
            end[i, m] = earliest + duration

    makespan = end[-1, -1]
    completion_sum = float(end[:, -1].sum())
    machine_idle_time = 0.0
    for m in range(n_machines):
        busy = processing_times[permutation, m].sum()
        machine_idle_time += end[-1, m] - busy

    return start, end, makespan, completion_sum, machine_idle_time


class FlowShopProblem(ElementwiseProblem):
    """Permutation flow-shop formulation with three objectives."""

    def __init__(self, processing_times: np.ndarray):
        self.processing_times = processing_times
        n_jobs, _ = processing_times.shape
        super().__init__(n_var=n_jobs, n_obj=3, n_constr=0, xl=0, xu=n_jobs - 1, type_var=int)

    def _evaluate(self, x: Iterable[int], out: Dict, *args, **kwargs) -> None:
        permutation = np.array(x, dtype=int)
        _, _, makespan, completion_sum, idle_time = evaluate_permutation(
            permutation, self.processing_times
        )
        out["F"] = np.array([makespan, completion_sum, idle_time])


def configure_algorithms(problem: FlowShopProblem) -> Dict[str, object]:
    """Instantiate three pymoo algorithms suited for permutation problems."""

    pop_size = 60
    sampling = PermutationRandomSampling()
    crossover = PMX()
    mutation = InversionMutation()

    ref_dirs = get_reference_directions("das-dennis", problem.n_obj, n_partitions=12)

    algorithms: Dict[str, object] = {
        "NSGA2": NSGA2(
            pop_size=pop_size,
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
            eliminate_duplicates=True,
        ),
        "NSGA3": NSGA3(
            ref_dirs=ref_dirs,
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
            eliminate_duplicates=True,
        ),
        "MOEAD": MOEAD(
            ref_dirs=ref_dirs,
            n_neighbors=15,
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
        ),
    }
    return algorithms


def solve(problem: FlowShopProblem) -> Dict[str, object]:
    """Optimize with all configured algorithms and return results keyed by name."""

    algorithms = configure_algorithms(problem)
    termination = get_termination("n_gen", 150)

    results: Dict[str, object] = {}
    for name, algo in algorithms.items():
        results[name] = minimize(problem, algo, termination, save_history=False, verbose=False)
    return results


def choose_best_solution(results: Dict[str, object]) -> Tuple[str, np.ndarray]:
    """Pick the permutation with the smallest makespan across all results."""

    best_alg = ""
    best_perm: np.ndarray | None = None
    best_makespan = np.inf

    for name, result in results.items():
        makespans = result.F[:, 0]
        idx = int(np.argmin(makespans))
        if makespans[idx] < best_makespan:
            best_makespan = makespans[idx]
            best_alg = name
            best_perm = result.X[idx]

    if best_perm is None:
        raise RuntimeError("No solution found")

    return best_alg, np.array(best_perm, dtype=int)


def create_gantt_chart(
    job_ids: Sequence[str],
    machine_order: Sequence[str],
    permutation: Sequence[int],
    processing_times: np.ndarray,
    output_path: Path,
) -> None:
    """Plot and save a Gantt chart for the provided permutation."""

    start, end, makespan, completion_sum, idle_time = evaluate_permutation(
        permutation, processing_times
    )
    cmap = plt.cm.get_cmap("tab20")
    colors = list(cmap.colors)

    fig, ax = plt.subplots(figsize=(12, 6))

    for m_idx, machine in enumerate(machine_order):
        for pos, job_idx in enumerate(permutation):
            s = start[pos, m_idx]
            e = end[pos, m_idx]
            duration = e - s
            ax.broken_barh(
                [(s, duration)],
                (m_idx - 0.4, 0.8),
                facecolors=colors[job_idx % len(colors)],
                edgecolor="black",
                linewidth=0.5,
            )
            ax.text(
                s + duration / 2,
                m_idx,
                f"{job_ids[job_idx]}",
                va="center",
                ha="center",
                fontsize=8,
                color="black",
            )

    ax.set_xlabel("Time")
    ax.set_ylabel("Machine")
    ax.set_yticks(range(len(machine_order)))
    ax.set_yticklabels(machine_order)
    ax.set_title(
        "Flow-shop schedule\n"
        f"Makespan={makespan:.2f}, completion sum={completion_sum:.2f}, idle time={idle_time:.2f}"
    )
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    data_path = Path("data/flowshop_jobs.tsv")
    df = load_jobs(data_path)
    job_ids, processing_matrix = build_processing_matrix(df, MACHINE_ORDER)

    problem = FlowShopProblem(processing_matrix)
    results = solve(problem)
    best_algorithm, best_perm = choose_best_solution(results)

    output_path = Path("flowshop_gantt.png")
    create_gantt_chart(job_ids, MACHINE_ORDER, best_perm, processing_matrix, output_path)

    print("Ran multi-objective optimization with:")
    for name, result in results.items():
        print(f"  - {name}: {len(result.F)} solutions, example f-values head = {result.F[:3]}")
    print(f"Best makespan achieved by {best_algorithm}: {results[best_algorithm].F.min(axis=0)}")
    print(f"Gantt chart saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
