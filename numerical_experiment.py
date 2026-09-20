import csv
from pathlib import Path

import numpy as np
from scipy.linalg import eigh


Q_VALUES = (-2.0, -1.5, -1.0, -0.5, -0.25)
N_VALUES = range(4, 11)
STARTS = 120
SEED = 20260919
TOL = 1.0e-10
MAX_ITER = 500


def traceless_basis(n):
    matrices = []
    for j in range(n):
        for i in range(n):
            if i != j:
                x = np.zeros((n, n), dtype=complex)
                x[i, j] = 1.0
                matrices.append(x)
    for k in range(1, n):
        x = np.zeros((n, n), dtype=complex)
        x[:k, :k] += np.eye(k) / np.sqrt(k * (k + 1))
        x[k, k] = -k / np.sqrt(k * (k + 1))
        matrices.append(x)
    return np.column_stack([x.reshape(-1, order="F") for x in matrices])


def normalize_traceless(x):
    n = x.shape[0]
    x = x - np.trace(x) * np.eye(n) / n
    return x / np.linalg.norm(x, "fro")


def update_a(b, q, basis):
    n = b.shape[0]
    operator = np.kron(b.T, np.eye(n)) - q * np.kron(np.eye(n), b)
    restricted = operator @ basis
    gram = restricted.conj().T @ restricted
    last = gram.shape[0] - 1
    _, vectors = eigh(gram, subset_by_index=[last, last])
    return (basis @ vectors[:, -1]).reshape((n, n), order="F")


def update_b(a, q, basis):
    n = a.shape[0]
    operator = np.kron(np.eye(n), a) - q * np.kron(a.T, np.eye(n))
    restricted = operator @ basis
    gram = restricted.conj().T @ restricted
    last = gram.shape[0] - 1
    _, vectors = eigh(gram, subset_by_index=[last, last])
    return (basis @ vectors[:, -1]).reshape((n, n), order="F")


def rq(a, b, q):
    commutator = a @ b - q * b @ a
    return float(np.linalg.norm(commutator, "fro") ** 2)


def run_start(n, q, basis, rng):
    a = normalize_traceless(
        rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    )
    b = normalize_traceless(
        rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    )
    old = rq(a, b, q)

    for _ in range(MAX_ITER):
        a = update_a(b, q, basis)
        a /= np.linalg.norm(a, "fro")
        b = update_b(a, q, basis)
        b /= np.linalg.norm(b, "fro")
        new = rq(a, b, q)
        if abs(new - old) <= TOL * max(1.0, abs(new)):
            return new
        old = new
    return old


def proposed_bound(n, q):
    g = (n * n - 3 * n + 3) / (n * (n - 1))
    return max(g * (1 - q) ** 2, 1 + q * q)


def main():
    rng = np.random.default_rng(SEED)
    rows = []

    for n in N_VALUES:
        basis = traceless_basis(n)
        for q in Q_VALUES:
            best = max(run_start(n, q, basis, rng) for _ in range(STARTS))
            bound = proposed_bound(n, q)
            row = {
                "n": n,
                "q": q,
                "best_value": f"{best:.12f}",
                "candidate_value": f"{bound:.12f}",
                "difference": f"{best - bound:+.12e}",
            }
            rows.append(row)
            print(
                f"n={n:2d} q={q:5g} best={best:.12f} "
                f"bound={bound:.12f} difference={best - bound:+.3e}",
                flush=True,
            )

    output = Path(__file__).with_name("results.csv")
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
