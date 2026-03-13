import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# ============================================================
# ПАРАМЕТРЫ
# ============================================================
fast_mode = True
save_folder = "results_hnc"
os.makedirs(save_folder, exist_ok=True)

# D+, He2+, C6+
Z = np.array([1.0, 2.0, 6.0], dtype=float)
names = ["D", "He", "C"]

# состав смеси
x = np.array([0.70, 0.20, 0.10], dtype=float)

# параметры модели
gamma = 0.8
kappa = 1.0
rho_total = 1.0

# численные параметры
if fast_mode:
    r_min = 0.03
    r_max = 8.0
    n_r = 140

    k_min = 0.08
    k_max = 18.0
    n_k = 140

    max_iter = 100
    tol = 1e-3
    mix = 0.05
else:
    r_min = 0.02
    r_max = 10.0
    n_r = 220

    k_min = 0.05
    k_max = 24.0
    n_k = 220

    max_iter = 180
    tol = 5e-4
    mix = 0.04

pair_names = {
    (0, 0): "g_DD",
    (1, 1): "g_HeHe",
    (2, 2): "g_CC",
    (0, 1): "g_DHe",
    (0, 2): "g_DC",
    (1, 2): "g_HeC",
}

# ============================================================
# СЕТКИ
# ============================================================

def make_grids(r_min, r_max, n_r, k_min, k_max, n_k):
    r = np.linspace(r_min, r_max, n_r)
    k = np.linspace(k_min, k_max, n_k)

    dr = r[1] - r[0]
    dk = k[1] - k[0]

    return r, k, dr, dk
def make_transform_matrices(r, k, dr, dk):
    """
    Матрицы для радиального 3D преобразования Фурье.
    """
    kr1 = np.outer(k, r)
    kr2 = np.outer(r, k)

    s1 = np.sin(kr1) / kr1
    s2 = np.sin(kr2) / kr2

    forward_matrix = 4.0 * np.pi * s1 * (r ** 2) * dr
    inverse_matrix = s2 * (k ** 2) * dk / (2.0 * np.pi ** 2)

    return forward_matrix, inverse_matrix


def radial_ft(f_r, forward_matrix):
    return forward_matrix @ f_r


def radial_ift(f_k, inverse_matrix):
    return inverse_matrix @ f_k
# ============================================================
# ПОТЕНЦИАЛ ЮКАВЫ
# ============================================================

def yukawa_potential(r, Z, gamma, kappa):
    """
    beta*u_ij(r) = gamma * Zi * Zj * exp(-kappa*r) / r
    """
    n_comp = len(Z)
    u = np.zeros((n_comp, n_comp, len(r)), dtype=float)

    for i in range(n_comp):
        for j in range(n_comp):
            gamma_ij = gamma * Z[i] * Z[j]
            u[i, j, :] = gamma_ij * np.exp(-kappa * r) / r

    return u
# ============================================================
# УРАВНЕНИЕ ORNSTEIN-ZERNIKE
# ============================================================
def solve_oz(c_r, rho, forward_matrix, inverse_matrix):
    n_comp = c_r.shape[0]
    n_k = forward_matrix.shape[0]
    n_r = inverse_matrix.shape[0]

    c_k = np.zeros((n_comp, n_comp, n_k), dtype=float)
    h_k = np.zeros((n_comp, n_comp, n_k), dtype=float)
    h_r = np.zeros((n_comp, n_comp, n_r), dtype=float)

    # c(r) -> c(k)
    for i in range(n_comp):
        for j in range(n_comp):
            c_k[i, j, :] = radial_ft(c_r[i, j, :], forward_matrix)

    R = np.diag(rho)
    I = np.eye(n_comp)

    # решение OZ для каждого k
    for m in range(n_k):
        C = c_k[:, :, m]
        A = I - C @ R

        try:
            H = np.linalg.solve(A, C)
        except np.linalg.LinAlgError:
            H = np.linalg.pinv(A) @ C

        h_k[:, :, m] = H

    # h(k) -> h(r)
    for i in range(n_comp):
        for j in range(n_comp):
            h_r[i, j, :] = radial_ift(h_k[i, j, :], inverse_matrix)

    return h_r
# ============================================================
# HNC
# ============================================================

def run_hnc(Z, x, gamma, kappa, rho_total, r, k, forward_matrix, inverse_matrix,
            max_iter=100, tol=1e-3, mix=0.05, verbose=True):
    if not np.isclose(np.sum(x), 1.0):
        raise ValueError("Сумма концентраций должна быть равна 1")

    rho = rho_total * x
    u = yukawa_potential(r, Z, gamma, kappa)

    # стартовое приближение
    c = -u.copy()

    errors = []
    ok = False

    for step in range(1, max_iter + 1):
        h = solve_oz(c, rho, forward_matrix, inverse_matrix)

        g = 1.0 + h
        g = np.clip(g, 1e-10, None)

        # HNC closure
        c_new = h - np.log(g) - u

        err = np.max(np.abs(c_new - c))
        errors.append(err)

        c = mix * c_new + (1.0 - mix) * c

        if verbose:
            print(f"iter={step:03d}, diff={err:.6e}")

        if np.isnan(err) or np.isinf(err):
            print("Расчет стал неустойчивым")
            break

        if err < tol:
            ok = True
            print("Сходимость достигнута")
            break

    # финальный пересчет
    h = solve_oz(c, rho, forward_matrix, inverse_matrix)
    g = 1.0 + h
    g = np.clip(g, 0.0, None)

    result = {
        "r": r,
        "k": k,
        "u": u,
        "c": c,
        "h": h,
        "g": g,
        "errors": np.array(errors),
        "converged": ok,
        "iterations": len(errors),
        "rho": rho,
        "gamma": gamma,
        "kappa": kappa,
        "x": x,
    }

    return result
# ============================================================
# СОХРАНЕНИЕ
# ============================================================

def save_rdf_table(result, folder):
    r = result["r"]
    g = result["g"]

    table = pd.DataFrame({"r": r})
    for (i, j), label in pair_names.items():
        table[label] = g[i, j, :]

    path = os.path.join(folder, "rdf_data.csv")
    table.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def save_peak_table(result, folder):
    r = result["r"]
    g = result["g"]

    rows = []
    for (i, j), label in pair_names.items():
        index_max = np.argmax(g[i, j, :])
        rows.append({
            "pair": label,
            "peak_r": r[index_max],
            "peak_g": g[i, j, index_max]
        })

    table = pd.DataFrame(rows)
    path = os.path.join(folder, "rdf_peaks.csv")
    table.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def draw_rdf(result, folder):
    r = result["r"]
    g = result["g"]

    # ---------- ОБЩИЙ ГРАФИК ----------
    plt.figure(figsize=(11, 7))
    for (i, j), label in pair_names.items():
        plt.plot(r, g[i, j, :], linewidth=1.8, label=label)

    plt.xlabel("r")
    plt.ylabel("g_ij(r)")
    plt.title(
        f"RDF for ternary ionic mixture: Gamma={result['gamma']}, "
        f"kappa={result['kappa']}, x={result['x']}"
    )

    # крупная и мелкая сетка
    plt.minorticks_on()
    plt.grid(True, which="major", linewidth=0.8, alpha=0.7)
    plt.grid(True, which="minor", linewidth=0.4, alpha=0.35)

    plt.legend()
    plt.tight_layout()

    common_path = os.path.join(folder, "rdf_plot_all.png")
    plt.savefig(common_path, dpi=300)
    plt.show()

    # ---------- ОТДЕЛЬНЫЕ ГРАФИКИ ----------
    single_paths = []

    for (i, j), label in pair_names.items():
        y = g[i, j, :]

        plt.figure(figsize=(9, 5.5))
        plt.plot(r, y, linewidth=2)

        plt.xlabel("r")
        plt.ylabel(label)
        plt.title(
            f"{label}: Gamma={result['gamma']}, "
            f"kappa={result['kappa']}, x={result['x']}"
        )

        plt.minorticks_on()
        plt.grid(True, which="major", linewidth=0.8, alpha=0.7)
        plt.grid(True, which="minor", linewidth=0.4, alpha=0.35)

        # небольшой отступ по оси Y, чтобы пики не прилипали к краям
        y_min = np.min(y)
        y_max = np.max(y)
        pad = 0.08 * (y_max - y_min + 1e-8)
        plt.ylim(y_min - pad, y_max + pad)

        plt.tight_layout()

        path = os.path.join(folder, f"{label}.png")
        plt.savefig(path, dpi=300)
        plt.show()

        single_paths.append(path)

    return common_path, single_paths

def draw_convergence(result, folder):
    errors = result["errors"]

    plt.figure(figsize=(8.5, 5.5))
    plt.semilogy(
        np.arange(1, len(errors) + 1),
        errors,
        marker="o",
        markersize=3,
        linewidth=1.5
    )

    plt.xlabel("Iteration")
    plt.ylabel("Max error")
    plt.title("HNC convergence")

    plt.minorticks_on()
    plt.grid(True, which="major", linewidth=0.8, alpha=0.7)
    plt.grid(True, which="minor", linewidth=0.4, alpha=0.35)

    plt.tight_layout()

    path = os.path.join(folder, "hnc_convergence.png")
    plt.savefig(path, dpi=300)
    plt.show()
    return path
# ============================================================
# ОСНОВНОЙ ЗАПУСК
# ============================================================
if __name__ == "__main__":
    print("Start calculation...")

    r, k, dr, dk = make_grids(r_min, r_max, n_r, k_min, k_max, n_k)
    forward_matrix, inverse_matrix = make_transform_matrices(r, k, dr, dk)

    result = run_hnc(
        Z=Z,
        x=x,
        gamma=gamma,
        kappa=kappa,
        rho_total=rho_total,
        r=r,
        k=k,
        forward_matrix=forward_matrix,
        inverse_matrix=inverse_matrix,
        max_iter=max_iter,
        tol=tol,
        mix=mix,
        verbose=True,
    )

    rdf_file = save_rdf_table(result, save_folder)
    peaks_file = save_peak_table(result, save_folder)
    rdf_plot_all, rdf_single_plots = draw_rdf(result, save_folder)
    conv_plot = draw_convergence(result, save_folder)

    print("\n=== DONE ===")
    print(f"Converged: {result['converged']}")
    print(f"Iterations: {result['iterations']}")
    print(f"RDF data: {rdf_file}")
    print(f"Peaks data: {peaks_file}")
    print(f"RDF common plot: {rdf_plot_all}")
print("RDF separate plots:")
for p in rdf_single_plots:
    print(p)
    print(f"Convergence plot: {conv_plot}")
