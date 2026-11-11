"""
Ejercicio 1 - Ajuste del parámetro de temperatura del Recocido Simulado (SA)
con Metodología de Superficie de Respuesta (RSM, Opción B).

Resumen del diseño
  - Algoritmo : Recocido Simulado con vecindario 2-opt sobre el TSP.
  - Factor    : temperatura inicial T0 (5 niveles: axial bajo, bajo, centro, alto, axial alto).
  - Bloque    : tamaño del problema n in {5, 10, 15} (una instancia por tamaño).
  - Réplicas  : REP_EXTERNOS corridas en cada punto externo y REP_CENTRO en el centro
                (por defecto 5 y 10 -> 30 corridas por tamaño, 90 en total).
  - Modelo    : Costo = b0 + b1*T + b2*T^2 (uno por tamaño), con ANOVA de regresión,
                falta de ajuste vs error puro, y T* = -b1 / (2*b2).
  - Bootstrap : mediana (IC 95%) del costo en cada punto de diseño.

Presupuesto igual para todos los niveles: cada corrida hace PASOS cambios de temperatura
y ITERS_POR_TEMP iteraciones por temperatura, enfriando de forma geométrica desde T0
hasta T_FINAL. Así lo único que cambia entre niveles es T0 (no el esfuerzo de cómputo).

Uso:
    pip install numpy pandas scipy matplotlib        (statsmodels es opcional)
    python sa_rsm_temperatura.py
"""

import math
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# =====================================================================
# PARÁMETROS DEL EXPERIMENTO  (edita aquí)
# =====================================================================
MATRICULA = 2068543            # <-- CAMBIA ESTO por tu matrícula (fija tu valor de r)
TAMANOS = [5, 10, 15]         # bloques: tamaño del TSP
T_MIN, T_MAX = 0.5, 25.0      # rango de temperatura que defines tú (elegido con un barrido
                              # piloto: las T0 buenas están por debajo de ~10 en estos TSP)
REP_EXTERNOS = 5              # réplicas en axial bajo, bajo, alto, axial alto
REP_CENTRO = 10               # réplicas en el punto central (error puro)
                              # (usa REP_EXTERNOS = 10 si quieres 50 obs. por tamaño)

PASOS = 30                    # cambios de temperatura por corrida
ITERS_POR_TEMP = 5            # iteraciones por cada temperatura (presupuesto = 150)
                              # OJO: con presupuestos grandes el SA llega al óptimo con
                              # cualquier T0 y el experimento no puede distinguir niveles.
T_FINAL = 0.01                # temperatura final (igual para todos los niveles)

BOOTSTRAP_B = 5000            # remuestreos para el IC de la mediana
ALFA = 0.05                   # nivel de significancia

# ---- Valor aleatorio r: se genera UNA sola vez y se usa en todo el experimento ----
random.seed(MATRICULA)
R = random.random()

# Fracciones del rango para los 5 niveles (Opción B)
NOMBRES_NIVEL = ["axial bajo", "bajo", "centro", "alto", "axial alto"]
FRACCIONES = [0.2 * R, 0.2 * R + 0.2, 0.2 * R + 0.4, 0.2 * R + 0.6, 0.2 * R + 0.8]
NIVELES_T = [T_MIN + f * (T_MAX - T_MIN) for f in FRACCIONES]
REPLICAS = [REP_EXTERNOS, REP_EXTERNOS, REP_CENTRO, REP_EXTERNOS, REP_EXTERNOS]


# =====================================================================
# TSP y Recocido Simulado
# =====================================================================
def generar_instancia(n, semilla):
    rng = random.Random(semilla)
    ciudades = [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(n)]
    return [[math.dist(a, b) for b in ciudades] for a in ciudades]


def longitud_tour(tour, d):
    n = len(tour)
    return sum(d[tour[i]][tour[(i + 1) % n]] for i in range(n))


def optimo_held_karp(d):
    """Longitud óptima exacta (solo como referencia; n <= 15 es rápido)."""
    n = len(d)
    m = n - 1
    INF = math.inf
    dp = [[INF] * m for _ in range(1 << m)]
    for j in range(m):
        dp[1 << j][j] = d[0][j + 1]
    for mask in range(1, 1 << m):
        for j in range(m):
            if not (mask >> j) & 1 or dp[mask][j] == INF:
                continue
            for k in range(m):
                if (mask >> k) & 1:
                    continue
                nm = mask | (1 << k)
                v = dp[mask][j] + d[j + 1][k + 1]
                if v < dp[nm][k]:
                    dp[nm][k] = v
    full = (1 << m) - 1
    return min(dp[full][j] + d[j + 1][0] for j in range(m))


def recocido_simulado(d, T0, semilla):
    """
    SA con movimiento 2-opt aleatorio y enfriamiento geométrico T0 -> T_FINAL en PASOS pasos.
    Devuelve la mejor longitud encontrada (el 'costo' de la corrida).
    """
    rng = random.Random(semilla)
    n = len(d)
    tour = list(range(n))
    rng.shuffle(tour)
    actual = longitud_tour(tour, d)
    mejor = actual
    alpha = (T_FINAL / T0) ** (1.0 / PASOS)
    T = T0
    for _ in range(PASOS):
        for _ in range(ITERS_POR_TEMP):
            i = rng.randrange(0, n - 2)
            j = rng.randrange(i + 2, n)
            if i == 0 and j == n - 1:
                continue
            a, b = tour[i], tour[i + 1]
            c, e = tour[j], tour[(j + 1) % n]
            delta = d[a][c] + d[b][e] - d[a][b] - d[c][e]
            if delta < 0 or rng.random() < math.exp(-delta / T):
                tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                actual += delta
                if actual < mejor:
                    mejor = actual
        T *= alpha
    return mejor


# =====================================================================
# Estadística: regresión cuadrática, ANOVA, falta de ajuste, bootstrap
# =====================================================================
def ajustar_cuadratico(T, y):
    """OLS de y = b0 + b1*T + b2*T^2. Devuelve dict con coeficientes, p-values y ANOVA."""
    T, y = np.asarray(T, float), np.asarray(y, float)
    N = len(y)
    X = np.column_stack([np.ones(N), T, T ** 2])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    sse = float(np.sum((y - fitted) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    ssr = sst - sse
    gl_res = N - 3

    out = dict(beta=beta, N=N, sse=sse, sst=sst, ssr=ssr, gl_res=gl_res,
               r2=(ssr / sst) if sst > 0 else np.nan)

    if sse < 1e-12 or sst < 1e-12:      # sin variación: no hay nada que modelar
        out["r2"] = np.nan
        out.update(p_coef=[np.nan] * 3, F=np.nan, p_F=np.nan, degenerado=True)
        return out

    sigma2 = sse / gl_res
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    tvals = beta / se
    out["p_coef"] = list(2 * stats.t.sf(np.abs(tvals), gl_res))
    out["F"] = (ssr / 2) / sigma2
    out["p_F"] = stats.f.sf(out["F"], 2, gl_res)
    out["degenerado"] = False
    return out


def falta_de_ajuste(df_n, ajuste):
    """F de falta de ajuste: separa error puro (réplicas) del error del modelo."""
    N = ajuste["N"]
    k = df_n["nivel"].nunique()
    sspe = float(sum(((g["costo"] - g["costo"].mean()) ** 2).sum()
                     for _, g in df_n.groupby("nivel")))
    sslof = max(ajuste["sse"] - sspe, 0.0)
    gl_pe, gl_lof = N - k, k - 3
    res = dict(sspe=sspe, sslof=sslof, gl_pe=gl_pe, gl_lof=gl_lof)
    if sspe < 1e-12 or gl_lof <= 0 or gl_pe <= 0:
        res.update(F=np.nan, p=np.nan)
    else:
        F = (sslof / gl_lof) / (sspe / gl_pe)
        res.update(F=F, p=stats.f.sf(F, gl_lof, gl_pe))
    return res


def bootstrap_mediana(x, rng, B=BOOTSTRAP_B):
    """Mediana y su IC 95% (percentil) por bootstrap."""
    x = np.asarray(x, float)
    muestras = rng.choice(x, size=(B, len(x)), replace=True)
    meds = np.median(muestras, axis=1)
    return float(np.median(x)), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def fmt_p(p):
    return "  n/d  " if np.isnan(p) else (" <0.001" if p < 0.001 else f"{p:7.3f}")


# =====================================================================
# Programa principal
# =====================================================================
if __name__ == "__main__":
    print("=" * 78)
    print("DOE / RSM - Temperatura inicial del Recocido Simulado (TSP con 2-opt)")
    print("=" * 78)
    print(f"Matrícula (semilla) = {MATRICULA}  ->  r = {R:.6f}  (se usa en todo el experimento)")
    print(f"Rango de temperatura: [{T_MIN}, {T_MAX}]")
    print(f"{'Nivel':<12} | {'Fracción':>9} | {'T0':>8} | {'Réplicas':>8}")
    print("-" * 46)
    for nom, f, t, rep in zip(NOMBRES_NIVEL, FRACCIONES, NIVELES_T, REPLICAS):
        print(f"{nom:<12} | {f:9.4f} | {t:8.3f} | {rep:8d}")
    total_por_n = sum(REPLICAS)
    print(f"\nCorridas por tamaño: {total_por_n}  |  total: {total_por_n * len(TAMANOS)}")
    print(f"Presupuesto por corrida: {PASOS} pasos x {ITERS_POR_TEMP} iteraciones "
          f"= {PASOS * ITERS_POR_TEMP}  (T_final = {T_FINAL})")

    # ---------------- 1) Correr el experimento ----------------
    filas, instancias = [], {}
    for n in TAMANOS:
        d = generar_instancia(n, semilla=MATRICULA + n)
        instancias[n] = (d, optimo_held_karp(d))
        for idx, (nom, f, T0, rep) in enumerate(zip(NOMBRES_NIVEL, FRACCIONES,
                                                    NIVELES_T, REPLICAS)):
            for r_i in range(rep):
                semilla = MATRICULA * 100_000 + n * 1_000 + idx * 100 + r_i   # única por corrida
                filas.append(dict(n=n, nivel=nom, fraccion=f, T=T0, rep=r_i + 1,
                                  costo=recocido_simulado(d, T0, semilla)))
    df = pd.DataFrame(filas)
    df.to_csv("sa_rsm_resultados.csv", index=False)
    print("\nResultados crudos guardados en sa_rsm_resultados.csv")

    rng_boot = np.random.default_rng(MATRICULA)
    resumen_final, modelos = [], {}

    # ---------------- 2) Análisis por tamaño ----------------
    for n in TAMANOS:
        df_n = df[df["n"] == n]
        opt = instancias[n][1]
        aj = ajustar_cuadratico(df_n["T"], df_n["costo"])
        lof = falta_de_ajuste(df_n, aj)
        b0, b1, b2 = aj["beta"]

        print("\n" + "=" * 78)
        print(f"TAMAÑO n = {n}   (costo óptimo exacto de la instancia = {opt:.2f})")
        print("=" * 78)

        # Bootstrap en cada punto de diseño
        puntos = []
        print(f"{'Nivel':<11} | {'T0':>8} | {'Media':>8} | {'Mediana':>8} | "
              f"{'IC 95% de la mediana':>22}")
        print("-" * 70)
        for nom, T0 in zip(NOMBRES_NIVEL, NIVELES_T):
            x = df_n[df_n["nivel"] == nom]["costo"].values
            med, lo, hi = bootstrap_mediana(x, rng_boot)
            puntos.append(dict(nivel=nom, T=T0, media=x.mean(), mediana=med, lo=lo, hi=hi))
            print(f"{nom:<11} | {T0:8.3f} | {x.mean():8.2f} | {med:8.2f} | "
                  f"[{lo:8.2f}, {hi:8.2f}]")

        # Modelo cuadrático
        print(f"\nModelo ajustado:  Costo = {b0:.3f} + ({b1:.4f})*T + ({b2:.6f})*T^2")
        if aj["degenerado"]:
            print("  Los costos no varían (el algoritmo encontró siempre el mismo valor): "
                  "no hay curvatura que estimar.")
        else:
            pc = aj["p_coef"]
            print(f"  p-values:  b0 {fmt_p(pc[0])} | b1 {fmt_p(pc[1])} | b2 {fmt_p(pc[2])}"
                  f"   |  R2 = {aj['r2']:.3f}")
            print(f"\n  ANOVA de regresión")
            print(f"  {'Fuente':<16} | {'SC':>10} | {'gl':>3} | {'CM':>9} | {'F':>7} | {'p':>7}")
            print("  " + "-" * 62)
            print(f"  {'Regresión':<16} | {aj['ssr']:10.2f} | {2:3d} | {aj['ssr'] / 2:9.2f} | "
                  f"{aj['F']:7.2f} | {fmt_p(aj['p_F'])}")
            print(f"  {'Residual':<16} | {aj['sse']:10.2f} | {aj['gl_res']:3d} | "
                  f"{aj['sse'] / aj['gl_res']:9.2f} |         |")
            if lof["gl_lof"] > 0:
                cm_lof = lof["sslof"] / lof["gl_lof"]
                print(f"  {'  Falta de ajuste':<16} | {lof['sslof']:10.2f} | {lof['gl_lof']:3d} | "
                      f"{cm_lof:9.2f} | {lof['F']:7.2f} | {fmt_p(lof['p'])}")
            print(f"  {'  Error puro':<16} | {lof['sspe']:10.2f} | {lof['gl_pe']:3d} | "
                  f"{lof['sspe'] / lof['gl_pe']:9.2f} |         |")
            print(f"  {'Total':<16} | {aj['sst']:10.2f} | {aj['N'] - 1:3d} |")

            if not np.isnan(lof["p"]):
                if lof["p"] >= ALFA:
                    print(f"  Falta de ajuste NO significativa (p={lof['p']:.3f}): "
                          f"el modelo cuadrático describe bien los datos.")
                else:
                    print(f"  Falta de ajuste SIGNIFICATIVA (p={lof['p']:.3f}): "
                          f"un término cuadrático no basta para esta curva.")
            if pc[2] < ALFA:
                print(f"  Curvatura significativa (p de b2 = {fmt_p(pc[2]).strip()}).")
            else:
                print(f"  Curvatura NO significativa (p de b2 = {fmt_p(pc[2]).strip()}).")

        # Temperatura óptima del modelo
        T_star, valido, motivo = np.nan, False, ""
        if aj["degenerado"] or abs(b2) < 1e-15:
            motivo = "sin curvatura"
        elif b2 <= 0:
            T_star = -b1 / (2 * b2)
            motivo = "b2 <= 0: la parábola abre hacia abajo, T* es un MÁXIMO (no sirve)"
        else:
            T_star = -b1 / (2 * b2)
            if NIVELES_T[0] <= T_star <= NIVELES_T[-1]:
                if aj["p_coef"][2] < ALFA:
                    valido, motivo = True, "mínimo dentro del rango y curvatura significativa"
                else:
                    motivo = ("mínimo dentro del rango, pero la curvatura (b2) NO es "
                              "significativa: T* no es confiable")
            else:
                motivo = (f"mínimo fuera del rango explorado "
                          f"[{NIVELES_T[0]:.2f}, {NIVELES_T[-1]:.2f}]")
        if not np.isnan(T_star):
            print(f"\n  T* = -b1/(2*b2) = {T_star:.3f}   ->  {motivo}")
        else:
            print(f"\n  T* no definida ({motivo})")

        mejor = min(puntos, key=lambda p: p["mediana"])
        resumen_final.append(dict(
            n=n, T_star=T_star, T_star_valida=valido, b2=b2,
            p_b2=aj["p_coef"][2], p_lof=lof["p"],
            mejor_nivel=mejor["nivel"], mejor_T=mejor["T"],
            mediana=mejor["mediana"], ic_lo=mejor["lo"], ic_hi=mejor["hi"], optimo=opt))
        modelos[n] = dict(beta=aj["beta"], puntos=puntos, aj=aj, lof=lof,
                          T_star=T_star, valido=valido)

    # ---------------- 3) Tabla resumen ----------------
    print("\n" + "=" * 100)
    print("TABLA RESUMEN (mejor punto de diseño según la mediana bootstrap)")
    print("=" * 100)
    print(f"{'n':>3} | {'Mejor T0':>9} | {'Nivel':<10} | {'Mediana':>8} | "
          f"{'IC 95% bootstrap':>20} | {'Óptimo':>8} | {'T* modelo':>10} | {'Confiable':>9}")
    print("-" * 100)
    for r in resumen_final:
        ts = "   n/d" if np.isnan(r["T_star"]) else f"{r['T_star']:10.3f}"
        print(f"{r['n']:>3} | {r['mejor_T']:9.3f} | {r['mejor_nivel']:<10} | {r['mediana']:8.2f} | "
              f"[{r['ic_lo']:7.2f}, {r['ic_hi']:7.2f}] | {r['optimo']:8.2f} | {ts} | "
              f"{'sí' if r['T_star_valida'] else 'no':>9}")

    # ---------------- 4) Tendencia entre n y T* ----------------
    print("\nTENDENCIA ENTRE n Y LA TEMPERATURA ÓPTIMA")
    mejores = [(r["n"], r["mejor_T"]) for r in resumen_final]
    print("  Mejor nivel de diseño por tamaño: " +
          ", ".join(f"n={n}: T0={t:.2f}" for n, t in mejores))
    validas = [(r["n"], r["T_star"]) for r in resumen_final if r["T_star_valida"]]
    if len(validas) >= 2:
        print("  T* confiables del modelo:        " +
              ", ".join(f"n={n}: {t:.2f}" for n, t in validas))
        ns, ts = zip(*validas)
        pendiente = np.polyfit(ns, ts, 1)[0]
        print(f"  Pendiente de T* contra n: {pendiente:+.3f} por ciudad adicional")
    else:
        print("  Hay menos de 2 valores de T* confiables: la tendencia se debe leer del "
              "mejor nivel de diseño y de las gráficas.")

    # ---------------- 5) Gráficas ----------------
    fig, axs = plt.subplots(1, len(TAMANOS), figsize=(5.5 * len(TAMANOS), 5), squeeze=False)
    rng_jit = np.random.default_rng(1)
    for ax, n in zip(axs[0], TAMANOS):
        df_n = df[df["n"] == n]
        m = modelos[n]
        ax.scatter(df_n["T"] + rng_jit.uniform(-0.01, 0.01, len(df_n)) * (T_MAX - T_MIN),
                   df_n["costo"], s=16, alpha=0.45, color="gray", label="Corridas")
        p = m["puntos"]
        T_p = [q["T"] for q in p]
        med = [q["mediana"] for q in p]
        err = [[q["mediana"] - q["lo"] for q in p], [q["hi"] - q["mediana"] for q in p]]
        ax.errorbar(T_p, med, yerr=err, fmt="o", color="tab:blue", capsize=5,
                    label="Mediana bootstrap (IC 95%)", zorder=3)
        margen = 0.05 * (T_MAX - T_MIN)
        tt = np.linspace(NIVELES_T[0] - margen, NIVELES_T[-1] + margen, 200)
        b0, b1, b2 = m["beta"]
        ax.plot(tt, b0 + b1 * tt + b2 * tt ** 2, color="tab:red", label="Ajuste cuadrático")
        if m["valido"]:
            ax.axvline(m["T_star"], color="tab:green", linestyle="--",
                       label=f"T* = {m['T_star']:.2f}")
        ax.axhline(instancias[n][1], color="black", linestyle=":", linewidth=1,
                   label="Óptimo exacto")
        pb2 = m["aj"]["p_coef"][2]
        r2 = m["aj"]["r2"]
        ax.set_title(f"n = {n}   (R² = {'n/d' if np.isnan(r2) else f'{r2:.2f}'}, p(b2) = "
                     f"{'n/d' if np.isnan(pb2) else f'{pb2:.3f}'})")
        ax.set_xlabel("Temperatura inicial T0")
        ax.set_ylabel("Costo (longitud del tour)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"Costo vs. temperatura (RSM)   |   r = {R:.4f}", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig("sa_rsm_curvas.png", dpi=150)
    print("\nGráfica guardada en sa_rsm_curvas.png")
    plt.show()