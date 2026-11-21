"""Precisión de cada extractor frente a la lectura humana, con IC del 95 %.

Imprime, por categoría, la precisión estimada y su intervalo de confianza para
KDF y para el Request Parser con cada modelo, y genera una figura de punto e
intervalo con un panel por categoría en ``figuras/``.

Uso:
    python evaluar_precision.py [--por-texto]
"""

from __future__ import annotations

import math

from matplotlib.lines import Line2D

from evaluacion_comun import (
    CATEGORIAS,
    COLOR_EXTRACTOR,
    DIR_FIGURAS,
    EXTRACTOR_KDF,
    EXTRACTORES,
    NOMBRE_CATEGORIA,
    PARSERS,
    TINTA,
    TINTA_SECUNDARIA,
    Precision,
    argumentos_comunes,
    calcular_precision,
    cargar_registros,
    configurar_salida,
    contar_sin_etiqueta,
    preparar_matplotlib,
    sufijo_unidad,
)

BLANCO = "white"


def imprimir_tabla(categoria: str, resultados: dict[str, Precision], sin_etiqueta: int) -> None:
    print(f"\n=== {NOMBRE_CATEGORIA[categoria]}  (textos anotados como NA: {sin_etiqueta}) ===")
    print(f"{'Extractor':<13} {'Aciertos':>9} {'Precisión':>10} {'IC 95 %':>19} {'±':>7}  Aprox. normal")
    for extractor in EXTRACTORES:
        r = resultados[extractor]
        intervalo = f"[{r.limite_inferior:.3f}, {r.limite_superior:.3f}]"
        valida = "sí" if r.aproximacion_valida else f"no ({r.errores} errores, {r.aciertos} aciertos)"
        print(
            f"{extractor:<13} {f'{r.aciertos}/{r.n}':>9} {r.p:>10.3f} {intervalo:>19} "
            f"{r.semiamplitud:>7.3f}  {valida}"
        )


def dibujar_panel(ax, categoria: str, resultados: dict[str, Precision], x_min: float) -> None:
    kdf = resultados[EXTRACTOR_KDF]
    parsers = sorted(PARSERS, key=lambda m: (-resultados[m].p, m))
    orden = [EXTRACTOR_KDF, *parsers]
    # KDF separado de los parsers por un pequeño hueco.
    posiciones = [0.0] + [1.5 + i for i in range(len(parsers))]

    ax.axvline(kdf.p, color=COLOR_EXTRACTOR[EXTRACTOR_KDF], linestyle=":", linewidth=1.2, zorder=1)

    for extractor, y in zip(orden, posiciones):
        r = resultados[extractor]
        color = COLOR_EXTRACTOR[extractor]
        li, ls = r.limite_inferior, r.limite_superior  # ecuación (4.3), sin recorte
        ax.hlines(y, li, ls, color=color, linewidth=2, zorder=2)
        ax.vlines([li, ls], y - 0.12, y + 0.12, color=color, linewidth=1.4, zorder=2)
        ax.plot(
            r.p,
            y,
            marker="s" if extractor == EXTRACTOR_KDF else "o",
            markersize=9 if extractor == EXTRACTOR_KDF else 8,
            markerfacecolor=color if r.aproximacion_valida else BLANCO,
            markeredgecolor=color,
            markeredgewidth=1.6,
            linestyle="none",
            zorder=3,
        )
        ax.text(ls + 0.012, y, f"{r.p:.3f}", va="center", ha="left", fontsize=9, color=TINTA)

    ax.set_yticks(posiciones, orden)
    ax.set_ylim(posiciones[-1] + 0.7, -0.7)
    ax.set_xlim(x_min, 1.09)
    ax.set_xticks([t / 10 for t in range(math.ceil(x_min * 10), 11)])
    ax.set_title(NOMBRE_CATEGORIA[categoria], fontweight="bold")
    ax.tick_params(axis="y", length=0)


def generar_figura(resultados: dict[str, dict[str, Precision]], n: int, sufijo: str) -> list:
    plt = preparar_matplotlib()

    limite_inf = min(r.limite_inferior for res in resultados.values() for r in res.values())
    x_min = math.floor(limite_inf * 10) / 10 - 0.02

    fig, ejes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for ax, categoria in zip(ejes.flat, CATEGORIAS):
        dibujar_panel(ax, categoria, resultados[categoria], x_min)
    for ax in ejes[1]:
        ax.set_xlabel("Precisión frente a la lectura humana")

    # La leyenda explica las formas en tinta neutra; el color identifica al
    # extractor y su nombre ya aparece en cada fila.
    gris = TINTA_SECUNDARIA
    leyenda = [
        Line2D([], [], marker="s", color=gris, markersize=8, linestyle="none", label="KDF (referencia)"),
        Line2D([], [], marker="o", color=gris, markersize=7, linestyle="none", label="Request Parser"),
        Line2D([], [], color=gris, linewidth=2, label="IC 95 %"),
        Line2D([], [], color=COLOR_EXTRACTOR[EXTRACTOR_KDF], linewidth=1.2, linestyle=":",
               label="Precisión de KDF"),
        Line2D(
            [], [], marker="o", markerfacecolor=BLANCO, markeredgecolor=gris, markeredgewidth=1.6,
            markersize=7, linestyle="none", label="Menos de 5 aciertos o 5 errores",
        ),
    ]
    fig.legend(handles=leyenda, loc="lower center", ncol=5, frameon=False, fontsize=9)
    fig.suptitle(f"Precisión por extractor y categoría (n = {n})", fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 1))

    rutas = [DIR_FIGURAS / f"precision_intervalos{sufijo}.{ext}" for ext in ("png", "pdf")]
    for ruta in rutas:
        fig.savefig(ruta, dpi=200)
    plt.close(fig)
    return rutas


def main() -> None:
    configurar_salida()
    args = argumentos_comunes(__doc__.splitlines()[0])
    registros = cargar_registros(por_texto=args.por_texto)
    n = len(registros)
    unidad = "textos distintos" if args.por_texto else "apariciones en la muestra"
    print(f"Unidades evaluadas: {n} ({unidad}). IC 95 %: p̂ ± 1.96·sqrt(p̂(1−p̂)/n).")

    resultados = {
        c: {e: calcular_precision(registros, e, c) for e in EXTRACTORES} for c in CATEGORIAS
    }
    for categoria in CATEGORIAS:
        imprimir_tabla(categoria, resultados[categoria], contar_sin_etiqueta(registros, categoria))

    rutas = generar_figura(resultados, n, sufijo_unidad(args.por_texto))
    print("\nFigura guardada en:")
    for ruta in rutas:
        print(f"  {ruta}")


if __name__ == "__main__":
    main()
