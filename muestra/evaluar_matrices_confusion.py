"""Matriz de confusión de cada extractor en cada categoría.

Filas: lectura humana. Columnas: etiqueta del extractor (incluida
"sin etiqueta" para KDF en Day y Hour). Cada celda muestra el conteo, con una
intensidad proporcional al porcentaje de su fila, en una rampa de un solo tono
distinta para cada categoría. Se guarda una imagen por
categoría, con los seis extractores en una cuadrícula de 2 × 3, en
``figuras/confusion_{categoria}.png``.

Uso:
    python evaluar_matrices_confusion.py [--por-texto]
"""

from __future__ import annotations

from evaluacion_comun import (
    CATEGORIAS,
    DIR_FIGURAS,
    EXTRACTORES,
    NOMBRE_CATEGORIA,
    MatrizConfusion,
    argumentos_comunes,
    cargar_registros,
    color_texto_sobre,
    configurar_salida,
    mapa_categoria,
    matriz_confusion,
    preparar_matplotlib,
    sufijo_unidad,
)


def porcentajes_por_fila(m: MatrizConfusion) -> list[list[float]]:
    """Proporción de cada celda respecto al total de su fila (0 si la fila está vacía)."""
    return [[v / sum(fila) if sum(fila) else 0.0 for v in fila] for fila in m.conteos]


def imprimir_matriz(extractor: str, categoria: str, m: MatrizConfusion) -> None:
    ancho = max(len(e) for e in (*m.etiquetas_humanas, *m.etiquetas_predichas)) + 2
    print(f"\n--- {NOMBRE_CATEGORIA[categoria]} · {extractor} (filas: humano, columnas: extractor)")
    print(" " * ancho + "".join(f"{e:>{ancho}}" for e in m.etiquetas_predichas))
    for etiqueta, fila in zip(m.etiquetas_humanas, m.conteos):
        print(f"{etiqueta:<{ancho}}" + "".join(f"{v:>{ancho}}" for v in fila))


def dibujar_panel(
    ax, extractor: str, categoria: str, m: MatrizConfusion, mostrar_filas: bool, mostrar_columnas: bool
):
    porcentajes = porcentajes_por_fila(m)
    mapa = mapa_categoria(categoria)
    imagen = ax.imshow(porcentajes, cmap=mapa, vmin=0, vmax=1, aspect="auto")

    for i, fila in enumerate(m.conteos):
        for j, conteo in enumerate(fila):
            ax.text(
                j, i, str(conteo), ha="center", va="center", fontsize=8,
                color=color_texto_sobre(mapa(porcentajes[i][j])),
            )

    rotacion = 35 if categoria in ("day", "hour") else 0
    ax.set_xticks(
        range(len(m.etiquetas_predichas)), m.etiquetas_predichas if mostrar_columnas else [],
        rotation=rotacion,
        ha="right" if rotacion else "center", rotation_mode="anchor", fontsize=8,
    )
    ax.set_yticks(range(len(m.etiquetas_humanas)), m.etiquetas_humanas if mostrar_filas else [],
                  fontsize=8)
    ax.set_title(extractor, fontsize=10, fontweight="bold")
    if mostrar_filas:
        ax.set_ylabel("Lectura humana", fontsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    return imagen


def guardar_categoria(plt, categoria: str, matrices: dict[str, MatrizConfusion], n: int, sufijo: str):
    ancho = 15 if categoria == "day" else 13
    fig, ejes = plt.subplots(2, 3, figsize=(ancho, 8.5))
    imagen = None
    for k, (ax, extractor) in enumerate(zip(ejes.flat, EXTRACTORES)):
        imagen = dibujar_panel(
            ax, extractor, categoria, matrices[extractor],
            mostrar_filas=k % 3 == 0,  # columna izquierda
            mostrar_columnas=k >= 3,  # fila de abajo
        )

    fig.suptitle(f"Matrices de confusión · {NOMBRE_CATEGORIA[categoria]} (n = {n})",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 0.90, 0.96))
    eje_barra = fig.add_axes((0.915, 0.15, 0.012, 0.7))
    barra = fig.colorbar(imagen, cax=eje_barra)
    barra.set_label("Porcentaje de la fila", fontsize=9)
    barra.set_ticks([0, 0.25, 0.5, 0.75, 1], labels=["0 %", "25 %", "50 %", "75 %", "100 %"])
    barra.ax.tick_params(labelsize=8)

    ruta = DIR_FIGURAS / f"confusion_{categoria}{sufijo}.png"
    fig.savefig(ruta, dpi=200)
    plt.close(fig)
    return ruta


def main() -> None:
    configurar_salida()
    args = argumentos_comunes(__doc__.splitlines()[0])
    registros = cargar_registros(por_texto=args.por_texto)
    sufijo = sufijo_unidad(args.por_texto)
    plt = preparar_matplotlib()
    print(f"Unidades evaluadas: {len(registros)}")

    rutas = []
    for categoria in CATEGORIAS:
        matrices = {e: matriz_confusion(registros, e, categoria) for e in EXTRACTORES}
        for extractor, m in matrices.items():
            imprimir_matriz(extractor, categoria, m)
        rutas.append(guardar_categoria(plt, categoria, matrices, len(registros), sufijo))

    print("\nFiguras guardadas en:")
    for ruta in rutas:
        print(f"  {ruta}")


if __name__ == "__main__":
    main()
