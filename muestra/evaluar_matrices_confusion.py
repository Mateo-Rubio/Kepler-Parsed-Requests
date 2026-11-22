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

# Tamaños de letra (modificables).
TAMANO_NUMEROS = 12  # conteo dentro de cada celda
TAMANO_ETIQUETAS = 12  # etiquetas de los ejes x e y
TAMANO_TITULO_PANEL = 14  # nombre del extractor sobre cada matriz

# Etiquetas abreviadas solo para mostrar en las figuras (los datos no cambian).
ETIQUETA_VISIBLE = {"the day after tomorrow": "after tomorrow"}


def visibles(etiquetas) -> list[str]:
    return [ETIQUETA_VISIBLE.get(e, e) for e in etiquetas]


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
    ax.imshow(porcentajes, cmap=mapa, vmin=0, vmax=1, aspect="auto")

    for i, fila in enumerate(m.conteos):
        for j, conteo in enumerate(fila):
            ax.text(
                j, i, str(conteo), ha="center", va="center", fontsize=TAMANO_NUMEROS,
                color=color_texto_sobre(mapa(porcentajes[i][j])),
            )

    rotacion = 35 if categoria in ("day", "hour") else 0
    ax.set_xticks(
        range(len(m.etiquetas_predichas)), visibles(m.etiquetas_predichas) if mostrar_columnas else [],
        rotation=rotacion,
        ha="right" if rotacion else "center", rotation_mode="anchor", fontsize=TAMANO_ETIQUETAS,
    )
    ax.set_yticks(range(len(m.etiquetas_humanas)), visibles(m.etiquetas_humanas) if mostrar_filas else [],
                  fontsize=TAMANO_ETIQUETAS)
    ax.set_title(extractor, fontsize=TAMANO_TITULO_PANEL, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)


def guardar_categoria(plt, categoria: str, matrices: dict[str, MatrizConfusion], sufijo: str):
    ancho = 15 if categoria == "day" else 13
    fig, ejes = plt.subplots(2, 3, figsize=(ancho, 8.5))
    for k, (ax, extractor) in enumerate(zip(ejes.flat, EXTRACTORES)):
        dibujar_panel(
            ax, extractor, categoria, matrices[extractor],
            mostrar_filas=k % 3 == 0,  # columna izquierda
            mostrar_columnas=k >= 3,  # fila de abajo
        )

    fig.tight_layout()

    ruta = DIR_FIGURAS / f"confusion_{categoria}{sufijo}.pdf"
    fig.savefig(ruta)
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
        rutas.append(guardar_categoria(plt, categoria, matrices, sufijo))

    print("\nFiguras guardadas en:")
    for ruta in rutas:
        print(f"  {ruta}")


if __name__ == "__main__":
    main()
