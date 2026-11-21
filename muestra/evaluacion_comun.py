"""Lógica compartida por los scripts de evaluación de extractores.

Une la anotación humana (``anotacion_manual_limpio.csv``) con las etiquetas de
los seis extractores registradas en ``muestra.json`` (KDF y el Request Parser
con cinco modelos), normaliza ambos vocabularios y calcula precisión, intervalo
de confianza y matriz de confusión según las ecuaciones (4.1)-(4.3) del
documento.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

DIR_MUESTRA = Path(__file__).resolve().parent
RUTA_MUESTRA = DIR_MUESTRA / "muestra.json"
RUTA_ANOTACION = DIR_MUESTRA / "anotacion_manual_limpio.csv"
DIR_FIGURAS = DIR_MUESTRA / "figuras"

CATEGORIAS = ("day", "hour", "sensor", "priority")
NOMBRE_CATEGORIA = {"day": "Day", "hour": "Hour", "sensor": "Sensor", "priority": "Priority"}

EXTRACTOR_KDF = "KDF"
PARSERS = ("gemma2:27b", "llama3.1:8b", "phi3.5:3.8b", "phi4:14b", "qwen2:7b")
EXTRACTORES = (EXTRACTOR_KDF, *PARSERS)

# Vocabularios de control en su forma corta (la que usa la anotación humana).
VOCABULARIO = {
    "day": ("today", "tomorrow", "the day after tomorrow", "in three days", "in four days"),
    "hour": ("morning", "mid-day", "afternoon", "evening", "overnight"),
    "sensor": ("VISUAL", "SAR", "TIR", "VNIR", "NIR"),
    "priority": ("low", "medium", "high"),
}

SIN_ETIQUETA = "NA"  # lectura humana: el texto no expresa la categoría o es ambiguo (∅)
SIN_ETIQUETA_EXTRACTOR = "sin etiqueta"  # el extractor no registró etiqueta en muestra.json
# Categorías en las que KDF puede quedar sin etiqueta (columna extra en su matriz).
CATEGORIAS_SIN_ETIQUETA_KDF = ("day", "hour")
_VALORES_VACIOS = {"", "∅", "na", "n/a", "none", "null", "vacio", "vacío", "sin etiqueta"}
_ALIAS_SENSOR = {"VIS": "VISUAL"}

Z_95 = 1.96

# Colores (paleta categórica validada, en orden fijo de espacios).
COLOR_EXTRACTOR = {
    EXTRACTOR_KDF: "#2a78d6",  # azul
    "gemma2:27b": "#eb6834",  # naranja
    "llama3.1:8b": "#1baf7a",  # aguamarina
    "phi3.5:3.8b": "#eda100",  # amarillo
    "phi4:14b": "#e87ba4",  # magenta
    "qwen2:7b": "#008300",  # verde
}
# Rampa de un solo tono por categoría: superficie -> tono -> tono oscuro.
RAMPA_CATEGORIA = {
    "day": ("#fcfcfb", "#2a78d6", "#0d366b"),  # azul
    "hour": ("#fcfcfb", "#eb6834", "#7a2c0c"),  # naranja
    "sensor": ("#fcfcfb", "#1baf7a", "#0b4f37"),  # aguamarina
    "priority": ("#fcfcfb", "#6250d6", "#2a1f73"),  # violeta
}
TINTA = "#0b0b0b"
TINTA_SECUNDARIA = "#52514e"


def configurar_salida() -> None:
    """Permite imprimir ∅, ± y tildes en la consola de Windows."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def argumentos_comunes(descripcion: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=descripcion)
    parser.add_argument(
        "--por-texto",
        action="store_true",
        help=(
            "Usa cada texto anotado una sola vez (385 unidades). Por defecto se "
            "usa cada aparición en la muestra (390 filas), aplicando la etiqueta "
            "humana a todas las apariciones de un mismo texto."
        ),
    )
    return parser.parse_args()


def sufijo_unidad(por_texto: bool) -> str:
    return "_por_texto" if por_texto else ""


def normalizar(categoria: str, valor: str | None, vacio: str = SIN_ETIQUETA) -> str:
    """Lleva una etiqueta (humana o de un extractor) al vocabulario corto.

    Un valor vacío se devuelve como ``vacio``: NA para la lectura humana y
    "sin etiqueta" para un extractor.
    """
    if valor is None or valor.strip().lower() in _VALORES_VACIOS:
        return vacio
    valor = valor.strip()
    if categoria == "priority":
        # Los extractores devuelven la descripción completa, p. ej.
        # "low priority background survey ..."; la primera palabra es el nivel.
        valor = valor.split()[0].lower()
    elif categoria == "sensor":
        valor = _ALIAS_SENSOR.get(valor.upper(), valor.upper())
    else:
        valor = valor.lower()
    if valor not in VOCABULARIO[categoria]:
        raise ValueError(f"Etiqueta '{valor}' fuera del vocabulario de {categoria}")
    return valor


@dataclass(frozen=True)
class Registro:
    """Un texto de la muestra con su lectura humana y las etiquetas de cada extractor."""

    id_anotacion: int
    humano: dict[str, str]
    extractores: dict[str, dict[str, str]]  # extractor -> categoría -> etiqueta


def _cargar_anotacion() -> dict[int, dict[str, str]]:
    with RUTA_ANOTACION.open(encoding="utf-8-sig", newline="") as fh:
        return {int(fila["id"]): fila for fila in csv.DictReader(fh)}


def cargar_registros(por_texto: bool = False) -> list[Registro]:
    """Devuelve los registros anotados.

    Por defecto, una entrada por fila de ``muestra.json`` cuyo texto fue anotado
    (un texto repetido en la muestra aparece tantas veces como fue muestreado).
    Con ``por_texto=True``, una entrada por texto anotado.
    """
    anotacion = _cargar_anotacion()
    with RUTA_MUESTRA.open(encoding="utf-8") as fh:
        filas = json.load(fh)["filas"]

    registros: list[Registro] = []
    vistos: set[int] = set()
    for fila in filas:
        id_ = fila["id_anotacion"]
        if id_ not in anotacion or (por_texto and id_ in vistos):
            continue
        anotada = anotacion[id_]
        if anotada["texto"].strip() != fila["texto"].strip():
            raise ValueError(f"El texto anotado {id_} no coincide con muestra.json")
        vistos.add(id_)

        humano = {c: normalizar(c, anotada[c]) for c in CATEGORIAS}
        vacio = SIN_ETIQUETA_EXTRACTOR
        extractores = {
            EXTRACTOR_KDF: {
                c: normalizar(c, fila["kdf"].get(f"predicted_{c}"), vacio) for c in CATEGORIAS
            }
        }
        for modelo in PARSERS:
            extractores[modelo] = {
                c: normalizar(c, fila["parsers"][modelo].get(c), vacio) for c in CATEGORIAS
            }
        registros.append(Registro(id_, humano, extractores))

    faltantes = set(anotacion) - vistos
    if faltantes:
        raise ValueError(f"Textos anotados sin fila en muestra.json: {sorted(faltantes)}")
    return registros


def es_acierto(humano: str, predicho: str) -> bool:
    """Ecuación (4.1): un texto sin etiqueta humana cuenta como fallo."""
    return humano != SIN_ETIQUETA and predicho == humano


@dataclass(frozen=True)
class Precision:
    aciertos: int
    n: int

    @property
    def p(self) -> float:
        return self.aciertos / self.n

    @property
    def errores(self) -> int:
        return self.n - self.aciertos

    @property
    def semiamplitud(self) -> float:
        """Ecuación (4.3): z * sqrt(p(1-p)/n)."""
        return Z_95 * math.sqrt(self.p * (1 - self.p) / self.n)

    @property
    def limite_inferior(self) -> float:
        return self.p - self.semiamplitud

    @property
    def limite_superior(self) -> float:
        return self.p + self.semiamplitud

    @property
    def aproximacion_valida(self) -> bool:
        """n·p̂ ≥ 5 y n(1−p̂) ≥ 5, es decir, al menos cinco aciertos y cinco errores."""
        return self.aciertos >= 5 and self.errores >= 5


def calcular_precision(registros: list[Registro], extractor: str, categoria: str) -> Precision:
    aciertos = sum(
        es_acierto(r.humano[categoria], r.extractores[extractor][categoria]) for r in registros
    )
    return Precision(aciertos=aciertos, n=len(registros))


def contar_sin_etiqueta(registros: list[Registro], categoria: str) -> int:
    return sum(r.humano[categoria] == SIN_ETIQUETA for r in registros)


@dataclass(frozen=True)
class MatrizConfusion:
    etiquetas_humanas: tuple[str, ...]  # filas
    etiquetas_predichas: tuple[str, ...]  # columnas
    conteos: tuple[tuple[int, ...], ...]

    @property
    def total(self) -> int:
        return sum(map(sum, self.conteos))


def matriz_confusion(registros: list[Registro], extractor: str, categoria: str) -> MatrizConfusion:
    """Filas: lectura humana (sin NA). Columnas: etiqueta del extractor,
    con "sin etiqueta" para KDF en Day y Hour."""
    vocab = VOCABULARIO[categoria]
    filas = vocab
    columnas = vocab
    if extractor == EXTRACTOR_KDF and categoria in CATEGORIAS_SIN_ETIQUETA_KDF:
        columnas = (*vocab, SIN_ETIQUETA_EXTRACTOR)
    indice_fila = {e: i for i, e in enumerate(filas)}
    indice_col = {e: i for i, e in enumerate(columnas)}
    conteos = [[0] * len(columnas) for _ in filas]
    for r in registros:
        if r.humano[categoria] == SIN_ETIQUETA:
            continue  # la fila NA no se muestra en la matriz
        predicho = r.extractores[extractor][categoria]
        if predicho not in indice_col:
            raise ValueError(f"{extractor} sin etiqueta en {categoria} (texto {r.id_anotacion})")
        conteos[indice_fila[r.humano[categoria]]][indice_col[predicho]] += 1
    return MatrizConfusion(filas, columnas, tuple(map(tuple, conteos)))


def slug(extractor: str) -> str:
    """Nombre de extractor apto para nombres de archivo: 'phi3.5:3.8b' -> 'phi3_5_3_8b'."""
    return extractor.lower().replace(":", "_").replace(".", "_")


def mapa_categoria(categoria: str):
    """Mapa de color secuencial (un solo tono) asignado a la categoría."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(f"rampa_{categoria}", RAMPA_CATEGORIA[categoria])


def color_texto_sobre(rgba) -> str:
    """Tinta (oscura o blanca) con mayor contraste sobre un color de fondo."""

    def lineal(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (lineal(c) for c in rgba[:3])
    luminancia = 0.2126 * r + 0.7152 * g + 0.0722 * b
    contraste_blanco = 1.05 / (luminancia + 0.05)
    contraste_negro = (luminancia + 0.05) / 0.05
    return "white" if contraste_blanco > contraste_negro else TINTA


def preparar_matplotlib():
    """Importa pyplot con un backend sin ventana y un estilo sobrio (tinta neutra)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "black",
            "text.color": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )
    DIR_FIGURAS.mkdir(exist_ok=True)
    return plt
