import io
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    print("Pega aquí tu tabla (cuando termines, pulsa Enter y luego Ctrl+D):")
    texto_pegado = sys.stdin.read()

    # Filtrar líneas vacías o títulos sueltos como "COMPARACIÓN DE MODELOS"
    lineas = [
        l.strip() for l in texto_pegado.strip().splitlines()
        if l.strip() and "COMPARACIÓN" not in l.upper()
    ]

    if not lineas:
        print("No se detectaron datos.")
        return

    # Convertir el texto alineado por espacios a DataFrame automáticamente
    df = pd.read_csv(io.StringIO("\n".join(lineas)), sep=r"\s+")

    # --------------------------------------------------
    # ORDENAR SEGÚN REQUERIMIENTO:
    # 1. Escenarios: real -> combined -> synthetic
    # 2. Modelos en parejas: word2vec -> bow
    # --------------------------------------------------
    escenario_orden = {"real": 1, "combined": 2, "synthetic": 3}
    modelo_orden = {"word2vec": 1, "bow": 2}

    df["esc_rank"] = df["scenario"].map(escenario_orden)
    df["mod_rank"] = df["model"].map(modelo_orden)

    df = df.sort_values(by=["esc_rank", "mod_rank"]).drop(columns=["esc_rank", "mod_rank"]).reset_index(drop=True)

    # Formatear columnas numéricas a 4 decimales para la tabla visual
    df_display = df.copy()
    num_cols = df_display.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        df_display[col] = df_display[col].map(lambda x: f"{x:.4f}")

    # --------------------------------------------------
    # 1. GENERAR IMAGEN DE LA TABLA ORGANIZADA
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=(13, len(df_display) * 0.6 + 1.5))
    ax.axis("off")

    tabla = ax.table(
        cellText=df_display.values,
        colLabels=df_display.columns,
        loc="center",
        cellLoc="center"
    )
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(9.5)
    tabla.scale(1, 1.6)

    # Estilizar cabecera y filas alternas
    for (row, col), cell in tabla.get_celld().items():
        cell.set_edgecolor("#d0d7de")
        if row == 0:
            cell.set_facecolor("#1f2937")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f6f8fa")
        else:
            cell.set_facecolor("white")

    plt.title("Comparación de Modelos (Ordenada)", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig("tabla_modelos.png", dpi=300, bbox_inches="tight")
    plt.show()

    # --------------------------------------------------
    # 2. GENERAR GRÁFICO COMPARATIVO ORDENADO
    # --------------------------------------------------
    if {"model", "scenario", "accuracy", "f1_macro"}.issubset(df.columns):
        df["etiqueta"] = df["model"] + "\n(" + df["scenario"] + ")"
        x = np.arange(len(df))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 5))
        b1 = ax.bar(x - width/2, df["accuracy"], width, label="Accuracy", color="#4f81bd")
        b2 = ax.bar(x + width/2, df["f1_macro"], width, label="F1 Macro", color="#c0504d")

        ax.set_ylabel("Puntuación")
        ax.set_title("Rendimiento por Modelo y Escenario (Ordenado)")
        ax.set_xticks(x)
        ax.set_xticklabels(df["etiqueta"])
        ax.set_ylim(0, 1.0)
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        for barra in list(b1) + list(b2):
            h = barra.get_height()
            ax.annotate(f"{h:.3f}",
                        xy=(barra.get_x() + barra.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5, fontweight="bold")

        plt.tight_layout()
        plt.savefig("grafico_modelos.png", dpi=300)
        plt.show()

    print("\nArchivos guardados: 'tabla_modelos.png' y 'grafico_modelos.png'")

if __name__ == "__main__":
    main()