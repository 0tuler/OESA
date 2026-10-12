import csv
from sys import argv
from typing import Dict, List

import matplotlib.pyplot as plt


def pegar_resultados(nome: str) -> List[Dict[str, str]]:
    """
    Abre o arquivo e pega os resultados
    """
    ret = []
    with open(nome, "r", newline="") as file:
        reader = csv.DictReader(file)
        for linha in reader:
            ret.append(linha)

    return ret


def grafico_gflops(
    ntasks: List[str], values: List[float], PICO_TEORICO: float = 1612.8
):
    fig, ax = plt.subplots()
    xcords = [int(n) for n in ntasks]

    # Desempenho obtido
    ax.plot(xcords, values, marker="D", color="green", linewidth=2.0, label="Obtido")

    # Pico Teórico do Santos Dumont (Xeon Gold 6252)
    # Dependendo da eficiência do seu código, essa linha pode achatar muito o gráfico.
    # Se ficar visualmente ruim, comente a linha abaixo.
    ax.axhline(
        y=PICO_TEORICO,
        color="red",
        linestyle="--",
        label=f"Pico Teórico ({PICO_TEORICO} GFLOP/s)",
    )

    ax.set_title("Desempenho (GFLOP/s) x N")
    ax.set_xlabel("Número de Threads (N)")
    ax.set_ylabel("Desempenho (GFLOP/s)")

    configurar_eixo_x(ax, xcords)
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.6)

    fig.savefig("desempenho_gflops.png")
    plt.close(fig)


def configurar_eixo_x(ax, xcords):
    """
    Garante que a escala visual do eixo X seja proporcional ao número real de threads.
    Evita que o espaço entre 1-2 pareça igual ao espaço entre 16-32.
    """
    ax.set_xticks(xcords)
    ax.set_xticklabels([str(x) for x in xcords])


def grafico_speedup(ntasks: List[str], values: List[float]):
    fig, ax = plt.subplots()
    xcords = [int(n) for n in ntasks]

    # Plota a linha dos resultados obtidos
    ax.plot(xcords, values, marker="o", linewidth=2.0, label="Obtido")

    # Plota a linha de escala 1:1 (Speedup Ideal)
    ax.plot(xcords, xcords, color="red", linestyle="--", label="Ideal (Linear)")

    ax.set_title("Speedup x N")
    ax.set_xlabel("Número de Threads (N)")
    ax.set_ylabel("Speedup")

    configurar_eixo_x(ax, xcords)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)

    fig.savefig("speedup.png")
    plt.close(fig)


def grafico_eficiencia(ntasks: List[str], values: List[float]):
    fig, ax = plt.subplots()
    xcords = [int(n) for n in ntasks]

    # Plota a eficiência calculada
    ax.plot(xcords, values, marker="o", linewidth=2.0, label="Obtida")

    # Plota a linha Ideal (1.0 constante)
    ax.axhline(y=1.0, color="red", linestyle="--", label="Ideal (100%)")

    ax.set_title("Eficiência x N")
    ax.set_xlabel("Número de Threads (N)")
    ax.set_ylabel("Eficiência")

    configurar_eixo_x(ax, xcords)
    ax.set_ylim(0, 1.15)  # Dá uma folga acima do 1.0 para o gráfico respirar
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)

    fig.savefig("eficiencia.png")
    plt.close(fig)


def grafico_medias(ntasks: List[str], values: List[float]):
    fig, ax = plt.subplots()

    ax.bar(ntasks, values, color="skyblue", edgecolor="black")
    ax.set_title("Tempo Médio x N")
    ax.set_xlabel("Número de Threads (N)")
    ax.set_ylabel("Tempo (Segundos)")

    ax.grid(axis="y", linestyle="--", alpha=0.6)

    fig.savefig("media_tempo.png")
    plt.close(fig)


def grafico_metrica_hardware(
    ntasks: List[str], values: List[float], nome_metrica: str, arquivo_saida: str
):
    """
    Plota métricas genéricas de hardware (como cache misses, ciclos, CPI)
    """
    fig, ax = plt.subplots()
    xcords = [int(n) for n in ntasks]

    ax.plot(xcords, values, marker="s", color="purple", linewidth=2.0)

    ax.set_title(f"{nome_metrica} x N")
    ax.set_xlabel("Número de Threads (N)")
    ax.set_ylabel(f"{nome_metrica}")

    configurar_eixo_x(ax, xcords)
    ax.grid(True, linestyle="--", alpha=0.6)

    fig.savefig(arquivo_saida)
    plt.close(fig)


if __name__ == "__main__":
    if len(argv) > 1 and argv[1] != "":
        arquivo = argv[1]
        resultados = pegar_resultados(arquivo)

        for line in resultados:
            nome_linha = line.get("N", "")

            # Filtra dinamicamente as colunas que representam o número de threads
            ntasks = [k for k in line.keys() if k != "N" and k.isdigit()]

            # Pula linhas de processamento que não possuem dados formatáveis
            if not ntasks:
                continue

            try:
                # Extrai e converte os valores
                values = [
                    float(line[n].replace(",", "."))
                    for n in ntasks
                    if line[n].strip() != ""
                ]
            except ValueError:
                continue

            # Garante que as listas possuem o mesmo tamanho antes de plotar
            if len(values) != len(ntasks):
                ntasks = ntasks[: len(values)]

            if not values:
                continue

            # Geração dos gráficos
            if nome_linha == "Média":
                grafico_medias(ntasks, values)
            elif nome_linha == "Speedup":
                grafico_speedup(ntasks, values)
            elif nome_linha == "Eficiencia":
                grafico_eficiencia(ntasks, values)
            elif nome_linha == "GFLOP/s":
                grafico_gflops(ntasks, values, 1612.8)
            # Captura dinâmica de métricas de Hardware injetadas no CSV
            elif any(
                keyword in nome_linha.lower()
                for keyword in [
                    "misses",
                    "cache",
                    "cpi",
                    "cycles",
                    "instructions",
                    "bandwidth",
                ]
            ):
                nome_seguro = nome_linha.replace(" ", "_").replace("/", "_")
                grafico_metrica_hardware(
                    ntasks, values, nome_linha, f"{nome_seguro}.png"
                )
