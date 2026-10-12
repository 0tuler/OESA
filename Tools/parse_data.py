import csv
import os
import re
import sys
from collections import defaultdict
from sys import argv
from typing import Any, Dict, List, Tuple

import numpy as np
import scipy.stats as st

# Pico Teórico em GFLOP/s (Intel Xeon Gold 6252 - 48 Cores @ 2.1GHz AVX-512)
# Fórmula: 48 Cores * 2.1 GHz * 2 (FMA) * 4 (AVX-512 double) * 2 (ops) = 1612.8
PICO_TEORICO_GFLOPS = 1612.8


class PerfBlock:
    def __init__(
        self,
        scalar_double,
        sse,
        avx2,
        avx512,
        instructions,
        cache_misses,
        LLC_misses,
        page_faults,
        context_switchs,
        migrations,
        time_elapsed,
        user,
        sys,
    ) -> None:
        # Convertemos para float já no construtor para facilitar a média depois
        self.scalar = float(scalar_double)
        self.sse = float(sse)
        self.avx2 = float(avx2)
        self.avx512 = float(avx512)
        self.instructions = float(instructions)
        self.cache_misses = float(cache_misses)
        self.llc_misses = float(LLC_misses)
        self.page_faults = float(page_faults)
        self.context_switchs = float(context_switchs)
        self.migrations = float(migrations)
        self.time_elapsed = float(time_elapsed)
        self.user = float(user)
        self.sys = float(sys)


def parse_perf(file: str) -> Dict[int, List[PerfBlock]]:
    """
    Faz parsing dos dados inseridos pelo perf
    """
    if not os.path.exists(file):
        print(
            f"Aviso: {os.path.basename(file)} ausente. Preenchendo hardware com zeros."
        )
        return {}

    pattern = re.compile(
        r"^TH:\s*(?P<threads>\d+).*?"
        r"^\s*(?P<scalar_double>[\d,\.]+)\s+fp_arith_inst_retired\.scalar_double.*?"
        r"^\s*(?P<sse>[\d,\.]+)\s+fp_arith_inst_retired\.128b_packed_double.*?"
        r"^\s*(?P<avx2>[\d,\.]+)\s+fp_arith_inst_retired\.256b_packed_double.*?"
        r"^\s*(?P<avx512>[\d,\.]+)\s+fp_arith_inst_retired\.512b_packed_double.*?"
        r"^\s*(?P<instructions>[\d,\.]+)\s+instructions.*?"
        r"^\s*(?P<cache_misses>[\d,\.]+)\s+cache-misses.*?"
        r"^\s*(?P<LLC_load_misses>[\d,\.]+)\s+LLC-load-misses.*?"
        r"^\s*(?P<page_faults>[\d,\.]+)\s+page-faults.*?"
        r"^\s*(?P<context_switches>[\d,\.]+)\s+cs.*?"
        r"^\s*(?P<migrations>[\d,\.]+)\s+migrations.*?"
        r"^\s*(?P<time_elapsed>[\d\.]+)\s+seconds time elapsed.*?"
        r"^\s*(?P<user>[\d\.]+)\s+seconds user.*?"
        r"^\s*(?P<sys>[\d\.]+)\s+seconds sys",
        re.MULTILINE | re.DOTALL,
    )
    with open(file, "r") as fd:
        content = fd.read()

        blocks = {}

        for match in pattern.finditer(content):
            block = PerfBlock(
                match.group("scalar_double").replace(",", ""),
                match.group("sse").replace(",", ""),
                match.group("avx2").replace(",", ""),
                match.group("avx512").replace(",", ""),
                match.group("instructions").replace(",", ""),
                match.group("cache_misses").replace(",", ""),
                match.group("LLC_load_misses").replace(",", ""),
                match.group("page_faults").replace(",", ""),
                match.group("context_switches").replace(",", ""),
                match.group("migrations").replace(",", ""),
                match.group("time_elapsed"),
                match.group("user"),
                match.group("sys"),
            )

            threads = int(match.group("threads"))
            if threads in blocks:
                blocks[threads].append(block)
            else:
                blocks[threads] = [block]

        return blocks


def calcular_medias(blocks: Dict[int, List[PerfBlock]]) -> Dict[int, PerfBlock]:
    medias = {}

    for threads, list_of_blocks in blocks.items():
        n = len(list_of_blocks)
        medias[threads] = PerfBlock(
            sum(b.scalar for b in list_of_blocks) / n,
            sum(b.sse for b in list_of_blocks) / n,
            sum(b.avx2 for b in list_of_blocks) / n,
            sum(b.avx512 for b in list_of_blocks) / n,
            sum(b.instructions for b in list_of_blocks) / n,
            sum(b.cache_misses for b in list_of_blocks) / n,
            sum(b.llc_misses for b in list_of_blocks) / n,
            sum(b.page_faults for b in list_of_blocks) / n,
            sum(b.context_switchs for b in list_of_blocks) / n,
            sum(b.migrations for b in list_of_blocks) / n,
            sum(b.time_elapsed for b in list_of_blocks) / n,
            sum(b.user for b in list_of_blocks) / n,
            sum(b.sys for b in list_of_blocks) / n,
        )

    return medias


# Considera o tempo de execução dado pelo tempo de usuario no perf, esse é tendencioso pois não considera o tempo de chamadas de kernel (SYSCALL), mas foi adicionado caso sirva para alguma aplicação
def calc_gflops_user(block: PerfBlock, threads):
    return (((block.scalar) + (block.avx2 * 4) + (block.avx512 * 8)) * 2) / (
        block.user / threads * 1e9
    )


# Considera o tempo de execução dado pelo perf
def calc_gflops_elapsed(block: PerfBlock):
    return (((block.scalar) + (block.avx2 * 4) + (block.avx512 * 8)) * 2) / (
        block.time_elapsed * 1e9
    )


# Considera o tempo de execução dado pela saida do programa
def calc_gflops(block: PerfBlock, tempo):
    return (((block.scalar) + (block.avx2 * 4) + (block.avx512 * 8)) * 2) / (
        tempo * 1e9
    )


def carregar_dados(caminho: str) -> Dict[str, np.ndarray]:
    grupos = defaultdict(list)
    with open(caminho, "r", newline="") as file:
        reader = csv.DictReader(file)
        for linha in reader:
            grupos[linha["N"]].append(float(linha["Tempo"]))
    return {nt: np.array(tempos) for nt, tempos in grupos.items()}


def fmt(valor: Any) -> str:
    if isinstance(valor, (int, float, np.floating)):
        # Formata o número com até 6 casas decimais ("3.140000")
        # rstrip("0") tira os zeros extras do final ("3.14")
        # rstrip(".") tira o ponto se não sobrar nenhum decimal ("5." vira "5")
        # replace troca o ponto pela vírgula para o padrão em português
        return f"{valor:.6f}".rstrip("0").rstrip(".").replace(".", ",")
    return str(valor)


#
##
### Funções de polimento de amostras
##
#


def polir_amostras_percentil(grupos: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Remove os 33% piores (mais altos) tempos de execução, mantendo os 67% melhores.
    Ideal para cenários de HPC onde o ruído ambiental apenas adiciona atrasos.
    """
    grupos_polidos = {}
    for nt, tempos in grupos.items():
        n = len(tempos)

        # ignora o polimento se a amostra for muito pequena
        if n <= 4:
            grupos_polidos[nt] = tempos
            continue

        # Calcula o limite no percentil 66.67% (mantém os melhores 2/3 da amostra)
        limite = np.percentile(tempos, 66.67)

        # A máscara seleciona apenas os tempos menores ou iguais ao limite
        mask = tempos <= limite
        grupos_polidos[nt] = tempos[mask]

    return grupos_polidos


def polir_amostras_IQR(grupos: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    grupos_polidos = {}
    for nt, tempos in grupos.items():
        n = len(tempos)

        # ignora o polimento se a amostra for muito pequena
        if n <= 4:
            grupos_polidos[nt] = tempos
            continue

        q1 = np.percentile(tempos, 25)
        q3 = np.percentile(tempos, 75)

        iqr = q3 - q1

        lim_inf = q1 - (1.5 * iqr)
        lim_sup = q3 + (1.5 * iqr)

        mask = (tempos >= lim_inf) & (tempos <= lim_sup)

        grupos_polidos[nt] = tempos[mask]
    return grupos_polidos


def polir_amostras_unilateral(grupos: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    grupos_polidos = {}
    for nt, tempos in grupos.items():
        n = len(tempos)

        # ignora o polimento se a amostra for muito pequena
        if n <= 4:
            grupos_polidos[nt] = tempos
            continue

        q1 = np.percentile(tempos, 25)
        q3 = np.percentile(tempos, 75)
        iqr = q3 - q1

        # Limite superior com fator 3.0 para pegar apenas atrasos severos (cauda direita)
        lim_sup = q3 + (3.0 * iqr)

        # Mantém tudo que for menor ou igual ao limite superior (não corta os tempos baixos)
        mask = tempos <= lim_sup
        grupos_polidos[nt] = tempos[mask]

    return grupos_polidos


def polir_amostras_mad(grupos: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    grupos_polidos = {}
    for nt, tempos in grupos.items():
        n = len(tempos)

        # ignora o polimento se a amostra for muito pequena
        if n <= 4:
            grupos_polidos[nt] = tempos
            continue

        mediana = np.median(tempos)
        desvios_absolutos = np.abs(tempos - mediana)
        mad = np.median(desvios_absolutos)

        # Proteção contra divisão por zero em casos de variância nula
        if mad == 0:
            mad = 1e-9

        # Calcula o escore Z modificado
        m_scores = (0.6745 * (tempos - mediana)) / mad

        # O critério de Chauvenet adaptado para MAD, descarta |Z| > 3.5
        mask = np.abs(m_scores) <= 3.5
        grupos_polidos[nt] = tempos[mask]

    return grupos_polidos


#
##
### Funções de polimento de amostras
##
#


# Processa as métricas principais
def processar_metricas(
    grupos: Dict[str, np.ndarray], significancia: float
) -> Tuple[Dict[str, Dict[str, Any]], int]:
    metricas_por_nt = {}
    maior_amostra = 0

    for nt, tempos in grupos.items():
        n = len(tempos)
        maior_amostra = max(maior_amostra, n)

        if n == 0:
            continue

        media = np.mean(tempos)
        mediana = np.median(tempos)
        val_min = np.min(tempos)
        desvio = np.std(tempos, ddof=1) if n > 1 else 0.0
        t_crit = st.t.ppf(1 - (significancia / 2), n - 1) if n > 1 else 0.0
        erro_padrao = desvio / np.sqrt(n) if n > 0 else 0.0
        valor_confianca = t_crit * erro_padrao

        metricas_por_nt[nt] = {
            "Tamanho da amostra": n,
            "Média": media,
            "Mediana": mediana,
            "Valor Mínimo": val_min,
            "Desvio padrão": desvio,
            "Nível de significância": significancia,
            "Valor critico": t_crit,
            "Valor de confiança": valor_confianca,
            "Lim. inf. intervalo de confiança": media - valor_confianca,
            "Lim. sup. intervalo de confiança": media + valor_confianca,
            "Intervalo confiança percentual (1/2)": (
                (valor_confianca / media * 100) if media != 0 else 0.0
            ),
        }
    return metricas_por_nt, maior_amostra


def criar_tabela(nome_entrada: str, nome_saida: str) -> int:
    significancia = 0.05  # significancia configurada para todo o projeto
    grupos_brutos = carregar_dados(nome_entrada)
    # Extrai a pasta onde o arquivo CSV de entrada está localizado
    base_path = os.path.dirname(os.path.abspath(nome_entrada))

    # Junta a pasta com o nome perf.txt
    caminho_perf = os.path.join(base_path, "perf.txt")
    blocks = parse_perf(caminho_perf)
    blocks = calcular_medias(blocks)

    if not grupos_brutos:
        return 1

    # grupos_polidos = polir_amostras_IQR(grupos_brutos)
    # grupos_polidos = polir_amostras_mad(grupos_brutos)
    # grupos_polidos = polir_amostras_unilateral(grupos_brutos)
    # Apresentou melhor resultados o para o tipo de dados que foi coletado com esse programa, mas pode ser alterado com base na necessidade como está acima
    grupos_polidos = polir_amostras_percentil(grupos_brutos)
    metricas, maior_amostra = processar_metricas(grupos_polidos, significancia)
    ntasks_unicas = list(metricas.keys())

    n_base = (
        "1"
        if "1" in metricas
        else str(min(int(nt) for nt in ntasks_unicas if nt.isdigit()))
    )

    media_base = metricas[n_base]["Média"]

    # Removido, rascunho mantido para depois verificar se com correções pode valer a pena, mas a métrica mantida foi a do gflops pelo perf
    # Fórmulas de GFLOP/s (2N^3 operações / tempo * 10^9)
    # total_ops = 2.0 * (tamanho_matriz**3)

    for nt in ntasks_unicas:
        media_atual = metricas[nt]["Média"]
        speedup = media_base / media_atual
        metricas[nt]["Speedup"] = speedup
        try:
            metricas[nt]["Eficiencia"] = speedup / int(nt)
        except ValueError:
            metricas[nt]["Eficiencia"] = 0.0

        # Calcula os GFLOP/s com base no tempo médio
        # if media_atual > 0:
        #    gflops = (total_ops / media_atual) / 1e9
        # else:
        #    gflops = 0.0

        # metricas[nt]["GFLOP/s"] = gflops
        # metricas[nt]["Percentual pico (%)"] = (gflops / PICO_TEORICO_GFLOPS) * 100.0

        # insere zeros caso seja vazio
        b = blocks.get(int(nt), PerfBlock(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))

        metricas[nt]["Média instruções escalares"] = b.scalar
        metricas[nt]["Média instruções SSE"] = b.sse
        metricas[nt]["Média instruções AVX2"] = b.avx2
        metricas[nt]["Média instruções AVX512"] = b.avx512
        metricas[nt]["Média instruções"] = b.instructions
        metricas[nt]["Média cache miss"] = b.cache_misses
        metricas[nt]["Média llc load miss"] = b.llc_misses
        metricas[nt]["Média page faults"] = b.page_faults
        metricas[nt]["Média context switchs"] = b.context_switchs
        metricas[nt]["Média migrations"] = b.migrations
        metricas[nt]["Média perf time elapsed"] = b.time_elapsed
        metricas[nt]["Média perf time user"] = b.user
        metricas[nt]["Média perf time sys"] = b.sys

        # Se o tempo de execução for zero ou menor que 0, então ocorreu um erro ou o programa não foi executado
        gflops_instrucoes = calc_gflops_elapsed(b) if b.time_elapsed > 0 else 0.0
        metricas[nt]["Média GFlops instruções"] = gflops_instrucoes
        metricas[nt]["Média percentual pico instruções (%)"] = (
            gflops_instrucoes / PICO_TEORICO_GFLOPS
        ) * 100.0

    tabela_csv = []
    for i in range(maior_amostra):
        linha = {"N": f"T{i}"}
        for nt in ntasks_unicas:
            tempos = grupos_polidos[nt]
            linha[nt] = fmt(tempos[i]) if i < len(tempos) else ""
        tabela_csv.append(linha)

    # Inserção das novas métricas na tabela
    ordem_metricas = [
        "Média",
        "Mediana",
        "Valor Mínimo",
        "Speedup",
        "Eficiencia",
        "Desvio padrão",
        "Nível de significância",
        "Tamanho da amostra",
        "Valor critico",
        "Valor de confiança",
        "Lim. inf. intervalo de confiança",
        "Lim. sup. intervalo de confiança",
        "Intervalo confiança percentual (1/2)",
        "Média instruções escalares",
        "Média instruções SSE",
        "Média instruções AVX2",
        "Média instruções AVX512",
        "Média instruções",
        "Média cache miss",
        "Média llc load miss",
        "Média page faults",
        "Média context switchs",
        "Média migrations",
        "Média perf time elapsed",
        "Média perf time user",
        "Média perf time sys",
        "Média GFlops instruções",
        "Média percentual pico instruções (%)",
    ]

    # Inserção
    for nome_metrica in ordem_metricas:
        linha = {"N": nome_metrica}
        for nt in ntasks_unicas:
            linha[nt] = fmt(metricas[nt][nome_metrica])
        tabela_csv.append(linha)

    fieldnames = ["N"] + ntasks_unicas
    with open(nome_saida, "w", newline="") as fd:
        writer = csv.DictWriter(fd, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(tabela_csv)

    print(
        # f"Gerou o arquivo {nome_saida} com sucesso! (Matriz utilizada: {tamanho_matriz}x{tamanho_matriz})"
        f"Gerou o arquivo {nome_saida} com sucesso!"
    )
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 2:
        res = criar_tabela(sys.argv[1], sys.argv[2])
        if res == 1:
            print(
                "Não foi encontrada nenhuma amostra. Olhe os seus dados e confirme se estão corretos."
            )
    else:
        print(
            "Uso incorreto. Formato esperado:\npython parse_data.py arquivo_entrada.csv arquivo_saida.csv"
        )
