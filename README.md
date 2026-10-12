# OESA - Orquestrador de Execuções para o sistema Slurm e Analise.

Este projeto consiste em um conjunto automatizado de ferramentas Python e arquivos batch do slurm para a submissão de jobs, coleta de dados estatísticos, remoção de *outliers*, cálculos de métricas de desempenho e geração de relatórios visuais para ambientes de Computação de Alto Desempenho (HPC).

O fluxo principal foi projetado para rodar em clusters gerenciados via **SLURM** ou de forma **Local**, a forma local ainda está em testes, suportando execuções baseadas em **MPI**, **OpenMP** e **OpenMP Offload (GPU)**, este ainda em testes.

Esta ferramenta foi usada para a execução e análise dos programas listados no repositório do livro [Programacao Paralela e Distribuida](https://github.com/Programacao-Paralela-e-Distribuida/OPENMP).

## Arquitetura do Projeto

O sistema é dividido em 4 scripts principais:

1. **`job_exec.py`:** Responsável por orquestrar a execução. Prepara um diretório no *SCRATCH*, copia dependências, repassa os parâmetros para o SLURM ou terminal local e invoca os scripts secundários quando as amostras são finalizadas.
2. **`parse_data.py`:** Efetua o *parsing* e a limpeza dos resultados coletados. Remove anomalias estatísticas, ou seja atrasos sistêmicos, e calcula o Speedup, Eficiência, GFLOP/s, Intervalos de Confiança, além de processar dados de hardware advindos do `perf`.
3. **`graficos.py`:** Usa a biblioteca `matplotlib` para gerar automaticamente gráficos que indicam as métricas escalares em função do número de threads/processos (como Tempo Médio, Speedup, Eficiência, GFLOP/s e métricas de cache/hardware).
4. **`enviar_email.py`:** Agrupa os resultados (csv e gráficos em `.png`) e os envia por e-mail usando SMTP.

---

## Requisitos e Configurações

*   **Linguagem:** Python 3.x
*   **Dependências:** `numpy`, `scipy`, `matplotlib`, `listadas no requirements.txt`
*   **Ambiente de E-mail:** O envio de e-mails exige que você exporte as credenciais no seu ambiente de terminal antes de executar o script:
    ```bash
    export email="seu_email@gmail.com"
    export passwd="sua_senha_de_app" # utiliza senha de aplicativos do google, necessário configurar à parte
    ```

---

## Instalação 

```bash

git clone https://github.com/0tuler/OESA.git

pip -r OESA/requirements.txt

```

Talvez seja necessário configurar um ambiente conda, no testes executados no ambiente slurm do Supercomputador Santos Dumont do LNCC houve a necessidade de utilizar o ambiente conda.

## Opções do `job_exec.py`

O script `job_exec.py` é o ponto de entrada da ferramenta. Ele deve ser executado passando opções via linha de comando. Abaixo estão detalhadas todas as opções suportadas:

### Opções Obrigatórias
| Opção | Comando Longo | Descrição |
| :--- | :--- | :--- |
| `-n` | `--name` | Nome da bateria de testes/projeto. Usado para criar arquivos de saída e diretórios. |
| `-p` | `--path` | Caminho do arquivo executável que será avaliado. |
| | `--header` | Define as colunas iniciais do arquivo CSV (ex: `"N,Tempo,Métricas..."`). |
| | `--sequana` | Define uma configuração específica para o ambiente sequana do job. |

### Configurações de Execução e Amostragem
| Opção | Comando Longo | Descrição | Padrão |
| :--- | :--- | :--- | :--- |
| `-a` | `--amostras` | Número **total** de execuções/amostras a serem coletadas. | `10` |
| | `--passo` | Quantidade de repetições efetuadas a cada submissão (divide a amostragem em blocos menores). | `5` |
| | `--tipo` | Onde os jobs vão executar: `slurm` (fila no cluster) ou `local` (máquina local). | `slurm` |
| | `--modo` | Define a API paralela. Suporta `mpi`, `openmp` ou `openoff` (para GPUs). | `mpi` |
| | `--nprocs` | Número de Processos/Threads a testar, passados entre aspas duplas e separados por espaço. | `"1 2 4 8 16 24 32"` |
| | `--args` | Argumentos textuais que devem ser repassados diretamente ao binário em teste. | `""` |
| | `--preload` | Bibliotecas ou strings de pré-carregamento (ex: `LD_PRELOAD=...`) para o run. | `""` |

### Configurações Específicas de Hardware e SLURM
| Opção | Comando Longo | Descrição |
| :--- | :--- | :--- |
| | `--mem` | Quantidade de memória solicitada no SLURM (ex: `16G`). |
| | `--batch_script` | Nome/Caminho de um script `.sh` customizado que sobrepõe o gerado automaticamente. |
| | `--gpus` | (Obrigatório se `--modo=openoff`) Número de GPUs solicitadas. |
| | `--teams` | (Obrigatório se `--modo=openoff`) Configura o número de times na diretiva Offload. |
| | `--omp_procbind`| Define como as threads OpenMP devem estar ligadas no SO. (Padrão: `close`) |
| | `--omp_places` | Determina em quais estruturas alocar (Padrão: `cores`). |

### Opções de Utilitários e Arquivos
| Opção | Comando Longo | Descrição |
| :--- | :--- | :--- |
| | `--extra_files` | Lista de arquivos (separados por espaço) que devem ser copiados junto com o executável para o diretório temporário (*SCRATCH*). |
| | `--info` | Texto adicional que será acrescentado ao final do assunto no e-mail de notificação. |
| | `--perf` | *Flag*: Se inserido, avisa o script de batch para rodar a extração de hardware `perf stat`. |
| | `--just_email` | *Flag*: Pula a geração de gráficos e filtragem de amostras, apenas enviando os dados puros extraídos para o e-mail cadastrado. |

---

## Exemplo de Uso

Um exemplo de coleta de desempenho de um programa OpenMP local, com 10 amostras, passando um arquivo extra (`entrada.txt`) usando de 1 até 16 threads:

```bash
# Setar variáveis
export email="eu@meuemail.com"
export passwd="senha"

# Executar
python job_exec.py \
-n teste_openmp\
-p ./matriz_mult\
--header "N,Tempo" \
  --sequana default_seq\
  -a 10\
  --passo 5\
  --tipo local\
  --modo openmp \
  --nprocs "1 2 4 8 16"\
  --extra_files entrada.txt

```

Toda essa execução pode ser mantida em um arquivo sh.
