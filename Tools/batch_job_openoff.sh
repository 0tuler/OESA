#!/bin/bash
#WIP - Work in Progress

#SBATCH --nodes=1
#SBATCH -p sequana_gpu_dev
#SBATCH --threads-per-core=1
#SBATCH --time=00:20:00

# carregamento do openmpi
module load openmpi/gnu/4.1.4_sequana

# seta o binario para a execuçao
execfile=$1

# envia os resultados para o arquivo csv
saida=$2

# Numero maximo e minimo de amostras
max_amostras=$4
min_amostras=$3

sequana=$5
teams=$6
nproc=$7
preload=$8
my_args=$9

if [[ -z "$SLURM_CPUS_PER_TASK" ]]; then
  echo "SLURM_CPUS_PER_TASK não foi setada!"
  exit 1
fi

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=true
export OMP_PLACES=cores
export OMP_TARGET_OFFLOAD=MANDATORY

export OMP_NUM_TEAMS=$teams # 2x ou 4x o número de SMs (Streaming multiprocessors)
export OMP_TEAMS_THREAD_LIMIT=$nproc # padrão 128 - 256, podendo diminuir para 64 ou 128

if [[ ! -x "./$execfile" ]]; then
    echo "ERRO: O arquivo '$execfile' não foi encontrado ou não é executável no diretório atual: $(pwd)"
    exit 1
fi

echo "------------------"
echo "N: ${nproc}"
for _ in $(seq -s " " $min_amostras $max_amostras); do
  srun --exclusive -p $sequana --mem-bind=local ./$execfile $my_args >> $saida
done
