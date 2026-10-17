#!/bin/env bash

#
##
### Exemplo de execução utilizada no ambiente SLURM do Supercomputador Santos Dumont do LNCC
##
#

# CARREGAMENTO DE AMBIENTE 
source gcc_module
source gmail.sh
source /scratch/app_sequana/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate omp_env

mkdir -p build

# Configuração Padrão do Experimento
AMOSTRAS=5
PASSO=5 # max 30 para 30 segundos, para um limite de 20 minutos

echo "========================================================="
echo " COMPILAÇÃO"
echo "========================================================="

make 

JOB_EXEC="./OESA/job_exec.py"

wait_run_cores_close() { 
  local programn="$1"
  local info="$2"
  local TEMPO="$3"
  local SIZE="$4"
  local amostras="$5"
  local passo="$6"
  local nproc="$7"

  python $JOB_EXEC \
    --name "benchmark_$programn" \
    --info "Execução paralela até 32 threads ($info)" \
    --path "./build/$programn" \
    --args "${TEMPO} ${SIZE}" \
    --header "N,Tamanho,Iterations,Tempo" \
    --sequana "sequana_cpu_dev" \
    --modo openmp \
    --nprocs "$nproc" \
    --omp_procbind "close" \
    --omp_places "cores" \
    --amostras "$amostras" \
    --passo "$passo" \
    --tipo slurm \
    --mem 10G
  
  mv $SCRATCH/"benchmark_$programn"/sim_data.bin ./"sim_data_${SIZE}_${TEMPO}.bin"
}

wait_run_cores_close_sequential() { 
  local programn="$1"
  local info="$2"
  local TEMPO="$3"
  local SIZE="$4"
  local amostras="$5"
  local passo="$6"
  local nproc="$7"

  python $JOB_EXEC \
    --name "benchmark_$programn" \
    --info "Execução sequencial ($info)" \
    --path "./build/$programn" \
    --args "${TEMPO} ${SIZE}" \
    --header "N,Tamanho,Iterations,Tempo" \
    --sequana "sequana_cpu_dev" \
    --modo openmp \
    --nprocs "1" \
    --omp_procbind "close" \
    --omp_places "cores" \
    --amostras "$amostras" \
    --passo "$passo" \
    --tipo slurm \
    --mem 10G 
  
  mv $SCRATCH/"benchmark_$programn"/sim_data.bin ./"sim_data_${SIZE}_${TEMPO}.bin"
}

echo "========================================================="
echo " SUBMETENDO JOBS PARA O CLUSTER "
echo "========================================================="

T=60
for i in 1024 2048 4096 8192; do
	wait_run_cores_close "main" "main N=$i" $T $i $AMOSTRAS $PASSO "1 2"
done

echo "Jobs submetidos com sucesso!"
