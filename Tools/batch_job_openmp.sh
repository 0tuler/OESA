#!/bin/bash

#SBATCH --nodes=1
#SBATCH -p sequana_cpu_dev
#SBATCH --threads-per-core=1
#SBATCH --time=00:20:00
#SBATCH --exclusive

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
preload=$6
my_args=$7

if [[ -z "$SLURM_CPUS_PER_TASK" ]]; then
	echo "SLURM_CPUS_PER_TASK não foi setada!"
	exit 1
fi

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_DYNAMIC=false

if [[ ! -x "./$execfile" ]]; then
	echo "ERRO: O arquivo '$execfile' não foi encontrado ou não é executável no diretório atual: $(pwd)"
	exit 1
fi

echo "------------------"
echo "N: ${OMP_NUM_THREADS}"
for i in $(seq -s " " $min_amostras $max_amostras); do
	perf stat -e task-clock,cycles,fp_arith_inst_retired.scalar_double,fp_arith_inst_retired.128b_packed_double,fp_arith_inst_retired.256b_packed_double,fp_arith_inst_retired.512b_packed_double,instructions,cache-misses,LLC-load-misses,page-faults,cs,migrations -o "perf_stat_${i}_${OMP_NUM_THREADS}.txt" ./$execfile $my_args >> $saida

	# Limpeza
	echo "TH: ${OMP_NUM_THREADS}" >> perf.txt
	cat "perf_stat_${i}_${OMP_NUM_THREADS}.txt" >> perf.txt
	rm "perf_stat_${i}_${OMP_NUM_THREADS}.txt"
done
