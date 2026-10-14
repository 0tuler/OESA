#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH -p sequana_cpu_dev
#SBATCH --threads-per-core=1
#SBATCH --time=00:20:00

# carregamento do openmpi
module load openmpi/gnu/4.1.4_sequana

# seta o binario para a execuçao
execfile=$1

# envia os resultados para o arquivo csv
saida=$2

# Numero maximo de amostras
max_amostras=$4
min_amostras=$3

# seta a sequana usada
sequana=$5
preload=$6
my_args=$7

if [[ ! -x "./$execfile" ]]; then
    echo "ERRO: O arquivo '$execfile' não foi encontrado ou não é executável no diretório atual: $(pwd)"
    exit 1
fi

echo "------------------"
echo "N: ${SLURM_NTASKS}"
for _ in $(seq -s " " $min_amostras $max_amostras); do
	# usando awk para prefixar o numero de tasks 
	#srun --exclusive -J $execfile -p $sequana ./$execfile | awk -v p=$SLURM_NTASKS '{print p "," $0}' >> $saida
	srun --exclusive -J $execfile -p $sequana ./$execfile $my_args >> $saida
done
