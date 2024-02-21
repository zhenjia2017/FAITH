#!/usr/bin/bash
#SBATCH -o out/slurm/out

## check argument length
if [[ $# -lt 1 ]]
then
	echo "Error: Invalid number of options: Please specify at least the pipeline-function."
	echo "Usage: bash scripts/distant_supervision.sh\\
		--train\\
		/--dev\\
	[<PATH_TO_CONFIG>]"
	exit 0
fi

## read config parameter: if no present, stick to default (default.yaml)
FUNCTION=$1
CONFIG=${2:-"config/evaluate.yml"}

## set path for output
# get function name
FUNCTION_NAME=${FUNCTION#"--"}
# get data name
IFS='/' read -ra NAME <<< "$CONFIG"
DATA=${NAME[1]}
# get config name
CFG_NAME=${NAME[2]%".yml"}

# set output path (include sources only if not default value)
if [[ $# -lt 3 ]]
then
	OUT="out/${DATA}/distant-supervision-${FUNCTION_NAME}-${CFG_NAME}.out"
fi

echo $OUT

## fix global vars (required for FiD)
export SLURM_NTASKS=1
export TOKENIZERS_PARALLELISM=false

## start script
if ! command -v sbatch &> /dev/null
then
	# no slurm setup: run via nohup
	export FUNCTION CONFIG SOURCES OUT
  	nohup sh -c 'python -u code/faith/distant_supervision/distant_supervision.py ${FUNCTION} ${CONFIG}' > $OUT 2>&1 &
else
	# run with sbatch
	sbatch <<EOT
#!/bin/bash

#SBATCH --job-name=$OUT
#SBATCH -p gpu20
#SBATCH --gres gpu:1
#SBATCH -t 0-01:00:00
#SBATCH -o $OUT
#SBATCH -d singleton
#SBATCH --mem 256G

python -u code/faith/distant_supervision/distant_supervision.py $FUNCTION $CONFIG
EOT
fi
