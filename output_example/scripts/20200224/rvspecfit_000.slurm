#!/bin/bash 
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# 
#SBATCH -N 1       
#SBATCH -n 20       
#SBATCH -t 00:10:00 
#SBATCH -p long 
#SBATCH -o rvspecfit.o%j 
#SBATCH -e rvspecfit.e%j 
#SBATCH --exclusive 
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# 
export OMP_NUM_THREADS=1 
source /home/meiyuw/desi_MWS_pipeline/setup.sh 
cd /home/meiyuw/desi_MWS_pipeline/rvspecfit/desi 
srun -n 1 -c 20 rvs_desi_fit --nthreads 20 --config /home/meiyuw/desi_MWS_pipeline/rvspecfit/desi/config.yaml --input_file_from /physics2/meiyuw/desi_out/script_output/20200224/x.000 --output_dir /physics2/meiyuw/desi_out --minexpid=-1
