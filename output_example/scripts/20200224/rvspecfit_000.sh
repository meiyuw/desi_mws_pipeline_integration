#!/bin/bash 
one=$(sbatch /physics2/meiyuw/desi_out/script_output/20200224/rvspecfit_000.slurm) 
echo $one 
wait 
cd /home/meiyuw/desi_MWS_pipeline/piferre 
python piferre.py --input_files /physics2/meiyuw/desi_out/script_output/20200224/x.000  --output_script_dir /physics2/meiyuw/desi_out/script_output  --output_dir /physics2/meiyuw/desi_out --minexpid=-1
wait 
j5600=$(sbatch /physics2/meiyuw/desi_out/script_output/20200224/56/5600/5600.slurm) 
echo $j5600 
j5605=$(sbatch /physics2/meiyuw/desi_out/script_output/20200224/56/5605/5605.slurm) 
echo $j5605 
j5607=$(sbatch /physics2/meiyuw/desi_out/script_output/20200224/56/5607/5607.slurm) 
echo $j5607 