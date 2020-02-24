#!/bin/bash 
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# 
#This script was written by piferre.py on Mon Feb 24 12:11:28 2020 
#SBATCH  -J 5605 
#SBATCH  -p long 
#SBATCH  -o 5605_%j.out 
#SBATCH  -e 5605_%j.err 
#SBATCH  -n 4 
#SBATCH  -t 00:10:00 
#SBATCH  -D /physics2/meiyuw/desi_out/script_output/20200224/56/5605 
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# 
export OMP_NUM_THREADS=4
source /home/meiyuw/desi_MWS_pipeline/setup.sh 
cd /physics2/meiyuw/desi_out/script_output/20200224/56/5605
time /home/meiyuw/desi_MWS_pipeline/ferre/src/a.out /physics2/meiyuw/desi_out/script_output/20200224/56/5605/input.nml_0 >& log_0 & 
time /home/meiyuw/desi_MWS_pipeline/ferre/src/a.out /physics2/meiyuw/desi_out/script_output/20200224/56/5605/input.nml_1 >& log_1 & 
time /home/meiyuw/desi_MWS_pipeline/ferre/src/a.out /physics2/meiyuw/desi_out/script_output/20200224/56/5605/input.nml_2 >& log_2 & 
time /home/meiyuw/desi_MWS_pipeline/ferre/src/a.out /physics2/meiyuw/desi_out/script_output/20200224/56/5605/input.nml_3 >& log_3 & 
time /home/meiyuw/desi_MWS_pipeline/ferre/src/a.out /physics2/meiyuw/desi_out/script_output/20200224/56/5605/input.nml_4 >& log_4 & 
time /home/meiyuw/desi_MWS_pipeline/ferre/src/a.out /physics2/meiyuw/desi_out/script_output/20200224/56/5605/input.nml_5 >& log_5 & 
time /home/meiyuw/desi_MWS_pipeline/ferre/src/a.out /physics2/meiyuw/desi_out/script_output/20200224/56/5605/input.nml_6 >& log_6 & 
time /home/meiyuw/desi_MWS_pipeline/ferre/src/a.out /physics2/meiyuw/desi_out/script_output/20200224/56/5605/input.nml_7 >& log_7 & 
time /home/meiyuw/desi_MWS_pipeline/ferre/src/a.out /physics2/meiyuw/desi_out/script_output/20200224/56/5605/input.nml_8 >& log_8  
python3 -c "import sys; sys.path.insert(0, '/home/meiyuw/desi_MWS_pipeline/piferre'); from piferre import opfmerge, write_tab_fits, write_mod_fits; opfmerge('5605', path='/physics2/meiyuw/desi_out/script_output/20200224/56/5605',wait_on_sorted=True); write_tab_fits('5605', '/physics2/meiyuw/desi_out/script_output/20200224/56/5605', path='/physics2/meiyuw/desi_out/56/5605'); write_mod_fits('5605', '/physics2/meiyuw/desi_out/script_output/20200224/56/5605', path='/physics2/meiyuw/desi_out/56/5605')"
