# desi_mws_pipeline_integration
This is a script to run MWS pipeline "rvspecfit" and "piferre." Currently it is only compatible with rcspecfit. 

This script reads in the latest exposure ID value from the latest report generated by previous runs. It will only process fibers from new exposures. A report (named by date) will be generated to show how many fibers are processed in this run, and also report the exposure ID range in this run.


**To setup:**

**(1)** Follow the instructions to install rvspecfit and piferre packages. Please select the installtion directory to be the same as "DESI_MWS_root" in the "setup.sh"

**(2)** Modify the "setup.sh."


An example command to run the script will be:

python MWS_pipeline_script.py --nthreads 64 --input_dir /.../spectra-64  --output_dir /.../  --output_script_dir /.../  --report_dir /.../ 

**Options:**

--nthreads: number of processors per node.

--input_dir: directory of input files to be processed.

--output_dir: directory for the output files.

--output_script_dir: Output directory for the slurm scripts, shell script, and ferre input files.

--report_dir: directory of the report files.

--whole_spectra64: the script will search and process every new exposures in the whole spectra64 directory. Otherwise it will read in pointing coordinates of new exposures from DESI db to find out which spectra64 files with those healpix numbers have been updated.

--allobjects: process not just MWS targets but every other types.