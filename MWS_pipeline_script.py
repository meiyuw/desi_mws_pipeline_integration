'''
DESI MWS pipeline intergration interface to use FERRE and rvspecfit
2020.2.18
Current version is compatible with rvspecfit and modified piferre. 
Author: Mei-Yu Wang
Some of the functions are adopted from "rvsepcfit" by Sergey Koposov and "piferre" by Carlos Allende Prieto
'''
import pdb
import sys
import os
import glob
import math
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import datetime, time
from datetime import date

os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import pandas
import itertools
import concurrent.futures
from collections import OrderedDict

import matplotlib
import astropy.io.fits as pyfits
#matplotlib.use('Agg')
import astropy.table
from multiprocessing import cpu_count
import astropy.table as atpy

from rvspecfit import fitter_ccf, vel_fit, spec_fit, utils

#---------- Sergey's function to select targets to fits -----------

def read_data(fname):
    fluxes = {}
    ivars = {}
    waves = {}
    masks = {}
    for s in 'brz':
        fluxes[s] = pyfits.getdata(fname, '%s_FLUX' % s.upper())
        ivars[s] = pyfits.getdata(fname, '%s_IVAR' % s.upper())
        masks[s] = pyfits.getdata(fname, '%s_MASK' % s.upper())
        waves[s] = pyfits.getdata(fname, '%s_WAVELENGTH' % s.upper())
    return fluxes, ivars, masks, waves
#---------------------------------------------------------------------------------
def get_sns(data, ivars, masks):
    """
    Return the vector of S/Ns
    """
    xind = (ivars <= 0) | (masks > 0)
    xsn = data * np.sqrt(ivars)
    xsn[xind] = np.nan
    xsn[xind] = np.nan
    sns = np.nanmedian(xsn, axis=1)

    return sns
#---------------------------------------------------------------------------------    
def select_fibers_to_fit(fibermap,
                         sns,
                         minsn=None,
                         mwonly=True,
                         expid_range=None):
    """
    Identify fibers to fit and number of fibers for different criteria.
    Criteria: MWS_TARGET, S/N cut, EXP ID range,
    Parameters:
    ----------
    fibermap: Table
        Fibermap table object
    sns: dict of numpy arrays 
        Array of S/Ns
    minsn: float
        Threshold S/N
    mwonly: bool
        Only fit MWS
    expid_range: int array
        EXP ID range       

    Returns:
    -------
    subset: bool numpy array
        Array with True for selected spectra
    n_EXPrange: int
    	Number of fibers within the EXP ID range
    n_MWS_traget: int
    	Number of fibers within the EXP ID range are MWS targets.
    n_minsn: int
    	Number of fibers within the EXP ID range are with snr > min(snr).
    n_proc: int
    	Number of fibers to be processed in this run.
    """    
    if expid_range is not None:
        mine, maxe = expid_range
        if mine is None:
            mine = -1
        if maxe is None:
            maxe = np.inf
    subset = (fibermap["EXPID"] > mine) & (fibermap['EXPID'] <= maxe)
    
    n_EXPrange=np.sum(subset)
    
    n_MWS_target=np.sum((fibermap['MWS_TARGET'][subset] != 0)) 
    if mwonly:
        subset = (fibermap['MWS_TARGET'][subset] != 0)
    else:
        subset = subset 
       
    
    if minsn is not None:
        maxsn = np.max(np.array([sns[_] for _ in 'brz']), axis=0)
        subset = subset & (maxsn > minsn)
        n_minsn=np.sum((maxsn > minsn))
    else:
        n_minsn=np.sum(subset)
    n_proc=np.sum(subset) 
    
    return subset,n_EXPrange, n_MWS_target, n_minsn, n_proc     

#---------------------------------------------------------------------------------
    
def cal_node_n(n_proc_now,pix_list,sdir_list,nthreads_input):
    """
    Determining the number of nodes to be requested. 
    Combining files where the number of fibers are less than nthreads
    -----------
    Inputs:
    
    n_proc_now: int array
    Number of fibers will be processed in each spectra64 files.
    
    pix_list: string array
    List of pixel number of which will be processed in this run.
    
    nthreads: int
    Number of processors for one node.
    
    ------------
    Outputs:
    pix_gp_list: list
    List of grouped pixel number. Each group is assigned to one node to be processed. 
    
    fn_sum_list: list
    List of number of fibers to be processed by each node.
    """
    # Calculating how many fibers should be grouped for each node so that minimun run time of 10 min is reached.
    # Assuming analysis time per fiber per processor is 50 s. Minimun asking for 10 min.
    nthreads=int(9.5*(60.0/45.0)*nthreads_input) # Calculate how many fibers can be processed in 9.5 mins (0.5 min overhead)	
    print('nthreads=',nthreads)
    pix_list=np.array(pix_list)
    sdir_list=np.array(sdir_list)
    n_proc_now=np.array(n_proc_now)
    sort_index=np.arange(len(pix_list))
    
    job_ind=np.zeros(len(pix_list),dtype=int)
    
    # If nothing to be processed, the job index is -1 
    job_ind[np.array(n_proc_now==0)]=-1
    
    if(nthreads <= 0):
        print('ERROR: n_threads=', nthreads)
        return
    if(len(n_proc_now) <= 0):
        print('ERROR: empty n_proc_now array.')
        return
    if(len(pix_list) <= 0 or len(sdir_list) <= 0):
        print('ERROR: empty pix_list or sdir_list array.')
        return
    
    if(len(pix_list) <= 1):
        return n_proc_now,pix_list

    pix_list=pix_list[np.argsort(n_proc_now)[::-1]]
    sdir_list=sdir_list[np.argsort(n_proc_now)[::-1]]
    job_ind=job_ind[np.argsort(n_proc_now)[::-1]]
    sort_index=sort_index[np.argsort(n_proc_now)[::-1]]
    n_proc_now=n_proc_now[np.argsort(n_proc_now)[::-1]] 
       
    n_node=np.sum(n_proc_now >= nthreads)
    pix_list_sub=pix_list[n_proc_now < nthreads]
    sdir_list_sub=sdir_list[n_proc_now < nthreads]
    n_proc_now_sub=n_proc_now[n_proc_now < nthreads]
    
    n_empty_file=np.sum(n_proc_now_sub <= 0)
    pix_list_sub=pix_list_sub[n_proc_now_sub > 0]
    sdir_list_sub=sdir_list_sub[n_proc_now_sub > 0]
    n_proc_now_sub=n_proc_now_sub[n_proc_now_sub > 0]       
    
    pix_gp_list=[]
    sdir_gp_list=[]
    fn_sum_list=[]
    for i in range(n_node):
        pix_gp_list.append([pix_list[i]])
        sdir_gp_list.append([sdir_list[i]])
        fn_sum_list.append(n_proc_now[i])
        job_ind[i]=i
        
    if(n_empty_file>0):
        for i in range(len(job_ind)-1,len(job_ind)-n_empty_file-1,-1):
            job_ind[i]=-1
    
    # If there are files with fiber less than nthreads
    j_t=len(n_proc_now_sub)-1
    tmp=[]
    tmp_sdir=[]
    ind=n_node
    if(np.sum(n_proc_now < nthreads) > 0):
        for i in range(len(n_proc_now_sub)):            
            fd=0
            sum_sub=n_proc_now_sub[i]
            tmp=[pix_list_sub[i]]    
            tmp_sdir=[sdir_list_sub[i]]                   
            if(i < j_t):
                for j in range(j_t,i,-1):
                    if(sum_sub+n_proc_now_sub[j] > nthreads):
                        j+=1
                        break
                    if(sum_sub+n_proc_now_sub[j] <= nthreads):
                        sum_sub+=n_proc_now_sub[j]
                        tmp.append(pix_list_sub[j])
                        tmp_sdir.append(sdir_list_sub[j])
                        job_ind[j+n_node]=ind   
                        job_ind[i+n_node]=ind  
                #=== j had looped over the rest of the array or break==
                pix_gp_list.append(tmp)
                sdir_gp_list.append(tmp_sdir)
                fn_sum_list.append(sum_sub)  
                if (j <= j_t):                    
                    j_t=j-1
                else:
                    job_ind[i+n_node]=ind 
                ind += 1  
            elif(i == j_t):
                pix_gp_list.append(tmp)
                sdir_gp_list.append(tmp_sdir)
                fn_sum_list.append(sum_sub)
                job_ind[i+n_node]=ind 
                break
            elif(i > j_t):
                break                
            
    return pix_gp_list,sdir_gp_list,fn_sum_list,job_ind,sort_index   

#---------------------------------------------------------------------------------

def write_slurm_tot(out_script_path,out_path,file_ind,in_path,sdirs,pixels,min_expid,n_fiber,nthreads=1, suffix='',whole_spectra64=False,mwonly=True):
	"""
	Writing Slurm script for both rvspecfit and ferre.
	---------------
	Input:
	out_script_path: string
	Directory where the slurm scripts will be stored
	
	out_path: string
	Directory where output files of rvspecfit will be stored
	
	file_ind: int
	Index number for slurm scripts. It is determined by how the jobs are grouped.
	
	sdirs: string list
	Name of the first layer spectra64 directories.	
	
	pixels: string list
	Name of the second layer spectra64 directories.		
	
	min_expid: int
	Minimun Exposure ID of this run. Anything with Exposure ID greater and equal to this will be processed.
	
	n_fiber: int
	Number of fibers will be processed by this job.
	
	nthreads: int
	Number of processors requested. The default is all the available processors in one node.
	
	suffix: string
	Suffix of the rvspecfit output files.
	
	whole_spectra64: boolean
    Processing the whole spectra64 folder or not.
    
	whole_spectra64: boolean
    Whether processing the whole spectra64 folder or not.    
    
	mwonly: boolean
    Whether or not processing MWS targets only.        
	
	---------------
	Output:
	None.
	"""
	yr= datetime.date.today().year
	month=datetime.date.today().month
	day=datetime.date.today().day
	now=str(yr*10000+month*100+day)
	
	if not os.path.exists(os.path.join(out_script_path,now)): os.mkdir(os.path.join(out_script_path,now))
	try:
		host=os.environ['HOST']
	except:
		host='Unknown'
	
	# Calculate the requested runtime. Assuming analysis time per fiber per processor is 45 s.
	# Minimun asking for 10 min, and adding additional 5 min each time when more runtime is needed.
	if (n_fiber > 0):
            runtime=45.0*np.ceil(n_fiber/nthreads) # in second.
            rt_5m=np.ceil(runtime/(5.0*60.0))
            rt_hr=int(rt_5m//12)
            rt_min=max(int((rt_5m%12)*5),10)
	else:
            print('No fibers need to be processed...')
            return
	
	f=open(os.path.join(out_script_path,now,suffix+'.slurm'),'w')
	f.write("#!/bin/bash \n")
	f.write("#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# \n")
	f.write("#SBATCH -N 1       \n")
	f.write("#SBATCH -n "+str(nthreads)+"       \n")
	f.write("#SBATCH -t "+str(rt_hr)+":"+str(rt_min)+":00 \n")
	f.write("#SBATCH -p regular \n")
	f.write("#SBATCH -L SCRATCH \n")
	f.write("#SBATCH -C haswell \n")
	f.write("#SBATCH -a 0-31 \n")
	f.write("#SBATCH -A desi \n")
	f.write("#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# \n")
	f.write("export OMP_NUM_THREADS=1 \n")
	f.write("source "+os.environ['DESI_MWS_root']+"/setup.sh \n")
	f.write("cd "+os.environ['DESI_MWS_root']+"/rvspecfit/desi \n")
	f.write("srun -n 1 -c "+str(nthreads))
	f.write(" rvs_desi_fit --nthreads "+str(nthreads)+" ")
	f.write("--config "+os.environ['DESI_MWS_root']+"/rvspecfit/desi/config.yaml ")
	f.write("--input_file_from "+os.path.join(out_script_path,now)+"/x."+file_ind)
	f.write(" --output_dir "+out_path+" ")  # problematic as it outputs to one single folder
	if (not mwonly):
		f.write(" --allobjects ")
	f.write("--minexpid="+str(int(min_expid)))
	f.write(""+"\n")
	# Add piferre commands if zbest files do not exist
	
	zf=[]
	for pixel,sdir in zip(pixels,sdirs):
		input_path=os.path.join(in_path,sdir,pixel)
		zf.append(glob.glob(input_path+"/zbest-64-"+str(pixel)+".fits"))

	if(len(zf) != len(pixels)):
		f.write("wait \n")
		f.write("cd "+os.environ['DESI_MWS_root']+"/piferre \n")
		f.write("python piferre.py --input_files "+os.path.join(out_script_path,now)+"/x."+file_ind+" ")
		f.write(" --output_script_dir "+out_script_path+" ")	
		f.write(" --output_dir "+out_path+" ")
		if (not mwonly):
			f.write(" --allobjects ")
		f.write("--minexpid="+str(int(min_expid)))
		f.write("\n")

	f.close()
	os.chmod(os.path.join(out_script_path,now,suffix+'.slurm'),0o755)
	return None    
	
#---------------------------------------------------------------------------------    
def write_script_tot_gp(out_script_path,out_path,in_path,sdirs,pixels,file_ind,min_expid,nthreads=1, suffix='',mwonly=False):
	"""
	Writing shell script to excute job submission.
	"""
	yr= datetime.date.today().year
	month=datetime.date.today().month
	day=datetime.date.today().day
	now=str(yr*10000+month*100+day)

	
	zf=[]
	for pixel,sdir in zip(pixels,sdirs):
		input_path=os.path.join(in_path,sdir,pixel)
		zf.append(glob.glob(input_path+"/zbest-64-"+str(pixel)+".fits"))
		
	# If zbest files do not exisit, submitting pyferre jobs after rvspecfit is done			
	f=open(os.path.join(out_script_path,now,suffix+'.sh'),'w')
	f.write("#!/bin/bash \n")
	if(len(zf) == len(pixels)):
		f.write("one=$(sbatch "+os.path.join(out_script_path,now,suffix)+".slurm) \n")
	else:
		f.write("one=$(sbatch "+os.path.join(out_script_path,now,suffix)+".slurm | cut -f 4 -d' ') \n")
		
	f.write("echo $one \n")	

	if(len(zf) == len(pixels)):
		f.write("wait \n")
		f.write("cd "+os.environ['DESI_MWS_root']+"/piferre \n")
		f.write("python piferre.py --input_files "+os.path.join(out_script_path,now)+"/x."+file_ind+" ")
		f.write(" --output_script_dir "+out_script_path+" ")	
		f.write(" --output_dir "+out_path+" ")
		if (not mwonly):
			f.write(" --allobjects ")
		f.write("--minexpid="+str(int(min_expid)))
		f.write("\n")
		f.write("wait \n")	
		
	for pixel,sdir in zip(pixels,sdirs):

		if(len(zf) == len(pixels)):
			f.write("j"+str(pixel)+"=$(sbatch "+os.path.join(out_script_path,now,sdir,pixel,pixel)+".slurm) \n")
		else:
			f.write("j"+str(pixel)+"=$(sbatch --dependency=afterany:$one "+os.path.join(out_script_path,now,sdir,pixel,pixel)+".slurm) \n")
		f.write("echo $j"+str(pixel)+" \n")
	f.close()
	os.chmod(os.path.join(out_script_path,now,suffix+'.sh'),0o755)
	return None        
	
#---------------------------------------------------------------------------------  
def getpixels(root):
#find all pixels in 'root' directory (spectra-64)
  d1=os.listdir(root)
  sdirs=[]
  pixels=[]
  for x in d1:
    d2=os.listdir(os.path.join(root,x))
    res=[i for i in d2 if '.fits' in i] 
    for y in d2: 
      if len(res) == 0: # there are no fits files in the 1st directory, so 2 layer
        sdirs.append(x)
        pixels.append(y)
      else: 
        entry=os.path.join(root,x)
        sdirs.append(x) #keep only the first layer 

  return sdirs,pixels
  

#---------------------------------------------------------------------------------    
def run_scripts(path):
	print('run_scripts(path):',path)
	o=glob.glob(path+"/*.sh")
	for cmd in o:
		job="sh "+cmd
		print('jobs=',job)
		err=subprocess.call(job,shell=True)
	return
#---------------------------------------------------------------------------------
def check_latest_report(path):
    """
    Find the latest report and read in date and largest exposure ID value from the latest run.	

    Input:
	path:string
	directory where the reports are stored.
	
	Output:
	report_file_name:string
	name of the latest report
	
	last_expid:int
	Largest exposure ID value from the latest run.
	
	prev_date:int
	The date of previous run.
    """
    root=os.path.join(path,'run-report-')
    report_files=sorted(glob.glob(root+"*.dat"))

    if(len(report_files)==0):
        print('No Previous Run Report....')
        return 'Nane', -1,20000000
	
    dates=[]
    for i in report_files:
        date_tmp=i.split('-')
        day=int(date_tmp[4].split('.')[0])
        one_date=int(date_tmp[2])*10000+int(date_tmp[3])*100+day
        dates.append(one_date)
    dates=np.array(dates)
    file_rp = open(report_files[np.where(dates==np.max(dates))[0][0]],"r")
    file_rp.readline()  # read the header
    file_rp.readline()  # read the header
    last_expid=file_rp.readline()  # read the header
    file_rp.close()
    prev_date=np.max(dates)

    return report_files[np.where(dates==np.max(dates))[0][0]],int(last_expid),prev_date

#---------------------------------------------------------------------------------

def desi_db_new_exposure(prev_date,one_day=False):
	"""
    Access the desi database and retrive coordinate of new exposures to figure out which spectra64 files are updated
    
    Input:
    prev_date:int64
    The date of previous MWS pipeline run.
    
    Output:
    pix_list:string
    List of pixel group (related to names of spectra64 files) that were updated after the date of previous MWS pipeline run.
    
    n_points: int
    Number of new poitings
    
    last_expid: int
    The largest exposure ID value to date.
    """
	from desietcimg.db import DB, Exposures, NightTelemetry
	import desietcimg
	import healpy as hp
	import getpass
	
	if not os.path.exists('db.yaml'):
		pw = getpass.getpass(prompt='Enter database password: ')
	
	with open('db.yaml', 'w') as f:
		print('host: db.replicator.dev-cattle.stable.spin.nersc.org', file=f)
		print('dbname: desi_dev', file=f)
		print('port: 60042', file=f)
		print('user: desi_reader', file=f)
		print(f'password: {pw}', file=f)
	
	print('Created db.yaml')

	db = DB()
	if(one_day):
		newexps = db.query("select id,night,reqra,reqdec from exposure where night = "+str(prev_date)+" order by id asc").dropna()
	else:
		newexps = db.query("select id,night,reqra,reqdec from exposure where night > "+str(prev_date)+" order by id asc").dropna()	
	n_points=len(newexps)
	
	ra = newexps['reqra']
	dec = newexps['reqdec']
	theta = 0.5 * np.pi - np.deg2rad(dec)
	phi = np.deg2rad(ra)
	nside=64
	pix_list = hp.ang2pix(nside, theta, phi)
	last_expid=np.max(id)
	

	
	return pix_list,n_points,last_expid
	
#---------------------------------------------------------------------------------	
def check_spectra64(report_dir,input_path,out_path,mwonly,last_expid,prev_date,whole_spectra64):
    """
    Access the desi database and retrive coordinates of new exposures to figure out which spectra64 files are updated.
    A run report will be generated to indicate which spectra64 and how many fibers will be ran during this run.
    This report doesn't show which runs are successful and which are not, but it will check how many fibers were processed (via rvtab files).
    
    Input:
    
    report_dir: string
    Directory where the reports are stored.
    
    input_path: string
    Directory where the spectra64 files are stored.
    
    outpath: string
    Directory where the MWS output files are stored.
    
    mwonly: boolean
    Whether fit all objects or only MW_TARGET.
    
    last_expid: int
    The largest exposure ID value to date reading in from previous reports.
    
    prev_date: int
    The date of previous run.
    
    whole_spectra64: boolean
    Processing the whole spectra64 folder or not.
    
    """
    #=== If not processing the whole spectra64 directory, reading in information about new exposures from DESI db ===#
    if(not whole_spectra64):
        pix_list,n_points,last_expid=desi_db_new_exposure(prev_date)
        sdir_list=[i[:-2] for i in pix_list]
    else:
        sdir_list,pix_list=getpixels(input_path)
        n_points=1
	
    n_p_proc=np.zeros(len(pix_list),dtype=int)
    n_proc=np.zeros(len(pix_list),dtype=int)
    n_proc_now=np.zeros(len(pix_list),dtype=int)
    n_EXPrange=np.zeros(len(pix_list),dtype=int)
    n_MWS_target=np.zeros(len(pix_list),dtype=int)
    n_minsn=np.zeros(len(pix_list),dtype=int)
    tot_exp=np.zeros(len(pix_list),dtype=int)	
	
    #------ Finding the new largest EXPID and checking how many new fibers need to be processed ------- 
    max_expid=0
    for i,pixel in enumerate(pix_list):
        sdir=sdir_list[i]
        fname_p=out_path+'/'+str(sdir)+'/'+str(pixel)+'/'+'rvtab-64-'+str(pixel)+'.fits'
        if os.path.exists(fname_p):
            fm = pyfits.getdata(fname_p, 'FIBERMAP')
            n_p_proc[i]=len(fm["EXPID"]) 
        else:
            n_p_proc[i]=0
        fname=input_path+'/'+str(sdir)+'/'+str(pixel)+"/spectra-64-"+str(pixel)+".fits"
        fibermap = pyfits.getdata(fname, 'FIBERMAP')
        expids = np.array(fibermap["EXPID"])
        fluxes, ivars, masks, waves = read_data(fname)
        sns = dict([(_, get_sns(fluxes[_], ivars[_], masks[_])) for _ in 'brz'])
        mask,n_EXPrange_t, n_MWS_target_t, n_minsn_t, n_proc_t = select_fibers_to_fit(fibermap,sns,mwonly=mwonly,expid_range=(last_expid, np.inf))
        n_proc[i]=n_proc_t
        n_proc_now[i]=np.sum(mask)
        n_EXPrange[i]=n_EXPrange_t
        n_MWS_target[i]=n_MWS_target_t
        n_minsn[i]=n_minsn_t
        tot_exp[i]=len(expids)
        if (np.max(expids) > max_expid):
            max_expid=np.max(expids)
        if(i ==0):
            min_expid=np.min(expids)
        else:
            if(np.min(expids) < min_expid):
                min_expid=np.min(expids)
    
    print('EXP ID range:',min_expid,max_expid)
    expid_range=[last_expid,max_expid]
    mwonly_list=[mwonly for i in range(len(tot_exp))]
    report_data=np.column_stack((np.array(sdir_list),np.array(pix_list),tot_exp,n_proc_now,n_EXPrange,n_MWS_target,n_minsn,mwonly_list,n_p_proc))
    return n_proc_now, pix_list,sdir_list,expid_range,prev_date,report_data
	
#---------------------------------------------------------------------------------	
def generate_reports(input_path,report_dir,expid_range,prev_date,report_data,job_ind,sort_index):
    """
    Access the desi database and retrive coordinates of new exposures to figure out which spectra64 files are updated.
    A run report will be generated to indicate which spectra64 and how many fibers will be ran during this run.
    This report doesn't show which runs are successful and which are not, but it will check how many fibers were processed (via rvtab files).
    
    Input:
    
    report_dir: string
    Directory where the reports are stored.
    
    input_path: string
    Directory where the spectra64 files are stored.
    
    outpath: string
    Directory where the MWS output files are stored.
    
    mwonly: boolean
    Whether fit all objects or only MW_TARGET.
    
    last_expid: int
    The largest exposure ID value to date reading in from previous reports.
    
    prev_date: int
    The date of previous run.
    
    whole_spectra64: boolean
    Processing the whole spectra64 folder or not.
    
    """
    report_data=report_data[sort_index,:]
    sdir_list=report_data[:,0]
    pix_list=report_data[:,1]
    tot_exp=report_data[:,2]
    n_proc_now=report_data[:,3]
    n_EXPrange=report_data[:,4]
    n_MWS_target=report_data[:,5]
    n_minsn=report_data[:,6]
    mwonly_list=report_data[:,7]
    n_proc_p=report_data[:,8]
		
    min_expid=expid_range[0]		
    max_expid=expid_range[1]
    d1 = datetime.date.today()
    rp_output = open(report_dir+"/run-report-"+str(d1)+".dat","w")
    date_tmp=str(d1).split('-')
    day=int(date_tmp[2])
    today_date=int(date_tmp[0])*10000+int(date_tmp[1])*100+day
    rp_output.writelines('#EXPID range, today\'s date, date of previous run: \n')
    rp_output.writelines('#(1) Spectra64 file name, (2) job script index, Number of: (3) total fiber counts, (4) fibers processed in previous runs, (5) fibers processed in this run, (6) new exposures, (7) new fibers with snr > min(snr), (8) new MWS targets, (9) MWS targets only? : \n')	
    rp_output.writelines(str(max_expid))
    rp_output.writelines('\n')
    rp_output.writelines(str(min_expid))		
    rp_output.writelines('\n')
    rp_output.writelines(str(today_date))
    rp_output.writelines('\n')	
    rp_output.writelines(str(prev_date))	
    rp_output.writelines('\n')
    for i,entry in enumerate(pix_list):
        pixel=entry
        sdir=sdir_list[i]
        rp_output.writelines(input_path+'/'+str(sdir)+'/'+str(pixel)+"/spectra-64-"+str(pixel)+".fits "+" "+str(job_ind[i])+" "+str(int(tot_exp[i]))+" "+str(int(n_proc_p[i]))+" "+str(int(n_proc_now[i])))
        rp_output.writelines(" "+str(int(n_EXPrange[i]))+" "+str(int(n_minsn[i]))+" "+str(int(n_MWS_target[i]))+" "+str(mwonly_list[i]))
        rp_output.writelines('\n')
    rp_output.close()
	
    return None 
#---------------------------------------------------------------------------------    
def proc_mws(args):
    """
    Process DESI MWS spectral files

    """
   # N_processor=64  # Number of processors on Haswell   
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--nthreads',
        help='Number of processor per node',
        type=int,
        default=1)

    parser.add_argument(
        '--input_dir',
        help='Output directory of files to process',
        type=str,
        default=None,
        required=True)

    parser.add_argument(
        '--output_dir',
        help='Output directory for the data tables',
        type=str,
        default=None,
        required=True)
        
    parser.add_argument(
        '--output_script_dir',
        help='Output directory for the slurm scripts and ferre input files',
        type=str,
        default=None,
        required=True)
        
    parser.add_argument(
        '--report_dir',
        help='directory of the report files',
        type=str,
        default=None,
        required=True)    
        
    parser.add_argument(
    	'--allobjects',
        help='Fit all objects, not just MWS_TARGET',
        action='store_true',
        default=False) 

    parser.add_argument(
    	'--whole_spectra64',
        help='Process the whole spectra64 directory',
        action='store_true',
        default=False)  
                                   

    args = parser.parse_args(args)
    path = args.input_dir 
    report_dir=args.report_dir
    mwonly= not args.allobjects
    whole_spectra64 = args.whole_spectra64
    out_path = args.output_dir
    out_script_path = args.output_script_dir
    nthreads = args.nthreads
    
    config_path=os.environ['rvspecfit_config']+"/config.yaml"
    python_path=os.environ['DESI_MWS_root']+"/piferre"
    
    config = utils.read_config(config_path)

    
    #-------- Read in latest previous report and generate a new report -----------
    latest_report,last_expid, prev_date=check_latest_report(report_dir)
    print('Reading in latest report:',latest_report)
    print('Previous largest expid:',last_expid)
    n_proc_now, pix_list,sdir_list,expid_range,prev_date,report_data=check_spectra64(report_dir,path,out_path,mwonly,last_expid,prev_date,whole_spectra64) 
    min_expid=last_expid
    
    #== Store all the scripts in a folder named with Today's date
    yr= datetime.date.today().year
    month=datetime.date.today().month
    day=datetime.date.today().day
    now=str(yr*10000+month*100+day)
	
	#== Create directories if not existing.    
    for i,pixel in enumerate(pix_list):
    	sdir=sdir_list[i]
        
    	if not os.path.exists(os.path.join(out_path,sdir)): os.mkdir(os.path.join(out_path,sdir))
    	if not os.path.exists(os.path.join(out_script_path,now)): os.mkdir(os.path.join(out_script_path,now))
    	if not os.path.exists(os.path.join(out_script_path,now,sdir)): os.mkdir(os.path.join(out_script_path,now,sdir))
    	if not os.path.exists(os.path.join(out_script_path,now,sdir,pixel)): os.mkdir(os.path.join(out_script_path,now,sdir,pixel))
    	if not os.path.exists(os.path.join(out_path,sdir,pixel)): os.mkdir(os.path.join(out_path,sdir,pixel))

    # Determining how many jobs to be submitted 
    pix_gp_list,sdir_gp_list,fn_sum_list,job_ind,sort_index=cal_node_n(n_proc_now,pix_list,sdir_list,nthreads)
    generate_reports(path,report_dir,expid_range,prev_date,report_data,job_ind,sort_index)	
    
    #== Writing slurm scripts and lists of input files (x.xxx) for each job (group jobs according to fiber numbers and processors per node)
    n_node=len(pix_gp_list) # requested node number = pixel group number
    print('Number of jobs to be submitted:',n_node+len(pix_list))
    for i in range(n_node):   	
    	file_ind=str(1000+i)[1:]
    	
    	#== Generating files that store lists of input files
    	f=open(os.path.join(out_script_path,now,'x.'+file_ind),'w')
    	sub_pixels=pix_gp_list[i]
    	sub_sdirs=sdir_gp_list[i]
    	for pixel,sdir in zip(sub_pixels,sub_sdirs):
    		entry=path+'/'+str(sdir)+'/'+str(pixel)
    		f.write(entry+"/spectra-64-"+str(pixel)+".fits \n")
    		#print(entry+"/spectra-64-"+str(pixel)+".fits")
    	f.close()
    	suffix='rvspecfit_'+file_ind 
    		
    	#== Generating slurm scripts
    	n_fiber=fn_sum_list[i]
    	write_slurm_tot(out_script_path,out_path,file_ind,path,sdir_gp_list[i],pix_gp_list[i],min_expid,n_fiber,nthreads, suffix,whole_spectra64,mwonly)
    	for sdir,pixel in zip(sdir_list,pix_list):
    		cmd="python3 -c \"import sys; sys.path.insert(0, '"+python_path+"'); from piferre import write_slurm; write_slurm(\'"+str(sdir)+"\',\'"+str(pixel)+"\',\'"+str(out_path)+"\', script_path='"+os.path.join(out_script_path,now)+"', ngrids=9, nthreads=4)\"\n"
    		#print('cmd=',cmd)
    		err=subprocess.call(cmd,shell=True)
    		   		
    	write_script_tot_gp(os.path.join(out_script_path),out_path,path,sdir_gp_list[i],pix_gp_list[i],file_ind,min_expid,nthreads=nthreads, suffix=suffix, mwonly=mwonly)
    		
    #== Executing the shell script and submitting jobs	
    run_scripts(os.path.join(out_script_path,now))
    	
    
if __name__ == "__main__":

	proc_mws(sys.argv[1:])
