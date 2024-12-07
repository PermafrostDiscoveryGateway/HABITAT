from final_model_config import *
import glob
import os

def mkdir_p(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

job_directory = "%s/.job" % os.getcwd()
out_directory = "%s/.out" % os.getcwd()

mkdir_p(job_directory)
mkdir_p(out_directory)

name = 'train_eval_' + Final_Config.NAME

job_file = os.path.join(job_directory, "%s.job" % name)
print(name)
with open(job_file,"w") as fh:
    fh.writelines("#!/bin/bash\n")
    fh.writelines("#SBATCH -o .out/%s.o\n" % name)
    fh.writelines("#SBATCH -e .error/%s.e\n" % name)
    fh.writelines("#SBATCH -N 1\n")
    fh.writelines("#SBATCH -n 1\n")
    fh.writelines("#SBATCH -p gpu-a100\n") 
    fh.writelines("#SBATCH -A DPP20001\n")
    fh.writelines("#SBATCH -t 06:00:00\n")

    fh.writelines("python train_and_eval.py")

os.system("sbatch %s" % job_file)