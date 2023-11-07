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

name = Final_Config.NAME

job_file = os.path.join(job_directory, "%s.job" % name)
print(name)
with open(job_file,"w") as fh:
    fh.writelines("#!/bin/bash\n")
    fh.writelines("#SBATCH -o .out/%s.o\n" % name)
    fh.writelines("#SBATCH --mem=200g\n")
    fh.writelines("#SBATCH --nodes=1\n")
    fh.writelines("#SBATCH --ntasks-per-node=1\n")
    fh.writelines("#SBATCH --cpus-per-task=16\n")
    fh.writelines("#SBATCH --partition=gpuA100x4\n") 
    fh.writelines("#SBATCH --gres=gpu:1\n")
    fh.writelines("#SBATCH --account=bbou-delta-gpu\n")
    fh.writelines("#SBATCH --time=00:10:00\n")
    fh.writelines("### GPU options ###\n")
    fh.writelines("#SBATCH --gpus-per-node=1\n")
    fh.writelines("##SBATCH --gpu-bind=verbose\n")

    fh.writelines("python train_and_eval.py")

os.system("sbatch %s" % job_file)