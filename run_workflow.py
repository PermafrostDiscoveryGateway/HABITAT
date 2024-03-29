from operational_config import *
import glob
import os

# Function for creating folders to hold SLURM job files and output files
def mkdir_p(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

# Define path to input satellite image scenes and create sorted list of filenames
file_path = Operational_Config.INPUT_SCENE_DIR+ "/*.tif"

files = sorted(glob.glob(file_path))

# Define at which files the pipeline will start start and finish at
start = 0
end = 2
selected_files = files[start:end]

# Store the names of the files selected for processing in a list
file_names = []
for file in selected_files:
    file_names.append(file.split('/')[-1])

# Create job and output file directories if they do not exist
job_directory = "%s/.job" % os.getcwd()
out_directory = "%s/.out" % os.getcwd()

mkdir_p(job_directory)
mkdir_p(out_directory)

# For each selected file, create a SLURM job with all of the specified parameters. 
# Adjust parameter values as needed. Each job will automatically be submitted to the queue.
count = start
for idx,name in enumerate(file_names):
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
        fh.writelines("#SBATCH --time=03:00:00\n")
        fh.writelines("### GPU options ###\n")
        fh.writelines("#SBATCH --gpus-per-node=1\n")
        fh.writelines("##SBATCH --gpu-bind=verbose\n")

        fh.writelines("python full_pipeline.py --image=%s\n"%name)

    count = count + 1
    os.system("sbatch %s" % job_file)