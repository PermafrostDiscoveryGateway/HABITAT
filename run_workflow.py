from operational_config import *
import glob
import os

def mkdir_p(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

file_path = Operational_Config.INPUT_SCENE_DIR+ "/*.tif"

files = sorted(glob.glob(file_path))

start = 508
end = 538
selected_files = files[start:end]


file_names = []
for file in selected_files:
    file_names.append(file.split('/')[-1])

job_directory = "%s/.job" % os.getcwd()
out_directory = "%s/.out" % os.getcwd()
error_directory = "%s/.error" % os.getcwd()

mkdir_p(job_directory)
mkdir_p(out_directory)
mkdir_p(error_directory)

count = start
for idx,name in enumerate(file_names):
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
        fh.writelines("#SBATCH -t 02:00:00\n")

        fh.writelines("python full_pipeline.py --image=%s\n"%name)

    count = count + 1
    os.system("sbatch %s" % job_file)