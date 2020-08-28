import os
import sys

if __name__ == '__main__':
    if len(sys.argv) > 0:
        working_directory = sys.argv[1]
    else:
        working_directory = os.getcwd()

    for directory in os.listdir(working_directory):
        directory = os.path.join(working_directory, directory)
        job_filename = os.path.join(directory, 'slurm.job')
        if os.path.isdir(directory) and os.path.isfile(job_filename):
            os.system(f'sbatch {job_filename}')
