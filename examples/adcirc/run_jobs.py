#! /usr/bin/env python

import os
from pathlib import Path
import sys

if __name__ == '__main__':
    working_directory = Path(sys.argv[1]) if len(sys.argv) > 0 else Path.cwd()
    for directory in working_directory.iterdir():
        if directory.is_dir():
            job_filename = directory / 'slurm.job'
            if job_filename.exists():
                os.system(f'sbatch {job_filename}')
