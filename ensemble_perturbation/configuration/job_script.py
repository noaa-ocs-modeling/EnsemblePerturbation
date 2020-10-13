from datetime import timedelta
from enum import Enum
from os import PathLike
from pathlib import Path
import textwrap
import uuid


class SlurmEmailType(Enum):
    NONE = 'NONE'
    BEGIN = 'BEGIN'
    END = 'END'
    FAIL = 'FAIL'
    REQUEUE = 'REQUEUE'
    STAGE_OUT = 'STAGE_OUT'  # burst buffer stage out and teardown completed
    ALL = 'ALL'  # equivalent to BEGIN, END, FAIL, REQUEUE, and STAGE_OUT)
    TIME_LIMIT = 'TIME_LIMIT'
    TIME_LIMIT_90 = 'TIME_LIMIT_90'  # reached 90 percent of time limit
    TIME_LIMIT_80 = 'TIME_LIMIT_80'  # reached 80 percent of time limit
    TIME_LIMIT_50 = 'TIME_LIMIT_50'  # reached 50 percent of time limit
    ARRAY_TASKS = 'ARRAY_TASKS'  # send emails for each array task


class HPC(Enum):
    TACC = 'TACC'
    ORION = 'ORION'
    HERA = 'HERA'


class EnsembleSlurmScript:
    shebang = '#!/bin/bash --login'

    def __init__(
            self,
            account: str,
            tasks: int,
            duration: timedelta,
            partition: str,
            basename: str = None,
            directory: str = None,
            launcher: str = None,
            run: str = None,
            email_type: SlurmEmailType = None,
            email_address: str = None,
            log_filename: str = None,
            nodes: int = None,
            modules: [str] = None,
            path_prefix: str = None,
            commands: [str] = None,
            hpc: HPC = False
    ):
        """
        Instantiate a new Slurm shell script (`*.job`).

        :param account: Slurm account name
        :param tasks: number of total tasks for Slurm to run
        :param duration: duration to run job in job manager
        :param partition: partition to run on
        :param basename: file name of driver shell script
        :param directory: directory to run in
        :param launcher: command to start processes on target system (`srun`, `ibrun`, etc.)
        :param run: Slurm run name
        :param email_type: email type
        :param email_address: email address
        :param log_filename: file path to output log file
        :param nodes: number of physical nodes to run on
        :param modules: list of file paths to modules to load
        :param path_prefix: file path to prepend to the PATH
        :param commands: list of extra shell commands to insert into script
        :param hpc: HPC to run script on
        """

        self.account = account
        self.tasks = tasks
        self.duration = duration
        self.partition = partition

        self.basename = basename if basename is not None else 'slurm.job'
        self.directory = directory if directory is not None else '.'
        self.launcher = launcher if launcher is not None else 'srun'

        self.run = run if run is not None else uuid.uuid4().hex
        self.email_type = email_type
        self.email_address = email_address

        self.log_filename = log_filename if log_filename is not None else 'slurm.log'
        self.nodes = nodes
        self.modules = modules
        self.path_prefix = path_prefix
        self.commands = commands
        self.hpc = hpc

    @property
    def nodes(self) -> int:
        if self.hpc == HPC.TACC:
            return (self.tasks % 68)[0] + 1
        else:
            return None

    @property
    def configuration(self) -> str:
        lines = [
            f'#SBATCH -D {self.directory}',
            f'#SBATCH -J {self.run}'
        ]

        if self.account is not None:
            lines.append(f'#SBATCH -A {self.account}')
        if self.email_type not in (None, SlurmEmailType.NONE):
            lines.append(f'#SBATCH --mail-type={self.email_type.value}')
            if self.email_address is None or len(self.email_address) == 0:
                raise ValueError('missing email address')
            lines.append(f'#SBATCH --mail-user={self.email_address}')
        if self.log_filename is not None:
            lines.append(f'#SBATCH --output={self.log_filename}')

        if self.nodes is not None:
            lines.append(f'#SBATCH -N {self.nodes}')
        lines.append(f'#SBATCH -n {self.tasks}')

        hours, remainder = divmod(self.duration, timedelta(hours=1))
        minutes, remainder = divmod(remainder, timedelta(minutes=1))
        seconds = round(remainder / timedelta(seconds=1))
        lines.extend((
            f'#SBATCH --time={hours:02}:{minutes:02}:{seconds:02}',
            f'#SBATCH --partition={self.partition}',
        ))

        return '\n'.join(lines)

    def __str__(self) -> str:
        lines = [
            self.configuration,
            'set -e',
        ]

        if self.modules is not None:
            modules_string = ' '.join(module for module in self.modules)
            lines.extend((
                f'',
                f'module load {modules_string}'
            ))

        if self.path_prefix is not None:
            lines.extend((
                f'',
                f'PATH={self.path_prefix}:$PATH'
            ))

        if self.commands is not None:
            lines.append('')
            lines.extend(str(command) for command in self.commands)

        lines.extend([
            bash_function(
                name='main',
                function_block=bash_for_loop(
                    for_variable='directory',
                    directory='/*',
                    for_block='\n'.join([
                        'echo "Starting configuration $directory..."',
                        'cd "$directory"',
                        'SECONDS=0',
                        'run_coldstart_phase',
                        bash_if_statement(
                            if_condition=f'grep -Rq "ERROR: Elevation.gt.ErrorElev, ADCIRC stopping." {self.log_filename}',
                            if_block='\n'.join([
                                'duration=$SECONDS',
                                'echo "ERROR: Elevation.gt.ErrorElev, ADCIRC stopping."',
                                'echo "Wallclock time: $($duration / 60) minutes and $($duration % 60) seconds."',
                                'exit -1',
                            ]),
                            else_blocks=[
                                '\n'.join([
                                    'run_hotstart_phase',
                                    'duration =$SECONDS',
                                    bash_if_statement(
                                        if_condition=f'grep - Rq "ERROR: Elevation.gt.ErrorElev, ADCIRC stopping." {self.log_filename}',
                                        if_block='\n'.join([
                                            'echo "ERROR: Elevation.gt.ErrorElev, ADCIRC stopping."',
                                            'echo "Wallclock time: $($duration / 60) minutes and $($duration % 60) seconds."',
                                            'exit - 1',
                                        ])
                                    )
                                ])
                            ]),
                        'echo "Wallclock time: $($duration / 60) minutes and $($duration % 60) seconds."',
                        'cd.. /',
                    ]))
            ),
            bash_function(
                name='run_coldstart_phase',
                function_block='\n'.join([
                    'rm - rf coldstart',
                    'mkdir coldstart',
                    'cd coldstart',
                    'ln - sf.. / fort.14',
                    'ln - sf.. / fort.13',
                    'ln - sf.. / fort.15.coldstart. / fort.15',
                    'adcprep - -np $SLURM_NTASKS - -partmesh',
                    'adcprep - -np $SLURM_NTASKS - -prepall',
                    'ibrun',
                    'NEMS.x',
                    'clean_directory',
                    'cd..',
                ])
            ),
            bash_function(
                name='run_hotstart_phase',
                function_block='\n'.join([
                    'rm -rf hotstart',
                    'mkdir hotstart',
                    'cd hotstart',
                    'ln -sf ../fort.14',
                    'ln -sf ../fort.13',
                    'ln -sf ../fort.15.hotstart ./fort.15',
                    'ln -sf ../coldstart/fort.67.nc',
                    'adcprep --np $SLURM_NTASKS --partmesh',
                    'adcprep --np $SLURM_NTASKS --prepall',
                    'ibrun NEMS.x',
                    'clean_directory',
                    'cd ..',
                ])
            ),
            bash_function(
                name='clean_directory',
                function_block='\n'.join([
                    'rm - rf PE *',
                    'rm -rf partmesh.txt',
                    'rm -rf metis_graph.txt',
                    'rm -rf fort.13',
                    'rm -rf fort.14',
                    'rm -rf fort.15',
                    'rm -rf fort.16',
                    'rm -rf fort.80',
                    'rm -rf fort.68.nc',
                ])
            ),
            'main',
        ])

        return '\n'.join(lines)

    def write(self, filename: PathLike, overwrite: bool = False):
        """
        Write Slurm script to file.

        :param filename: path to output file
        :param overwrite: whether to overwrite existing files
        """

        if not isinstance(filename, Path):
            filename = Path(filename)

        output = f'{self}\n'
        if overwrite or not filename.exists():
            with open(filename) as file:
                file.write(output)


def bash_if_statement(if_condition: str, if_block: str, else_blocks: [str] = None, indent: str = '  ') -> str:
    lines = [
        f'if {if_condition}; then',
        textwrap.indent(if_block, indent)
    ]

    if else_blocks is not None:
        for else_block in else_blocks:
            if isinstance(else_block, str):
                lines.append('else')
            else:
                assert len(else_block) == 2, f'could not parse else condition / block: {else_block}'
                lines.append(f'elif {else_block[0]}; then')
                else_block = else_block[1]
            lines.append(textwrap.indent(else_block, indent))

    lines.append('fi')

    return '\n'.join(lines)


def bash_for_loop(for_variable: str, directory: str, for_block: str, indent='  ') -> str:
    return '\n'.join((
        f'for {for_variable} in {directory}; do',
        textwrap.indent(for_block, indent),
        'done',
    ))


def bash_function(name: str, function_block: str, indent: str = '  ') -> str:
    return '\n'.join([
        f'{name}() {{',
        textwrap.indent(function_block, indent),
        '}'
    ])
