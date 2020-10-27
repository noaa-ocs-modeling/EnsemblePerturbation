from datetime import timedelta
from enum import Enum
from os import PathLike
from pathlib import Path
import textwrap
from typing import Sequence
import uuid

import numpy


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
        hpc: HPC,
        basename: str = None,
        directory: str = None,
        launcher: str = None,
        run: str = None,
        email_type: SlurmEmailType = None,
        email_address: str = None,
        error_filename: str = None,
        log_filename: str = None,
        nodes: int = None,
        modules: [str] = None,
        path_prefix: str = None,
        commands: [str] = None,
    ):
        """
        Instantiate a new Slurm shell script (`*.job`).

        :param account: Slurm account name
        :param tasks: number of total tasks for Slurm to run
        :param duration: duration to run job in job manager
        :param partition: partition to run on
        :param hpc: HPC to run script on
        :param basename: file name of driver shell script
        :param directory: directory to run in
        :param launcher: command to start processes on target system (`srun`, `ibrun`, etc.)
        :param run: Slurm run name
        :param email_type: email type
        :param email_address: email address
        :param error_filename: file path to error log file
        :param log_filename: file path to output log file
        :param nodes: number of physical nodes to run on
        :param modules: list of file paths to modules to load
        :param path_prefix: file path to prepend to the PATH
        :param commands: list of extra shell commands to insert into script
        """

        if isinstance(modules, Sequence) and len(modules) == 0:
            modules = None

        self.account = account
        self.tasks = tasks
        self.duration = duration
        self.partition = partition
        self.hpc = hpc

        self.basename = basename if basename is not None else 'slurm.job'
        self.directory = directory if directory is not None else '.'
        self.launcher = launcher if launcher is not None else 'srun'

        self.run = run if run is not None else uuid.uuid4().hex
        self.email_type = email_type
        self.email_address = email_address

        self.error_filename = error_filename if error_filename is not None else 'slurm.log'
        self.log_filename = log_filename if log_filename is not None else 'slurm.log'
        self.nodes = nodes
        self.modules = modules
        self.path_prefix = path_prefix
        self.commands = commands

    @property
    def tasks(self) -> int:
        return self.__tasks

    @tasks.setter
    def tasks(self, tasks: int = 1):
        self.__tasks = int(tasks)

    @property
    def nodes(self) -> int:
        return self.__nodes

    @nodes.setter
    def nodes(self, nodes: int):
        if nodes is None and self.hpc == HPC.TACC:
            nodes = numpy.ceil(self.tasks / 68)
        self.__nodes = int(nodes)

    @property
    def configuration(self) -> str:
        lines = [f'#SBATCH -D {self.directory}', f'#SBATCH -J {self.run}']

        if self.account is not None:
            lines.append(f'#SBATCH -A {self.account}')
        if self.email_type not in (None, SlurmEmailType.NONE):
            lines.append(f'#SBATCH --mail-type={self.email_type.value}')
            if self.email_address is None or len(self.email_address) == 0:
                raise ValueError('missing email address')
            lines.append(f'#SBATCH --mail-user={self.email_address}')
        if self.error_filename is not None:
            lines.append(f'#SBATCH --error={self.error_filename}')
        if self.log_filename is not None:
            lines.append(f'#SBATCH --output={self.log_filename}')

        if self.nodes is not None:
            lines.append(f'#SBATCH -N {self.nodes}')
        lines.append(f'#SBATCH -n {self.tasks}')

        hours, remainder = divmod(self.duration, timedelta(hours=1))
        minutes, remainder = divmod(remainder, timedelta(minutes=1))
        seconds = round(remainder / timedelta(seconds=1))
        lines.extend(
            [
                f'#SBATCH --time={hours:02}:{minutes:02}:{seconds:02}',
                f'#SBATCH --partition={self.partition}',
            ]
        )

        return '\n'.join(lines)

    def __str__(self) -> str:
        lines = [
            self.shebang,
            self.configuration,
            '',
            'set -e',
            '',
        ]

        if self.modules is not None:
            modules_string = ' '.join(module for module in self.modules)
            lines.extend(
                [f'module load {modules_string}', '', ]
            )

        if self.path_prefix is not None:
            lines.extend(
                [f'PATH={self.path_prefix}:$PATH', '', ]
            )

        if self.commands is not None:
            lines.extend(
                [*(str(command) for command in self.commands), '', ]
            )

        lines.extend(
            [
                bash_function(
                    'main',
                    bash_for_loop(
                        'for directory in ./*/',
                        [
                            'echo "Starting configuration $directory..."',
                            'cd "$directory"',
                            'SECONDS=0',
                            'run_coldstart_phase',
                            bash_if_statement(
                                f'if grep -Rq "ERROR: Elevation.gt.ErrorElev, ADCIRC stopping." {self.log_filename}',
                                [
                                    'duration=$SECONDS',
                                    'echo "ERROR: Elevation.gt.ErrorElev, ADCIRC stopping."',
                                    'echo "Wallclock time: $($duration / 60) minutes and $($duration % 60) seconds."',
                                    'exit -1',
                                ],
                                'else',
                                [
                                    'run_hotstart_phase',
                                    'duration=$SECONDS',
                                    bash_if_statement(
                                        f'if grep -Rq "ERROR: Elevation.gt.ErrorElev, ADCIRC stopping." {self.log_filename}',
                                        [
                                            'echo "ERROR: Elevation.gt.ErrorElev, ADCIRC stopping."',
                                            'echo "Wallclock time: $($duration / 60) minutes and $($duration % 60) seconds."',
                                            'exit -1',
                                        ],
                                    ),
                                ],
                            ),
                            'echo "Wallclock time: $($duration / 60) minutes and $($duration % 60) seconds."',
                            'cd ..',
                        ],
                    ),
                ),
                '',
                bash_function(
                    'run_coldstart_phase',
                    [
                        'rm -rf ./coldstart/*',
                        'cd ./coldstart',
                        'ln -sf ../fort.13 ./fort.13',
                        'ln -sf ../fort.14 ./fort.14',
                        'ln -sf ../fort.15.coldstart ./fort.15',
                        'ln -sf ../../nems.configure.coldstart ./nems.configure',
                        'ln -sf ../../model_configure.coldstart ./model_configure',
                        'ln -sf ../../atm_namelist.rc.coldstart ./atm_namelist.rc',
                        'ln -sf ../../config.rc.coldstart ./config.rc',
                        'adcprep --np $SLURM_NTASKS --partmesh',
                        'adcprep --np $SLURM_NTASKS --prepall',
                        'ibrun NEMS.x',
                        'clean_directory',
                        'cd ..',
                    ],
                ),
                '',
                bash_function(
                    'run_hotstart_phase',
                    [
                        'rm -rf ./hotstart/*',
                        'cd ./hotstart',
                        'ln -sf ../fort.13 ./fort.13',
                        'ln -sf ../fort.14 ./fort.14',
                        'ln -sf ../fort.15.hotstart ./fort.15',
                        'ln -sf ../coldstart/fort.67.nc ./fort.67.nc',
                        'ln -sf ../../nems.configure.hotstart ./nems.configure',
                        'ln -sf ../../nems.configure.hotstart ./nems.configure',
                        'ln -sf ../../model_configure.hotstart ./model_configure',
                        'ln -sf ../../atm_namelist.rc.hotstart ./atm_namelist.rc',
                        'ln -sf ../../config.rc.hotstart ./config.rc',
                        'adcprep --np $SLURM_NTASKS --partmesh',
                        'adcprep --np $SLURM_NTASKS --prepall',
                        'ibrun NEMS.x',
                        'clean_directory',
                        'cd ..',
                    ],
                ),
                '',
                bash_function(
                    'clean_directory',
                    [
                        f'rm -rf {pattern}'
                        for pattern in [
                        'PE*',
                        'partmesh.txt',
                        'metis_graph.txt',
                        'fort.13',
                        'fort.14',
                        'fort.15',
                        'fort.16',
                        'fort.80',
                        'fort.68.nc',
                        'nems.configure',
                            'model_configure',
                            'atm_namelist.rc',
                            'config.rc',
                        ]
                    ],
                ),
                '',
                'main',
            ]
        )

        return '\n'.join(lines)

    def write(self, filename: PathLike, overwrite: bool = False):
        """
        Write Slurm script to file.

        :param filename: path to output file
        :param overwrite: whether to overwrite existing files
        """

        if not isinstance(filename, Path):
            filename = Path(filename)

        if filename.is_dir():
            filename = filename / 'slurm.job'

        output = f'{self}\n'
        if overwrite or not filename.exists():
            with open(filename, 'w') as file:
                file.write(output)


def bash_if_statement(
    condition: str, then: [str], *else_then: [[str]], indentation: str = '  '
) -> str:
    """
    Create a if statement in Bash syntax using the given condition, then statement(s), and else condition(s) / statement(s).

    :param condition: boolean condition to check
    :param then: Bash statement(s) to execute if condition is met
    :param else_then: arbitrary number of Bash statement(s) to execute if condition is not met, with optional conditions (`elif`)
    :param indentation: indentation
    :return: if statement
    """

    if not isinstance(then, str) and isinstance(then, Sequence):
        then = '\n'.join(then)

    condition = str(condition).strip('if ').strip('; then')

    lines = [f'if {condition}; then', textwrap.indent(then, indentation)]

    for else_block in else_then:
        if not isinstance(else_block, str) and isinstance(else_block, Sequence):
            else_block = '\n'.join(else_block)

        currently_else = else_block.startswith('else')
        currently_elif = else_block.startswith('elif')
        currently_then = else_block.startswith('then')

        previous_line = lines[-1].strip()
        hanging_else = previous_line.startswith('else') and not previous_line.endswith(';')
        hanging_elif = previous_line.startswith('elif') and not previous_line.endswith(';')

        if currently_else or currently_elif:
            if hanging_else or hanging_elif:
                lines.remove(-1)
            if currently_else:
                lines.append(else_block)
            elif currently_elif:
                else_block.strip('elif ')
                lines.append(f'elif {else_block}')
        else:
            if not hanging_else and not hanging_elif:
                lines.append('else')
            elif hanging_elif:
                lines[-1].append(';')
                if not currently_then:
                    lines[-1].append(' then')
            lines.append(textwrap.indent(else_block, indentation))

    lines.append('fi')

    return '\n'.join(lines)


def bash_for_loop(iteration: str, do: [str], indentation='  ') -> str:
    """
    Create a for loop in Bash syntax using the given variable, iterator, and do statement(s).

    :param iteration: for loop statement, such as `for dir in ./*`
    :param do: Bash statement(s) to execute on every loop iteration
    :param indentation: indentation
    :return: for loop
    """

    if not isinstance(do, str) and isinstance(do, Sequence):
        do = '\n'.join(do)

    return '\n'.join((f'{iteration}; do', textwrap.indent(do, indentation), 'done',))


def bash_function(name: str, body: [str], indentation: str = '  ') -> str:
    """
    Create a function in Bash syntax using the given name and function statement(s).

    :param name: name of function
    :param body: Bash statement(s) making up function body
    :param indentation: indentation
    :return: function
    """

    if not isinstance(body, str) and isinstance(body, Sequence):
        body = '\n'.join(body)

    return '\n'.join([f'{name}() {{', textwrap.indent(body, indentation), '}'])
