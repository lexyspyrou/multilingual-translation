import argparse
import datetime
import itertools
import os
import random
import shlex
import shutil
import subprocess
from collections import OrderedDict
from glob import glob
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser('Script for launching hyperparameter sweeps')
    parser.add_argument('-d', '--data', required=True, help='path to data directory')
    parser.add_argument('-p', '--prefix', required=True,
                        help='save checkpoints and logs in <checkpoints-dir>/<prefix>.<save_dir_key>')
    parser.add_argument('-t', '--num-trials', required=True, type=int,
                        help='number of random hyperparam configurations to try (-1 for grid search)')
    parser.add_argument('-g', '--num-gpus', type=int, required=True, help='number of GPUs per node')
    parser.add_argument('--gpu-type', type=str, default='', choices=['', 'volta'])
    parser.add_argument('-n', '--num-nodes', type=int, default=1, help='number of nodes for distributed training')
    parser.add_argument('--mem', '--mem', help='memory to request')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--baseline-model', help='path to baseline model from which to resume training')
    parser.add_argument('--checkpoints-dir',
                        default=os.path.join('/checkpoint', os.environ['USER'], str(datetime.date.today())),
                        help='save checkpoints and logs in <checkpoints-dir>/<prefix>.<save_dir_key>')
    parser.add_argument('--resume-failed', action='store_true',
                        help='resume any runs that failed (assumes --num-trials and --seed are the same)')
    parser.add_argument('--resume-finished', action='store_true',
                        help='force any runs that finished to begin again (uncommon)')
    parser.add_argument('--dry-run', action='store_true',
                        help='output only a list of actions to perform without performing them')
    parser.add_argument('--local', action='store_true',
                        help='run locally instead of submitting remote job')
    parser.add_argument('--partition', help='partition to run on', default='learnfair')
    parser.add_argument('--reservation', help='reservation to run on')
    parser.add_argument('--exclusive', action='store_true',
                        help='if set, get exclusive host')
    parser.add_argument('--dep', metavar='JOBID', type=int,
                        help='add JOBID as a dependency (i.e., wait for it to finish)')
    parser.add_argument('--sequential', action='store_true',
                        help='schedule jobs to run sequentially')
    parser.add_argument('--time', default='4320',
                        help='expected job duration in minutes')
    parser.add_argument('--constraint', metavar='CONSTRAINT',
                        help='gpu constraint, if any. e.g. "volta"')
    parser.add_argument('--one-task', action='store_true',
                        help='if true, starts one task per node instead of num gpus')
    parser.add_argument('--comment', help='comment string')
    parser.add_argument('--snapshot-code', action='store_true', default=False,
                        help='Flag for creating a snapshot of training code while creating slurm job,'
                             ' path is "./slurm_snapshot_code/<TIME_ISO_FORMAT/>:", '
                             'can find time from comment of slurm job.')
    parser.add_argument('--tensorboard-logdir',
                        default=os.path.join('/checkpoint', os.environ['USER'], 'tensorboard_logs',
                                             str(datetime.date.today())),
                        help='save tensorboard logs in <tensorboard-logdir>/<prefix>.<save_dir_key>')
    parser.add_argument('--log-tensorboard', action='store_true', help='enable tensorboard logging')

    parser.add_argument('--post-steps', nargs='+',
                        help='additional steps to execute after the primary job is complete. '
                             'this can be a file with the steps, or a string. some placeholders such as '
                             '{job_dir} will be replaced')
    parser.add_argument('--skip-primary-cmd', action='store_true',
                        help='if set, skips the primary run (useful if you want to only execute post steps')

    args = parser.parse_args()
    return args


class hyperparam(object):
    """Base class for defining hyperparameters."""

    def __init__(self, name, values=None, binary_flag=False, save_dir_key=None):
        """
        Arguments:
        - name : the name of the hyperparameter (e.g., `--dropout`)
        - values : the set of values to sweep over (e.g., `[0.0, 0.1, 0.2]`)
        - binary_flag : whether the hyperparameter uses a boolean flag (e.g., `--no-save`)
        - save_dir_key : function that takes the hyperparameter value and returns the "key"
                         to be appended to the output directory name
        """
        self.name = name
        if values is None:  # syntactic sugar for binary flags
            self.values = [True]
            self.binary_flag = True
        else:
            self.values = values if isinstance(values, list) else [values]
            self.binary_flag = binary_flag
        self.save_dir_key = save_dir_key
        self.current_value = None

        if len(self.values) > 1 and self.save_dir_key is None:
            raise ValueError(f'{name} has more than one value but is missing a save_dir_key!')

    def get_cli_args(self):
        if self.binary_flag:
            return [self.name] if self.current_value else []
        else:
            return [self.name, self.current_value]

    def get_save_dir_key(self):
        if self.save_dir_key is None:
            return None
        if self.binary_flag:
            return self.save_dir_key(1) if self.current_value else None
        return self.save_dir_key(self.current_value)


def main(get_grid, postprocess_hyperparams):
    args = get_args()

    if args.local:
        args.num_nodes = 1

    # compute all possible hyperparameter configurations
    grid = get_grid(args)
    grid_product = list(itertools.product(*[hp.values for hp in grid]))

    # randomly shuffle configurations
    random.seed(args.seed)
    random.shuffle(grid_product)

    for i, hp_values in enumerate(grid_product):
        config = OrderedDict()
        for hp, value in zip(grid, hp_values):
            config[hp.name] = hp
            config[hp.name].current_value = value

        # postprocess hyperparams
        postprocess_hyperparams(args, config)

        # launch training
        job_id = launch_train(args, config)
        print('Launched {}'.format(job_id))

        if args.sequential and not args.local and job_id is not None:
            args.dep = job_id

        if i == args.num_trials - 1:
            break


def copy_all_python_files(source, snapshot_main_dir, code_snapshot_hash):
    """
    Copies following files from source to destination:
        a) all *.py files at direct source location.
        b) all fairseq/*.py recursively.
    """
    os.makedirs(snapshot_main_dir, exist_ok=True)
    destination = os.path.join(snapshot_main_dir, code_snapshot_hash)
    assert not os.path.exists(destination), \
        'Code snapshot: {0} alredy exists'.format(code_snapshot_hash)
    os.makedirs(destination)
    all_pys = glob(os.path.join(source, 'fairseq/**/*.py'), recursive=True) + glob(os.path.join(source, '*.py'))

    for filepath in all_pys:
        directory, filename = os.path.split(filepath)
        if directory:
            os.makedirs(os.path.join(destination, directory), exist_ok=True)
        shutil.copy2(os.path.join(source, filepath), os.path.join(destination, filepath))
    return destination


def launch_train(args, config):
    def dry_run(msg):
        if args.dry_run:
            print(f'| dry-run:  {msg}')
        return args.dry_run

    destination = ''
    if args.snapshot_code:
        # Currently hash is just the current time in ISO format.
        code_snapshot_hash = datetime.datetime.now().isoformat()
        destination = copy_all_python_files('.', 'slurm_snapshot_code', code_snapshot_hash)

    # compute save_dir
    save_dir_key = '.'.join(filter(
        lambda save_dir_key: save_dir_key is not None,
        [hp.get_save_dir_key() for hp in config.values()]
    ))
    save_dir_key = save_dir_key.replace(",", "_")
    save_dir_key = save_dir_key.replace("(", "")
    save_dir_key = save_dir_key.replace(")", "")
    num_total_gpus = args.num_nodes * args.num_gpus
    save_dir = os.path.join(args.checkpoints_dir, f'{args.prefix}.{save_dir_key}.ngpu{num_total_gpus}')
    tensorboard_logdir = os.path.join(args.tensorboard_logdir, f'{args.prefix}.{save_dir_key}.ngpu{num_total_gpus}')

    # create save directory if it doesn't exist
    if not os.path.exists(save_dir):
        if not dry_run(f'create directory: {save_dir}'):
            os.makedirs(save_dir)

        # copy baseline model
        checkpoint_last = os.path.join(save_dir, 'checkpoint_last.pt')
        if args.baseline_model and not os.path.exists(checkpoint_last) and \
                not dry_run(f'initialize with baseline model: {args.baseline_model}'):
            if not os.path.exists(args.baseline_model):
                raise FileNotFoundError(f'Cannot find baseline model: {args.baseline_model}')
            shutil.copyfile(args.baseline_model, checkpoint_last)

    # check for whether the run failed
    if has_finished(save_dir):
        if args.resume_finished:
            dry_run(f'restart previously finished run: {save_dir}')
        else:
            print(f'skip finished run (override with --resume-finished): {save_dir}')
            return
    elif has_failed(save_dir):
        if args.resume_failed:
            dry_run(f'resume failed run: {save_dir}')
        else:
            print(f'skip failed run (override with --resume-failed): {save_dir}')
            return
    # elif has_started(save_dir):
    #     print(f'skip in progress run: {save_dir}')
    #     return

    # generate train command
    train_cmd = ['python', os.path.join(destination, 'train_old.py')]

    post_cmds = []
    if args.post_steps:
        for post_step in args.post_steps:
            if os.path.isfile(post_step):
                post_cmd = Path(post_step).read_text()
            else:
                post_cmd = post_step
            post_cmd = post_cmd.strip().format(job_dir=save_dir)
            post_cmds.append(post_cmd)

    if args.num_nodes > 1:
        train_cmd.extend([
            '--distributed-world-size', str(args.num_nodes * args.num_gpus),
            '--distributed-port', str(get_random_port()),
        ])
    train_cmd.extend([args.data, '--save-dir', save_dir])
    if args.log_tensorboard:
        train_cmd.extend(['--tensorboard-logdir', tensorboard_logdir])
    for hp in config.values():
        train_cmd.extend(map(str, hp.get_cli_args()))
    if args.dry_run:
        train_cmd_str = ' '.join(train_cmd)
        dry_run(f'train command: {train_cmd_str}')
        for post_cmd in post_cmds:
            dry_run(f'post steps command: {post_cmd}')

    # start training
    env = os.environ.copy()
    if args.local:
        assert args.num_nodes == 1, 'distributed training cannot be combined with --local'
        if not dry_run('start training locally'):
            if 'CUDA_VISIBLE_DEVICES' not in env:
                env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, range(args.num_gpus)))
            train_proc = subprocess.Popen(train_cmd, env=env)
            train_proc.wait()
            for post_cmd in post_cmds:
                post_cmd_proc = subprocess.Popen(post_cmd, shell=True, env=env)
                post_cmd_proc.wait()
    else:
        train_log = os.path.join(save_dir, 'train.log')
        train_stderr = os.path.join(save_dir, 'train.stderr.%j')  # %j = slurm job id

        # set environment
        # if args.num_nodes > 1:
        #     env['NCCL_SOCKET_IFNAME'] = '^docker0,lo'

        # build command
        excluded_hosts = os.environ.get('EXCLUDED_HOSTS', None)
        included_hosts = os.environ.get('INCLUDED_HOSTS', None)
        base_srun_cmd = [
            'srun',
            '--job-name', f'{args.prefix}.{save_dir_key}',
            '--output', train_log,
            '--error', train_stderr,
            '--open-mode', 'append',
            '--unbuffered',
        ]
        srun_cmd = base_srun_cmd + train_cmd
        ntasks_per_node = 1 if args.one_task or args.num_nodes == 1 or args.skip_primary_cmd else args.num_gpus
        gres = ':'.join(
            ['gpu'] +
            ([] if args.gpu_type == '' else [args.gpu_type]) +
            [str(args.num_gpus)]
        )
        sbatch_cmd = [
            'sbatch',
            '--job-name', f'{args.prefix}.{save_dir_key}',
            '--gres', gres,
            '--nodes', str(args.num_nodes) if not args.skip_primary_cmd else '1',
            '--ntasks-per-node', str(ntasks_per_node),
            '--cpus-per-task', str(int(8 * args.num_gpus / ntasks_per_node)),
            '--output', train_log,
            '--error', train_stderr,
            '--open-mode', 'append',
            '--signal', 'B:USR1@180',
        ]
        if args.constraint:
            sbatch_cmd += ['-C', args.constraint]

        if args.partition:
            sbatch_cmd += ['--partition', args.partition]
        if args.reservation:
            sbatch_cmd += ['--reservation', args.reservation]
        if args.exclusive:
            sbatch_cmd += ['--exclusive']
        if args.comment:
            comment = args.comment
            if args.snapshot_code:
                comment += ', Code Location: {0}'.format(destination)
            sbatch_cmd += ['--comment', comment]

        if args.snapshot_code:
            sbatch_cmd += ['--comment', 'Code Location: {0}'.format(destination)]

        if args.dep is not None:
            sbatch_cmd.extend(['-d', str(args.dep)])
        if args.time is not None:
            sbatch_cmd.extend(['--time', args.time])
        if args.mem is not None:
            sbatch_cmd += ['--mem', args.mem]
        else:
            sbatch_cmd += ['--mem-per-cpu', '6G']
        sbatch_cmd += ['-x', excluded_hosts] if excluded_hosts is not None else []
        sbatch_cmd += ['-w', included_hosts] if included_hosts is not None else []

        srun_cmd_str = ' '.join(map(shlex.quote, srun_cmd)) if not args.skip_primary_cmd else ''

        for post_cmd in post_cmds:
            post_cmd_str = ' '.join(map(shlex.quote, base_srun_cmd)) + f' {post_cmd}'
            srun_cmd_str = f'({srun_cmd_str} && {post_cmd_str})' if len(srun_cmd_str) > 0 else post_cmd_str

        wrapped_cmd = requeue_support() + '\n' + srun_cmd_str + ' & \n wait $! \n sleep 610 & \n wait $!'

        sbatch_cmd += ['--wrap', wrapped_cmd]
        sbatch_cmd_str = ' '.join(map(shlex.quote, sbatch_cmd))

        if args.dry_run:
            dry_run('start remote training')
            dry_run(f'- log stdout to: {train_log}')
            dry_run(f'- log stderr to: {train_stderr}')
            dry_run(f'- run command: {sbatch_cmd_str}')
            sbatch_cmd += ['--test-only']
            with subprocess.Popen(sbatch_cmd, stdout=subprocess.PIPE, env=env) as train_proc:
                stdout = train_proc.stdout.read().decode('utf-8')
                print(stdout)
        else:
            with open(train_log, 'a') as train_log_h:
                # log most recent git commit
                git_commit = subprocess.check_output(
                    'git log | head -n 1', shell=True, encoding='utf-8')
                print(git_commit.rstrip(), file=train_log_h)
                if args.baseline_model:
                    print(f'baseline model: {args.baseline_model}', file=train_log_h)

                env = os.environ.copy()
                # env['NCCL_SOCKET_IFNAME'] = 'enp1s0f0'
                env['NCCL_DEBUG'] = 'INFO'
                # env['CUDA_LAUNCH_BLOCKING'] = '1'

                print(f'running command: {sbatch_cmd_str}\n')
                with subprocess.Popen(sbatch_cmd, stdout=subprocess.PIPE, env=env) as train_proc:
                    stdout = train_proc.stdout.read().decode('utf-8')
                    print(stdout, file=train_log_h)
                    job_id = int(stdout.rstrip().split()[-1])
                    return job_id


def has_finished(save_dir):
    train_log = os.path.join(save_dir, 'train.log')
    if not os.path.exists(train_log):
        return False
    with open(train_log, 'r') as h:
        lines = h.readlines()
        if len(lines) == 0:
            return False
        if 'done training' in lines[-1]:
            return True
    return False


def has_failed(save_dir):
    if not os.path.exists(save_dir):
        return False

    # find max job id
    job_ids = []
    for fn in os.listdir(save_dir):
        if fn.startswith('train.stderr.'):
            job_ids.append(int(fn.split('.')[-1]))
    if len(job_ids) == 0:
        return False
    max_job_id = max(job_ids)

    def _has_failed(stderr_fn):
        with open(stderr_fn, 'r') as h:
            for line in h:
                if len(line.strip()) > 0:
                    # assume that any output in stderr indicates an error
                    return True
        return False

    return _has_failed(os.path.join(save_dir, f'train.stderr.{max_job_id}'))


def has_started(save_dir):
    train_log = os.path.join(save_dir, 'train.log')
    if not os.path.exists(train_log):
        return False
    return True


def get_random_port():
    return random.randint(30000, 60000)


def requeue_support():
    return """
        trap_handler () {
           echo "Caught signal: " $1
           # SIGTERM must be bypassed
           if [ "$1" = "TERM" ]; then
               echo "bypass sigterm"
           else
             # Submit a new job to the queue
             echo "Requeuing " $SLURM_JOB_ID
             scontrol requeue $SLURM_JOB_ID
           fi
        }


        # Install signal handler
        trap 'trap_handler USR1' USR1
        trap 'trap_handler TERM' TERM
    """
