#!/usr/bin/env python3
"""Conservative two-lane training queue. --dry-run validates without launching jobs."""
import argparse
import csv
import fcntl
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

import yaml

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / '.venv/bin/python'
POLL_SECONDS = 30


def now():
    return datetime.now(timezone.utc).isoformat()


def path(value):
    return (ROOT / value).resolve()


def atomic_json(target, value):
    temporary = target.with_name(target.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)
    directory_fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def process_identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return {'start_ticks': int(fields[19]), 'state': fields[0]}
    except FileNotFoundError:
        return None


def same_process(dependency):
    identity = process_identity(dependency['pid'])
    return (identity is not None and identity['start_ticks'] == dependency['start_ticks']
            and identity['state'] not in ('Z', 'X'))


def check_history(run_dir, epochs):
    rows = [json.loads(line) for line in (run_dir / 'history.jsonl').read_text().splitlines() if line.strip()]
    if [row.get('epoch') for row in rows] != list(range(1, epochs + 1)):
        raise ValueError(f'{run_dir}: history must contain exactly epochs 1..{epochs}')
    if rows[-1].get('eval_stride') != 1 or not rows[-1].get('validation'):
        raise ValueError(f'{run_dir}: final full validation is missing')
    for filename in ('last.pt', 'best_miou.pt', 'best_loss.pt'):
        if not (run_dir / filename).is_file() or (run_dir / filename).stat().st_size == 0:
            raise ValueError(f'{run_dir}: missing/empty {filename}')


def validate(plan):
    if plan.get('version') != 1 or len(plan.get('lanes', [])) != 2:
        raise ValueError('Expected version 1 with two lanes')
    names, outputs, devices, lane_ids = set(), set(), set(), set()
    config_contents = {}
    for lane in plan['lanes']:
        if lane['id'] in lane_ids or len(lane['gpus']) != 2 or devices.intersection(lane['gpus']):
            raise ValueError('Lane IDs and GPU assignments must be unique')
        if not all(type(gpu) is int and gpu >= 0 for gpu in lane['gpus']):
            raise ValueError('GPU indices must be nonnegative integers')
        lane_ids.add(lane['id'])
        devices.update(lane['gpus'])
        dep = lane['predecessor']
        if type(dep['pid']) is not int or dep['pid'] <= 0 or type(dep['start_ticks']) is not int or dep['start_ticks'] <= 0:
            raise ValueError('A frozen positive PID/start_ticks identity is required')
        resolved = json.loads((path(dep['run_dir']) / 'resolved_config.json').read_text())
        if resolved['train']['epochs'] != dep['expected_epochs'] or path(resolved['train']['output_dir']) != path(dep['run_dir']):
            raise ValueError('Predecessor saved configuration does not match its run/epochs')
        if not lane['jobs']:
            raise ValueError('Each lane needs jobs')
        for job in lane['jobs']:
            config_path = path(job['config'])
            content = config_path.read_bytes()
            config = yaml.safe_load(content)
            train = config['train']
            output = path(train['output_dir'])
            if job['name'] in names or output in outputs:
                raise ValueError('Job names and output directories must be unique')
            if config_path.stem != job['name'] or Path(train['output_dir']).name != job['name']:
                raise ValueError(f"{job['name']}: config/output name mismatch")
            if output.parent != path('runs') or output == path(dep['run_dir']):
                raise ValueError('Job output must be a fresh direct child of runs')
            if output.exists() and (not output.is_dir() or any(output.iterdir())):
                raise ValueError(f'Refusing existing run outputs: {output}')
            if (train['batch_size'], train['accumulation_steps'], train['epochs']) != (8, 4, job['expected_epochs']):
                raise ValueError('Expected batch_size=8, accumulation_steps=4 and matching epochs')
            if job['expected_epochs'] != 25 or train.get('min_effective_batch') != 64 or train.get('resume') is not None:
                raise ValueError('Expected fresh 25-epoch runs with effective batch 64')
            names.add(job['name'])
            outputs.add(output)
            job.update(config=str(config_path), run_dir=str(output))
            config_contents[job['name']] = content
        dep['run_dir'] = str(path(dep['run_dir']))
    if not PYTHON.is_file():
        raise ValueError(f'Python interpreter is missing: {PYTHON}')
    return config_contents


def gpu_occupancy(gpus):
    def query(arguments):
        result = subprocess.run(['nvidia-smi', *arguments], capture_output=True, text=True, timeout=20, check=True)
        return list(csv.reader(io.StringIO(result.stdout), skipinitialspace=True))
    gpu_rows = query(['--query-gpu=index,uuid', '--format=csv,noheader,nounits'])
    mapping = {int(row[0]): row[1].strip() for row in gpu_rows if len(row) == 2}
    if len(mapping) != len(gpu_rows) or any(gpu not in mapping for gpu in gpus):
        raise ValueError('Incomplete NVIDIA GPU UUID mapping')
    selected = {mapping[gpu] for gpu in gpus}
    processes = query(['--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'])
    occupied = []
    for row in processes:
        if len(row) != 2 or not row[0].strip().startswith('GPU-') or not row[1].strip().isdigit():
            raise ValueError(f'Unexpected NVIDIA process row: {row}')
        if row[0].strip() in selected:
            occupied.append({'gpu_uuid': row[0].strip(), 'pid': int(row[1])})
    return {'gpu_uuids': {str(gpu): mapping[gpu] for gpu in gpus}, 'occupied': occupied}


def freeze_configs(plan, config_contents):
    snapshots = path('runs/plans/2026-10-23/configs')
    snapshots.mkdir(parents=True, exist_ok=True)
    for lane in plan['lanes']:
        for job in lane['jobs']:
            original = Path(job['config'])
            content = original.read_bytes()
            if content != config_contents[job['name']]:
                raise ValueError(f'Configuration changed during startup: {original}')
            frozen = snapshots / original.name
            if frozen.exists():
                if frozen.read_bytes() != content:
                    raise ValueError(f'Conflicting frozen configuration: {frozen}')
            else:
                with frozen.open('xb') as stream:
                    stream.write(content)
                    stream.flush()
                    os.fsync(stream.fileno())
            job.update(original_config=str(original), config=str(frozen))


class Queue:
    def __init__(self, plan, prefix, config_contents):
        self.plan, self.prefix, self.mutex = plan, prefix, threading.RLock()
        self.config_contents = config_contents
        self.logfile = prefix.with_suffix('.log').open('a', buffering=1)
        self.state = {'version': 1, 'status': 'running', 'pid': os.getpid(),
                      'start_ticks': process_identity(os.getpid())['start_ticks'], 'started_at': now(),
                      'project_root': str(ROOT), 'runs_root': str(path('runs')),
                      'lanes': {lane['id']: dict(lane, status='pending') for lane in plan['lanes']}}
        for lane in self.state['lanes'].values():
            lane['jobs'] = [dict(job, status='pending', gpus=lane['gpus']) for job in lane['jobs']]
        self.save()

    def save(self):
        with self.mutex:
            self.state['updated_at'] = now()
            atomic_json(self.prefix.with_suffix('.status.json'), self.state)

    def update(self, target, **changes):
        with self.mutex:
            target.update(changes)
            self.save()
            message = f"{now()} {target.get('id', target.get('name', 'queue'))}: {json.dumps(changes)}"
            print(message, flush=True)
            print(message, file=self.logfile, flush=True)

    def lane(self, lane):
        current_job = None
        try:
            dep = lane['predecessor']
            self.update(lane, status='waiting_dependency')
            while same_process(dep):
                time.sleep(POLL_SECONDS)
            check_history(Path(dep['run_dir']), dep['expected_epochs'])
            self.update(lane, dependency_completed_at=now())
            for job in lane['jobs']:
                current_job = job
                self.update(lane, status='waiting_gpus', current_job=job['name'])
                while True:
                    try:
                        gpu_state = gpu_occupancy(lane['gpus'])
                    except (subprocess.SubprocessError, OSError, ValueError) as error:
                        self.update(lane, gpu_query_error=str(error), gpu_checked_at=now())
                        time.sleep(POLL_SECONDS)
                        continue
                    self.update(lane, **gpu_state, gpu_query_error=None, gpu_checked_at=now())
                    if not gpu_state['occupied']:
                        break
                    time.sleep(POLL_SECONDS)
                config_path, run_dir = Path(job['config']), Path(job['run_dir'])
                if config_path.read_bytes() != self.config_contents[job['name']]:
                    raise ValueError(f'Configuration changed since queue startup: {config_path}')
                if run_dir.exists() and any(run_dir.iterdir()):
                    raise ValueError(f'Refusing existing run outputs: {run_dir}')
                run_dir.mkdir(parents=False, exist_ok=True)
                env = os.environ.copy()
                env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=','.join(map(str, lane['gpus'])),
                           PYTHON=str(PYTHON), OMP_NUM_THREADS=env.get('OMP_NUM_THREADS', '4'))
                if set(lane['gpus']) == {5, 6}:
                    env['NCCL_P2P_DISABLE'] = '1'
                else:
                    env.pop('NCCL_P2P_DISABLE', None)
                command = [str(PYTHON), '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
                           '-m', 'moonseg.train', '--config', str(config_path)]
                self.update(job, status='launching', command=command, started_at=now())
                with (run_dir / 'train.log').open('x') as log:
                    child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, close_fds=True)
                    self.update(job, status='running', pid=child.pid, start_ticks=process_identity(child.pid)['start_ticks'])
                    self.update(lane, status='running')
                    exit_code = child.wait()
                self.update(job, exit_code=exit_code, exited_at=now())
                if exit_code != 0:
                    raise RuntimeError(f"{job['name']} exited {exit_code}")
                check_history(run_dir, job['expected_epochs'])
                marker = dict(status='completed', name=job['name'], run_dir=str(run_dir), config=str(config_path),
                              expected_epochs=job['expected_epochs'], exit_code=0, pid=child.pid, completed_at=now())
                atomic_json(run_dir / 'queue_completed.json', marker)
                self.update(job, status='completed', completed_at=marker['completed_at'])
                current_job = None
            self.update(lane, status='completed', completed_at=now())
        except Exception as error:
            if current_job is not None:
                self.update(current_job, status='failed', error=str(error), failed_at=now())
            self.update(lane, status='failed', error=str(error), failed_at=now())

    def run(self):
        threads = [threading.Thread(target=self.lane, args=(lane,)) for lane in self.state['lanes'].values()]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        success = all(lane['status'] == 'completed' for lane in self.state['lanes'].values())
        self.update(self.state, status='completed' if success else 'failed', finished_at=now())
        return 0 if success else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=ROOT / 'runs/plans/2026-10-23/training.json')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--status-prefix', type=Path, default=ROOT / 'runs/queue_2026-10-23')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    config_contents = validate(plan)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    prefix = args.status_prefix.resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with path('runs/.training_queue.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another queue instance holds the exclusive lock') from None
        if prefix.with_suffix('.status.json').exists():
            raise RuntimeError('Existing queue status requires inspection; automatic restart is disabled')
        freeze_configs(plan, config_contents)
        return Queue(plan, prefix, config_contents).run()


if __name__ == '__main__':
    sys.exit(main())
