#!/usr/bin/env python3
"""Run one explicit fine-tuning plan with shared GPU locks and durable receipts."""
import argparse
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_training_queue import (atomic_json, check_history, gpu_occupancy,
                                        now, process_identity)
from moonseg.utils import load_config


def main():
    os.chdir(ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    plan_path = Path(args.plan).resolve()
    plan = json.loads(plan_path.read_text())
    if plan.get('disabled'):
        raise ValueError(f"Plan disabled: {plan.get('disabled_reason', 'explicit cancellation')}")
    gpus = plan['gpus']
    frozen_da3 = plan.get('experiment') == 'da3mono_frozen_baseline'
    if gpus not in ([0, 7], [4, 5], [2, 3]):
        raise ValueError('Only approved pairs 0/7, 4/5 and the additional DA3 pair 2/3 are permitted')
    if gpus == [2, 3] and plan.get('experiment') not in ('da3mono_backbone', 'da3mono_frozen_baseline'):
        raise ValueError('The additional 2/3 authorization is only for the DA3MONO backbone run')
    config_path = (ROOT / plan['config']).resolve()
    content = config_path.read_bytes()
    config = load_config(config_path)
    settings = config['train']
    if gpus == [2, 3] and config['model'].get('backbone') != 'da3mono_vitl14':
        raise ValueError('The additional 2/3 pair requires the DA3MONO encoder')
    output = (ROOT / settings['output_dir']).resolve()
    if output.parent != (ROOT / 'runs').resolve() or output.name != config_path.stem:
        raise ValueError('Expected matching config name and fresh direct child of runs')
    batch, accumulation = (8, 4) if frozen_da3 else (4, 8)
    if (settings['batch_size'], settings['accumulation_steps'], settings['epochs']) != (batch, accumulation, plan['expected_epochs']):
        raise ValueError(f'Plan requires batch{batch}/accum{accumulation} and matching epoch budget')
    if settings.get('resume') or settings.get('min_effective_batch') != 64:
        raise ValueError('Fresh optimizer and effective batch64 required')
    if frozen_da3:
        if (gpus != [2, 3] or config['model'].get('freeze_backbone') is not True
                or config['model'].get('unfreeze_last_blocks') != 0 or settings.get('initialize_from')):
            raise ValueError('Frozen DA3 baseline requires pair2/3, a fully frozen encoder and fresh Adapter/head')
        baseline = load_config(ROOT / 'configs/dnv3_va_m2f_vitl16_rrc_clip001.yaml')
        for section in ('data', 'loss', 'inference'):
            if config[section] != baseline[section]:
                raise ValueError(f'Frozen DA3 must retain baseline {section}')
        if ({key: value for key, value in settings.items() if key != 'output_dir'}
                != {key: value for key, value in baseline['train'].items() if key != 'output_dir'}):
            raise ValueError('Frozen DA3 must retain every baseline training setting except output_dir')
        da3_values = dict(backbone='da3mono_vitl14', backbone_weights='ckpts/da3mono-large/model.safetensors',
                          backbone_gradient_checkpointing=False, model_family='da3mono-large',
                          source_revision='3d835ec1a5802d64a8b8b15f817a1ab54809bfe4',
                          native_patch_size=14, adapter_feature_stride=16)
        if any(config['model'].get(key) != value for key, value in da3_values.items()):
            raise ValueError('Frozen DA3 requires the approved native pretrained encoder and patch14/stride16 bridge')
        da3_only = set(da3_values)
        if ({key: value for key, value in config['model'].items() if key not in da3_only}
                != {key: value for key, value in baseline['model'].items() if key not in da3_only}):
            raise ValueError('Frozen DA3 must retain baseline Adapter/head and freeze settings')
    elif config['model'].get('freeze_backbone') is not False or config['model'].get('unfreeze_last_blocks') != 0:
        raise ValueError('These plans train the whole ViT')
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'Refusing existing run: {output}')
    receipt = dict(plan=plan, plan_path=str(plan_path), status='starting',
                   pid=os.getpid(), **process_identity(os.getpid()), started_at=now())
    status_path = plan_path.with_name(plan_path.stem + '.status.json')
    with ExitStack() as stack:
        locks = []
        # Both historical training and evaluation schedulers share these locks.
        for gpu in sorted(gpus):
            for filename in (f'runs/.training_gpu_{gpu}.lock',
                             f'runs/evaluations_2026-10-23/gpu_{gpu}.lock'):
                lock = stack.enter_context((ROOT / filename).open('a+'))
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                locks.append(lock)
        snapshot = gpu_occupancy(gpus)
        if snapshot['gpu_uuids'] != plan['expected_gpu_uuids'] or snapshot['occupied']:
            raise ValueError(f'GPU identity/occupancy mismatch: {snapshot}')
        snapshot_path = plan_path.with_name(config_path.name)
        if snapshot_path.exists() and snapshot_path.read_bytes() != content:
            raise ValueError('Conflicting config snapshot')
        snapshot_path.write_bytes(content)
        output.mkdir()
        env = os.environ.copy()
        env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=','.join(map(str, gpus)),
                   PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
                   PYTHON=str((ROOT / '.venv').resolve() / 'bin/python'), OMP_NUM_THREADS='4', NPROC_PER_NODE='2')
        if gpus in ([4, 5], [2, 3]):
            env['NCCL_P2P_DISABLE'] = '1'
        else:
            env.pop('NCCL_P2P_DISABLE', None)
        command = ['bash', 'scripts/train.sh', str(snapshot_path)]
        log_path = output / 'train.log'
        receipt.update(config=str(snapshot_path), run_dir=str(output), gpu_snapshot=snapshot,
                       command=command, log_path=str(log_path),
                       environment={key: env.get(key) for key in (
                           'CUDA_DEVICE_ORDER', 'CUDA_VISIBLE_DEVICES', 'PYTORCH_CUDA_ALLOC_CONF',
                           'NCCL_P2P_DISABLE', 'PYTHON', 'OMP_NUM_THREADS')})
        atomic_json(status_path, receipt)
        with log_path.open('x') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     pass_fds=tuple(lock.fileno() for lock in locks))
            receipt.update(status='running', child_pid=child.pid,
                           child_start_ticks=process_identity(child.pid)['start_ticks'])
            atomic_json(status_path, receipt)
            code = child.wait()
        receipt.update(exit_code=code, finished_at=now(), status='failed')
        if code == 0:
            try:
                check_history(output, plan['expected_epochs'])
                rows = [json.loads(line) for line in (output / 'history.jsonl').read_text().splitlines() if line.strip()]
                if rows[-1]['global_step'] != plan['expected_updates']:
                    raise ValueError('Final optimizer-step count differs from the approved budget')
                receipt['status'] = 'completed'
                atomic_json(output / 'queue_completed.json', dict(
                    status='completed', exit_code=0, name=output.name, run_dir=str(output),
                    config=str(snapshot_path), expected_epochs=plan['expected_epochs'],
                    pid=child.pid, start_ticks=receipt['child_start_ticks'], finished_at=now()))
            except Exception as error:
                receipt.update(error=str(error), status='failed')
                code = 1
        atomic_json(status_path, receipt)
        return code


if __name__ == '__main__':
    raise SystemExit(main())
