"""One local process per GPU, whole-run retries, no network scheduler or test evaluation."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import uuid

from checkpoints import write_json
from experiment import (EXECUTORS, PROFILE, make_plan, read_plan, read_result,
                        collect_results, protocol_info, canonical_hash)


@contextmanager
def queue_lock(root):
    root = Path(root);root.mkdir(parents=True, exist_ok=True)
    lock = root/'queue.lock'
    with lock.open('x') as f:
        json.dump({'pid': os.getpid(), 'host': platform.node()}, f)
    try:
        yield
    finally:
        lock.unlink()


def environment():
    import importlib.metadata
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('Expose exactly one CUDA GPU (CUDA_VISIBLE_DEVICES)')
    packages = {n: importlib.metadata.version(n) for n in ('torch', 'torchvision', 'transformers',
                'peft', 'accelerate', 'datasets', 'numpy', 'Pillow', 'huggingface-hub', 'safetensors')}
    requirements = Path(__file__).with_name('requirements.txt').read_text()
    for line in requirements.splitlines():
        if '==' in line and not line.lstrip().startswith('#'):
            name, version = line.strip().split('==')
            actual = importlib.metadata.version(name)
            if actual.split('+')[0] != version:
                raise ValueError(f'Package mismatch: {name}: {actual} != {version}')
    if platform.python_version_tuple()[:2] != ('3', '12') or torch.version.cuda != '12.6':
        raise ValueError('Scientific PC profile requires Python 3.12 and torch CUDA build 12.6')
    device = torch.cuda.get_device_properties(0)
    return {'python': platform.python_version(), 'packages': packages, 'cuda_build': torch.version.cuda,
            'gpu': device.name, 'vram_gib': device.total_memory/2**30, 'platform': platform.platform()}


def worker_command(plan_path, protocol, executor, run, output, admission):
    return [sys.executable, str(Path(__file__).with_name('train.py')), '--stage', 'lr_search',
            '--plan', str(Path(plan_path).resolve()), '--protocol', str(Path(protocol).resolve()),
            '--executor', executor, '--run-id', run['run_id'], '--output', str(Path(output).resolve()),
            '--admission', str(Path(admission).resolve())]


def validate_admission(path, plan, executor, current=None):
    receipt = json.loads(Path(path).read_text())
    if receipt.get('status') != 'passed' or receipt.get('plan_hash') != plan['plan_hash'] or receipt.get('executor') != executor:
        raise ValueError('Missing/mismatched preflight; run --preflight first')
    if current is not None:
        patterns = {'rtx4070ti': r'RTX\s*4070\s*TI(?!\s*SUPER)', 'rtx3090': r'RTX\s*3090(?!\s*TI)', 'rtx3060': r'RTX\s*3060(?!\s*TI)'}
        if not re.search(patterns[executor], current['gpu'], re.I):
            raise ValueError('Executor name does not match the actual GPU')
    if current is not None and receipt['environment'] != current:
        raise ValueError('Environment/device changed since preflight')
    return receipt


def execute_queue(plan, plan_path, protocol, executor, root, admission, retry_failed=False, launch=None):
    root = Path(root)
    launch = launch or (lambda command, log: subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode)
    resolved = []
    with queue_lock(root):
        for run in (r for r in plan['runs'] if r['executor'] == executor):
            attempts = root/'runs'/run['run_id']/'attempts'
            prior = sorted(attempts.iterdir()) if attempts.exists() else []
            completed = [p for p in prior if (p/'COMPLETE').exists()]
            if len(completed) > 1:
                raise ValueError('Multiple resolved attempts for '+run['run_id'])
            if completed:
                result = read_result(completed[0], plan, run)
                print('SKIP verified', run['run_id'], result['status']['status'], flush=True)
                resolved.append(result);continue
            if prior and not retry_failed:
                raise ValueError('Unresolved previous attempt; diagnose then use --retry-failed: '+run['run_id'])
            target = attempts/uuid.uuid4().hex
            attempts.mkdir(parents=True, exist_ok=True)
            logs = root/'logs';logs.mkdir(exist_ok=True)
            command = worker_command(plan_path, protocol, executor, run, target, admission)
            print('START', run['run_id'], run['config'], run['lr'], flush=True)
            with (logs/(run['run_id']+'-'+target.name+'.log')).open('x') as log:
                code = launch(command, log)
            if not (target/'COMPLETE').exists():
                target.mkdir(exist_ok=True)
                if not (target/'status.json').exists():
                    write_json(target/'status.json', {'stage': 'lr_search', 'status': 'technical_failed',
                                                      'reason': f'Worker setup failed, exit={code}; see {log.name}'})
                raise RuntimeError(f'Queue stopped: {run["run_id"]}, exit={code}; see {log.name}')
            result = read_result(target, plan, run)
            if code not in (0, 3) or (code == 3) != (result['status']['status'] == 'numerical_invalid'):
                raise RuntimeError('Worker exit/result mismatch')
            resolved.append(result)
            print('DONE', run['run_id'], result['status']['status'], flush=True)
    return resolved


def preflight(plan, plan_path, protocol, executor, root):
    env = environment()
    root = Path(root);root.mkdir(parents=True, exist_ok=True)
    directory = root/'preflight'/uuid.uuid4().hex;directory.mkdir(parents=True)
    source = Path(__file__).resolve().parent
    common = plan['common']
    profile = directory/'runtime-profile.json';write_json(profile, common['profile'])
    commands = [
        [sys.executable, str(source/'bootstrap.py'), '--require-cuda', '--storage', str(root), '--output', str(directory/'environment.json')],
        [sys.executable, str(source/'check_data.py'), '--manifest', str(Path(protocol)/'data_manifest.json'),
         '--model-revision', common['model_revision'], '--output', str(directory/'data.json')],
        [sys.executable, str(source/'test_params.py'), '--device', 'cuda', '--image-size', '224', '--output', str(directory/'parameters.json')],
        [sys.executable, str(source/'train.py'), '--stage', 'technical', '--config', 'B-iso',
         '--manifest', str(Path(protocol)/'data_manifest.json'), '--model-revision', common['model_revision'],
         '--runtime-profile', str(profile), '--output', str(directory/'trainer')],
    ]
    with queue_lock(root):
        for i, command in enumerate(commands):
            print('PREFLIGHT', i+1, '/', len(commands), flush=True)
            with (directory/f'check-{i}.log').open('x') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        receipt = {'schema_version': 1, 'status': 'passed', 'plan_hash': plan['plan_hash'],
                   'executor': executor, 'environment': env, 'evidence': str(directory.resolve())}
        # Use the same validation before issuing and before consuming a receipt.
        write_json(directory/'admission.json', receipt)
        try:
            validate_admission(directory/'admission.json', plan, executor, env)
        except Exception:
            (directory/'admission.json').unlink()
            raise
    print('Admission:', (directory/'admission.json').resolve(), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    modes = p.add_mutually_exclusive_group(required=True)
    for flag in ('make-plan', 'dry-run', 'preflight', 'run', 'collect'):
        modes.add_argument('--'+flag, action='store_true')
    p.add_argument('--plan', type=Path, default=Path('runs.json'))
    p.add_argument('--protocol', type=Path, default=Path('artifacts/protocol'))
    p.add_argument('--executor', choices=EXECUTORS)
    p.add_argument('--output', type=Path)
    p.add_argument('--admission', type=Path)
    p.add_argument('--retry-failed', action='store_true')
    p.add_argument('--results', type=Path, nargs='+')
    args = p.parse_args()
    if args.make_plan:
        write_json(args.plan, make_plan(args.protocol));print('Created', args.plan);return
    plan = read_plan(args.plan, args.protocol)
    if args.collect:
        if not args.results or not args.output:p.error('--collect needs --results DIR... --output selection.json')
        write_json(args.output, collect_results(plan, args.results));return
    if not args.executor:p.error('--executor is required')
    if args.dry_run:
        for run in plan['runs']:
            if run['executor'] == args.executor:print(json.dumps(run))
        return
    if not args.output:p.error('--output is required')
    root = args.output.resolve()/args.executor
    if args.preflight:
        preflight(plan, args.plan, args.protocol.resolve(), args.executor, root);return
    if not args.admission:p.error('--run needs --admission from a successful preflight')
    validate_admission(args.admission, plan, args.executor, environment())
    execute_queue(plan, args.plan, args.protocol, args.executor, root, args.admission, args.retry_failed)


if __name__ == '__main__':
    main()
