"""V2 coordinator and local queues: 24 LR trials, fresh GPU admission, 18 finals."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import uuid

from checkpoints import write_json
from data import canonical_hash
from experiment import EXECUTORS
from run_queue import environment, queue_lock, validate_admission as validate_gpu_admission
from study_configs import PROTOCOL_VERSION
from study_protocol import (make_lr_plan, make_final_plan, read_plan, write_final_files,
                            validate_freeze, source_manifest)
from study_results import read_result, collect_search_results, collect_final_results

GPU_PATTERNS = {'rtx4070ti': r'RTX\s*4070\s*TI(?!\s*SUPER)',
                'rtx3090': r'RTX\s*3090(?!\s*TI)', 'rtx3060': r'RTX\s*3060(?!\s*TI)'}


def validate_admission(path, plan, executor, current=None):
    lr_plan = plan if plan['stage'] == 'lr_search' else plan['lr_plan']
    receipt = validate_gpu_admission(path, lr_plan, executor, current)
    if (receipt.get('schema_version') != 2 or receipt.get('protocol_version') != PROTOCOL_VERSION or
            receipt.get('source_hash') != plan['common']['source_hash'] or
            receipt.get('profile_hash') != canonical_hash(plan['common']['profile'])):
        raise ValueError('V2 requires a new preflight including A-16/B-16; v1 admission is insufficient')
    return receipt


def worker_command(plan_path, freeze, protocol, executor, run, output, admission):
    command = [sys.executable, str(Path(__file__).with_name('study_train.py')),
        '--plan', str(Path(plan_path).resolve()), '--protocol', str(Path(protocol).resolve()),
        '--executor', executor, '--run-id', run['run_id'], '--output', str(Path(output).resolve()),
        '--admission', str(Path(admission).resolve())]
    if run['stage'] == 'final':
        command += ['--freeze', str(Path(freeze).resolve())]
    return command


def execute_queue(plan, plan_path, freeze, protocol, executor, root, admission,
                  retry_failed=False, launch=None):
    if executor not in EXECUTORS:
        raise ValueError('Unknown executor')
    root, stage = Path(root), plan['stage']
    launch = launch or (lambda command, log: subprocess.run(
        command, stdout=log, stderr=subprocess.STDOUT).returncode)
    names = ('COMPLETE',) if stage == 'lr_search' else ('TRAINED', 'TRAINING_FAILED')
    results = []
    with queue_lock(root):
        for run in (r for r in plan['runs'] if r['executor'] == executor):
            attempts = root/'runs'/run['run_id']/'attempts'
            prior = sorted(attempts.iterdir()) if attempts.exists() else []
            if any(p.is_symlink() or not p.is_dir() for p in prior):
                raise ValueError('Unexpected v2 attempt entry')
            resolved = [p for p in prior if any((p/name).exists() for name in names)]
            if len(resolved) > 1:
                raise ValueError('Duplicate resolved v2 attempts: '+run['run_id'])
            if resolved:
                result = read_result(resolved[0], plan, run)
                if stage == 'final' and result['status']['status'] == 'numerical_invalid':
                    raise RuntimeError('Numerically invalid final blocks the series: '+run['run_id'])
                print('SKIP verified', run['run_id'], result['status']['status'], flush=True)
                results.append(result)
                continue
            if stage == 'final':
                for attempt in prior:
                    for name in ('status.json', 'training_status.json'):
                        path = attempt/name
                        if path.exists() and json.loads(path.read_text()).get('status') == 'numerical_invalid':
                            raise RuntimeError('Unsealed numerical final failure needs diagnosis: '+run['run_id'])
            if prior and not retry_failed:
                raise ValueError('Unresolved v2 attempt; diagnose then --retry-failed: '+run['run_id'])
            attempts.mkdir(parents=True, exist_ok=True)
            target = attempts/uuid.uuid4().hex
            logs = root/'logs'
            logs.mkdir(exist_ok=True)
            command = worker_command(plan_path, freeze, protocol, executor, run, target, admission)
            print('START', run['run_id'], run['config'], 'LR', run['lr'], 'seed', run['seed'], flush=True)
            with (logs/(run['run_id']+'-'+target.name+'.log')).open('x') as log:
                code = launch(command, log)
            if not any((target/name).exists() for name in names):
                target.mkdir(exist_ok=True)
                if not (target/'status.json').exists():
                    write_json(target/'status.json', {'schema_version': 2, 'stage': stage,
                        'action': 'train', 'status': 'technical_failed',
                        'reason': f'Worker setup failed, exit={code}; see {log.name}'})
                raise RuntimeError(f'V2 queue stopped: {run["run_id"]}, exit={code}; see {log.name}')
            result = read_result(target, plan, run)
            invalid = result['status']['status'] == 'numerical_invalid'
            if code != (3 if invalid else 0):
                raise RuntimeError('Worker exit/result mismatch')
            if stage == 'final' and invalid:
                raise RuntimeError('Numerically invalid final blocks the series: '+run['run_id'])
            results.append(result)
            print('DONE', run['run_id'], result['status']['status'], flush=True)
    return results


def preflight(plan, plan_path, protocol, executor, root):
    if plan['stage'] != 'lr_search':
        raise ValueError('Run preflight with the v2 LR plan; final reuses its own GPU receipt')
    env = environment()
    if not re.search(GPU_PATTERNS[executor], env['gpu'], re.I):
        raise ValueError('Executor does not match the actual GPU')
    root, source = Path(root), Path(__file__).resolve().parent
    directory = root/'preflight'/uuid.uuid4().hex
    directory.mkdir(parents=True)
    common = plan['common']
    commands = [
        [sys.executable, str(source/'bootstrap.py'), '--require-cuda', '--storage', str(root),
         '--output', str(directory/'environment.json')],
        [sys.executable, str(source/'check_data.py'), '--manifest', str(Path(protocol)/'data_manifest.json'),
         '--model-revision', common['model_revision'], '--output', str(directory/'data.json')],
        [sys.executable, str(source/'study_check.py'), '--parameters', '--device', 'cuda',
         '--output', str(directory/'parameters.json')],
        [sys.executable, str(source/'study_check.py'), '--technical', '--plan', str(Path(plan_path).resolve()),
         '--protocol', str(Path(protocol).resolve()), '--output', str(directory/'trainer')],
    ]
    with queue_lock(root):
        for i, command in enumerate(commands):
            print('V2 PREFLIGHT', i+1, '/', len(commands), flush=True)
            with (directory/f'check-{i}.log').open('x') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        read_plan(plan_path, protocol)  # Refuse receipt if code changed during acceptance.
        receipt = {'schema_version': 2, 'protocol_version': PROTOCOL_VERSION, 'status': 'passed',
                   'plan_hash': plan['plan_hash'], 'source_hash': common['source_hash'],
                   'profile_hash': canonical_hash(common['profile']), 'executor': executor,
                   'environment': env, 'evidence': str(directory.resolve())}
        write_json(directory/'admission.json', receipt)
        try:
            validate_admission(directory/'admission.json', plan, executor, environment())
        except BaseException:
            (directory/'admission.json').unlink()
            raise
    print('Admission:', (directory/'admission.json').resolve(), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    modes = p.add_mutually_exclusive_group(required=True)
    for flag in ('make-lr-plan', 'make-final-plan', 'dry-run', 'preflight', 'run', 'collect'):
        modes.add_argument('--'+flag, action='store_true')
    p.add_argument('--plan', type=Path)
    p.add_argument('--lr-plan', type=Path, default=Path('runs-v2.json'))
    p.add_argument('--protocol', type=Path, default=Path('artifacts/protocol'))
    p.add_argument('--freeze', type=Path)
    p.add_argument('--selection', type=Path)
    p.add_argument('--runtime-profile', type=Path, help='Common profile JSON; only before creating an LR plan')
    p.add_argument('--results', type=Path, nargs='+')
    p.add_argument('--executor', choices=EXECUTORS)
    p.add_argument('--admission', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--retry-failed', action='store_true')
    args = p.parse_args()
    plan_path = args.plan or Path('final-runs-v2.json' if args.make_final_plan else 'runs-v2.json')
    freeze = args.freeze or args.protocol/'freeze-v2.json'
    if args.runtime_profile and not args.make_lr_plan:
        p.error('--runtime-profile is only accepted with --make-lr-plan')
    if args.make_lr_plan:
        profile = json.loads(args.runtime_profile.read_text()) if args.runtime_profile else None
        write_json(plan_path, make_lr_plan(args.protocol, profile))
        print('Created', plan_path.resolve())
        return
    if args.make_final_plan:
        if not args.results:
            p.error('--make-final-plan requires --results with all 24 LR outcomes')
        lr_plan = read_plan(args.lr_plan, args.protocol)
        selection = json.loads(args.selection.read_text()) if args.selection else None
        plan = make_final_plan(lr_plan, args.results, selection)
        write_final_files(plan, plan_path, freeze)
        print('Created', plan_path.resolve(), 'and', freeze.resolve())
        return
    plan = read_plan(plan_path, args.protocol)
    if plan['stage'] == 'final':
        validate_freeze(freeze, plan)
    if args.collect:
        if not args.results or not args.output:
            p.error('--collect needs --results DIR... --output FILE.json')
        collect = collect_search_results if plan['stage'] == 'lr_search' else collect_final_results
        write_json(args.output, collect(plan, args.results))
        return
    if not args.executor:
        p.error('--executor is required')
    if args.dry_run:
        for run in plan['runs']:
            if run['executor'] == args.executor:
                print(json.dumps(run))
        return
    default = Path('results/v2')/('lr' if plan['stage'] == 'lr_search' else 'final')
    root = (args.output or default).resolve()/args.executor
    if args.preflight:
        preflight(plan, plan_path, args.protocol.resolve(), args.executor, root)
        return
    if not args.admission:
        p.error('--run needs the v2 admission from a successful preflight')
    validate_admission(args.admission, plan, args.executor, environment())
    execute_queue(plan, plan_path, freeze, args.protocol, args.executor, root,
                  args.admission, args.retry_failed)


if __name__ == '__main__':
    main()
