"""Create the frozen final design, run local seed blocks, or collect trained artifacts."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import uuid

from checkpoints import write_json
from experiment import EXECUTORS, read_plan
from final_protocol import make_final_plan, read_final_plan, write_plan_files
from final_results import read_training_result, collect_training_results
from run_queue import queue_lock, environment, validate_admission


def worker_command(plan_path, freeze, protocol, executor, run, output, admission):
    return [sys.executable, str(Path(__file__).with_name('final_train.py')),
            '--plan', str(Path(plan_path).resolve()), '--freeze', str(Path(freeze).resolve()),
            '--protocol', str(Path(protocol).resolve()), '--executor', executor,
            '--run-id', run['run_id'], '--output', str(Path(output).resolve()),
            '--admission', str(Path(admission).resolve())]


def execute_final_queue(plan, plan_path, freeze, protocol, executor, root, admission,
                        retry_failed=False, launch=None):
    if executor not in EXECUTORS:
        raise ValueError('Unknown final executor')
    root = Path(root)
    launch = launch or (lambda command, log: subprocess.run(
        command, stdout=log, stderr=subprocess.STDOUT).returncode)
    trained = []
    with queue_lock(root):
        for run in (r for r in plan['runs'] if r['executor'] == executor):
            attempts = root/'runs'/run['run_id']/'attempts'
            prior = sorted(attempts.iterdir()) if attempts.exists() else []
            if any(p.is_symlink() or not p.is_dir() for p in prior):
                raise ValueError('Unexpected final attempt entry')
            resolved = [p for p in prior if (p/'TRAINED').exists() or (p/'TRAINING_FAILED').exists()]
            if len(resolved) > 1:
                raise ValueError('Multiple resolved final attempts: '+run['run_id'])
            if resolved:
                result = read_training_result(resolved[0], plan, run)
                if result['status']['status'] == 'numerical_invalid':
                    raise RuntimeError('Numerically invalid final; diagnose without replacing LR/seed: '+run['run_id'])
                print('SKIP verified TRAINED', run['run_id'], flush=True)
                trained.append(result)
                continue
            for attempt in prior:
                for name in ('training_status.json', 'status.json'):
                    path = attempt/name
                    if path.exists() and json.loads(path.read_text()).get('status') == 'numerical_invalid':
                        raise RuntimeError('Numerically invalid final; unsealed evidence needs diagnosis: '+run['run_id'])
            if prior and not retry_failed:
                raise ValueError('Unresolved final attempt; diagnose then use --retry-failed: '+run['run_id'])
            attempts.mkdir(parents=True, exist_ok=True)
            target = attempts/uuid.uuid4().hex
            logs = root/'logs'
            logs.mkdir(exist_ok=True)
            command = worker_command(plan_path, freeze, protocol, executor, run, target, admission)
            print('START', run['run_id'], run['config'], 'seed', run['seed'], 'LR', run['lr'], flush=True)
            with (logs/(run['run_id']+'-'+target.name+'.log')).open('x') as log:
                code = launch(command, log)
            if not any((target/name).exists() for name in ('TRAINED', 'TRAINING_FAILED')):
                target.mkdir(exist_ok=True)
                if not (target/'status.json').exists():
                    write_json(target/'status.json', {'schema_version': 1, 'stage': 'final',
                        'action': 'train', 'status': 'technical_failed',
                        'reason': f'Worker setup failed, exit={code}; see {log.name}'})
                raise RuntimeError(f'Final queue stopped: {run["run_id"]}, exit={code}; see {log.name}')
            result = read_training_result(target, plan, run)
            invalid = result['status']['status'] == 'numerical_invalid'
            if code != (3 if invalid else 0):
                raise RuntimeError('Final worker exit/result mismatch')
            if invalid:
                raise RuntimeError('Numerically invalid final; series blocked: '+run['run_id'])
            trained.append(result)
            print('TRAINED', run['run_id'], flush=True)
    return trained


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ('make-plan', 'dry-run', 'run', 'collect'):
        modes.add_argument('--'+mode, action='store_true')
    parser.add_argument('--lr-plan', type=Path, default=Path('runs.json'))
    parser.add_argument('--plan', type=Path, default=Path('final_runs.json'))
    parser.add_argument('--freeze', type=Path, help='Default: PROTOCOL/freeze.json')
    parser.add_argument('--protocol', type=Path, default=Path('artifacts/protocol'))
    parser.add_argument('--selection', type=Path, help='Optional selected_lrs.json to cross-check')
    parser.add_argument('--results', type=Path, nargs='+')
    parser.add_argument('--executor', choices=EXECUTORS)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--admission', type=Path)
    parser.add_argument('--retry-failed', action='store_true')
    args = parser.parse_args()
    freeze = args.freeze if args.freeze is not None else args.protocol/'freeze.json'
    if args.make_plan:
        if not args.results:
            parser.error('--make-plan requires all LR result directories via --results')
        lr_plan = read_plan(args.lr_plan, args.protocol)
        selection = json.loads(args.selection.read_text()) if args.selection else None
        plan = make_final_plan(lr_plan, args.results, selection)
        write_plan_files(plan, args.plan, freeze)
        print('Created', args.plan.resolve(), 'and', freeze.resolve(), flush=True)
        return
    plan = read_final_plan(args.plan, args.protocol, freeze)
    if args.collect:
        if not args.results or not args.output:
            parser.error('--collect requires --results DIR... --output trained_finals.json')
        write_json(args.output, collect_training_results(plan, args.results))
        return
    if not args.executor:
        parser.error('--executor is required')
    if args.dry_run:
        for run in plan['runs']:
            if run['executor'] == args.executor:
                print(json.dumps(run))
        return
    if not args.output or not args.admission:
        parser.error('--run requires --output DIR and the original LR --admission')
    validate_admission(args.admission, plan['lr_plan'], args.executor, environment())
    execute_final_queue(plan, args.plan, freeze, args.protocol, args.executor,
                        args.output.resolve()/args.executor, args.admission, args.retry_failed)


if __name__ == '__main__':
    main()
