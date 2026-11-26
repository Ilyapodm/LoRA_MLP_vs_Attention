"""Frozen final design derived from all twelve verified LR outcomes. No ML imports."""
from copy import deepcopy
import json
from pathlib import Path
import re

from checkpoints import file_hash, write_json
from configs import CONFIGS, FINAL_SEEDS
from data import canonical_hash
from experiment import (SOURCE_FILES, EXECUTORS, collect_results, protocol_info,
                        source_hash, validate_plan as validate_lr_plan)

FINAL_SOURCE_FILES = SOURCE_FILES + (
    'final_protocol.py', 'final_results.py', 'final_train.py', 'final_queue.py')
ORDERS = (('LP', 'A-iso', 'M-iso', 'B-iso'),
          ('A-iso', 'M-iso', 'B-iso', 'LP'),
          ('M-iso', 'B-iso', 'LP', 'A-iso'))


def source_manifest():
    root = Path(__file__).resolve().parent
    return {name: file_hash(root/name) for name in FINAL_SOURCE_FILES}


def portable_selection(selection):
    """Absolute collection paths are delivery details, not scientific identity."""
    return {**{k: selection[k] for k in ('schema_version', 'stage', 'plan_hash', 'selected')},
            'trials': sorted(({k: trial[k] for k in ('run', 'status', 'metrics')}
                              for trial in selection['trials']),
                             key=lambda t: t['run']['run_id'])}


def validate_selection(selection, lr_plan):
    if (selection['schema_version'], selection['stage'], selection['plan_hash']) != (
            1, 'lr_selection', lr_plan['plan_hash']):
        raise ValueError('Selection belongs to another LR plan')
    expected = {r['run_id']: r for r in lr_plan['runs']}
    found = {}
    for trial in selection['trials']:
        run, status, metrics = trial['run'], trial['status'], trial['metrics']
        identifier = run['run_id']
        if identifier in found or run != expected.get(identifier):
            raise ValueError('Duplicate/foreign LR trial in selection')
        if status['stage'] != 'lr_search' or status['status'] not in ('valid', 'numerical_invalid'):
            raise ValueError('Unresolved LR selection trial')
        if any(metrics.get(k) is not None for k in ('test_accuracy', 'test_correct_count', 'test_sample_count')):
            raise ValueError('LR selection contains test metrics')
        if status['status'] == 'valid':
            count = metrics['val_correct_count']
            if (type(count) is not int or not 0 <= count <= 2020 or
                    metrics['epochs_completed'] != 5 or metrics['val_sample_count'] != 2020 or
                    metrics['val_accuracy'] != count/2020 or
                    type(metrics['best_epoch']) is not int or not 1 <= metrics['best_epoch'] <= 5):
                raise ValueError('Invalid LR validation evidence')
        found[identifier] = trial
    if set(found) != set(expected):
        raise ValueError('Final requires all 12 resolved LR trials')
    winners = {}
    for name in CONFIGS:
        valid = [t for t in found.values() if t['run']['config'] == name and t['status']['status'] == 'valid']
        if not valid:
            raise ValueError('All LR candidates numerically invalid for '+name)
        best = min(valid, key=lambda t: (-t['metrics']['val_correct_count'], t['run']['lr']))
        winners[name] = {'lr': best['run']['lr'], 'run_id': best['run']['run_id'],
                         'val_correct_count': best['metrics']['val_correct_count'],
                         'val_sample_count': 2020, 'best_epoch': best['metrics']['best_epoch']}
    if selection['selected'] != winners:
        raise ValueError('Selected LR differs from verified maximum/tie-break')
    return selection


def final_run_id(common, selection_hash, config, lr, seed):
    return 'final-'+canonical_hash({'common': common, 'selection_hash': selection_hash,
        'stage': 'final', 'config': config, 'lr': float(lr), 'seed': seed})[:20]


def expected_runs(common, selection, selection_hash):
    runs = []
    for executor, seed, order in zip(EXECUTORS, FINAL_SEEDS, ORDERS):
        for config in order:
            lr = selection['selected'][config]['lr']
            runs.append({'run_id': final_run_id(common, selection_hash, config, lr, seed),
                         'stage': 'final', 'config': config, 'lr': lr,
                         'seed': seed, 'executor': executor})
    return runs


def make_final_plan(lr_plan, roots, selection=None):
    validate_lr_plan(lr_plan)
    if lr_plan['common']['source_hash'] != source_hash():
        raise ValueError('LR computational release changed; preserve the accepted LR sources')
    collected = collect_results(lr_plan, roots)
    verified = portable_selection(collected)
    validate_selection(verified, lr_plan)
    if selection is not None and portable_selection(selection) != verified:
        raise ValueError('Supplied selection differs from the verified LR artifacts')
    files = source_manifest()
    common = {k: deepcopy(v) for k, v in lr_plan['common'].items() if k not in ('seed', 'source_hash')}
    common.update(final_seeds=list(FINAL_SEEDS), source_hash=canonical_hash(files))
    selection_hash = canonical_hash(verified)
    evidence = {t['run']['run_id']: file_hash(Path(t['path'])/'checksums.json')
                for t in collected['trials']}
    plan = {'schema_version': 1, 'stage': 'final', 'lr_plan': deepcopy(lr_plan),
            'selection': verified, 'selection_hash': selection_hash, 'lr_evidence': evidence,
            'source_files': files, 'common': common,
            'runs': expected_runs(common, verified, selection_hash)}
    plan['plan_hash'] = canonical_hash(plan)
    return validate_final_plan(plan)


def validate_final_plan(plan):
    if (plan.get('schema_version') != 1 or plan.get('stage') != 'final' or
            canonical_hash({k: v for k, v in plan.items() if k != 'plan_hash'}) != plan.get('plan_hash')):
        raise ValueError('Final plan schema/hash mismatch')
    lr_plan = validate_lr_plan(plan['lr_plan'])
    selection = validate_selection(plan['selection'], lr_plan)
    if selection != portable_selection(selection) or canonical_hash(selection) != plan['selection_hash']:
        raise ValueError('Final selection hash/order mismatch')
    files = plan['source_files']
    if set(files) != set(FINAL_SOURCE_FILES) or any(not re.fullmatch('[0-9a-f]{64}', v) for v in files.values()):
        raise ValueError('Invalid final computational release')
    if canonical_hash({name: files[name] for name in SOURCE_FILES}) != lr_plan['common']['source_hash']:
        raise ValueError('Final must preserve the accepted LR training implementation')
    common = {k: v for k, v in lr_plan['common'].items() if k not in ('seed', 'source_hash')}
    common.update(final_seeds=list(FINAL_SEEDS), source_hash=canonical_hash(files))
    if plan['common'] != common or plan['runs'] != expected_runs(common, selection, plan['selection_hash']):
        raise ValueError('Final settings/seed blocks/order differ from the frozen design')
    evidence = plan['lr_evidence']
    if (set(evidence) != {r['run_id'] for r in lr_plan['runs']} or
            any(not re.fullmatch('[0-9a-f]{64}', v) for v in evidence.values())):
        raise ValueError('Missing LR checksum provenance')
    return plan


def make_freeze(plan):
    validate_final_plan(plan)
    freeze = {'schema_version': 1, 'stage': 'final', 'plan_hash': plan['plan_hash'],
              'lr_plan_hash': plan['lr_plan']['plan_hash'], 'selection_hash': plan['selection_hash'],
              'source_hash': plan['common']['source_hash'],
              'selected_lrs': {c: plan['selection']['selected'][c]['lr'] for c in CONFIGS}}
    freeze['freeze_hash'] = canonical_hash(freeze)
    return freeze


def read_final_plan(path, protocol, freeze):
    plan = validate_final_plan(json.loads(Path(path).read_text()))
    if plan['source_files'] != source_manifest():
        raise ValueError('Final code differs from the frozen release')
    if any(plan['common'][k] != v for k, v in protocol_info(protocol).items()):
        raise ValueError('Final protocol/data revisions differ')
    if json.loads(Path(freeze).read_text()) != make_freeze(plan):
        raise ValueError('Missing/mismatched final freeze')
    return plan


def write_plan_files(plan, plan_path, freeze_path):
    """Exclusive creation: a final design is never silently replaced."""
    plan_path, freeze_path = Path(plan_path), Path(freeze_path)
    if plan_path.resolve() == freeze_path.resolve() or plan_path.exists() or freeze_path.exists():
        raise FileExistsError('Use new final plan and freeze paths')
    freeze = make_freeze(plan)
    write_json(plan_path, plan)
    try:
        write_json(freeze_path, freeze)
    except BaseException:
        plan_path.unlink()  # This invocation exclusively created this file.
        raise


def config_record(plan, run):
    if run not in plan['runs']:
        raise ValueError('Run is not in the final plan')
    return {'schema_version': 1, 'common': plan['common'], 'run': run,
            'plan_hash': plan['plan_hash'], 'freeze_hash': make_freeze(plan)['freeze_hash']}
