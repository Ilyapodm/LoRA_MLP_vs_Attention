"""Immutable v2 LR/final plans, scientific recipe and six-method freeze. No ML imports."""
from copy import deepcopy
import json
from pathlib import Path
import re

from file_utils import file_hash, write_json
from data import canonical_hash, TRANSFORM_VERSION, SELECTION_VERSION, DATASET_ID
from study_runtime import EXECUTORS, PROFILE, protocol_info, validate_profile
from study_configs import (PROTOCOL_VERSION, MODEL_ID, CONFIGS, EPOCHS, LR_GRID, SEARCH_SEED,
                           FINAL_SEEDS, ALIASES, SERIES, RECIPE,
                           configuration_specs, statistics_spec)

SOURCE_FILES = (
    'file_utils.py', 'data.py', 'study_configs.py', 'study_runtime.py',
    'study_trainer.py', 'study_protocol.py', 'study_results.py', 'study_train.py',
    'study_queue.py', 'study_check.py', 'bootstrap.py', 'check_data.py', 'requirements.txt')


def source_manifest():
    root = Path(__file__).resolve().parent
    return {name: file_hash(root/name) for name in SOURCE_FILES}


def seal_hash(payload, key):
    result = deepcopy(payload)
    result[key] = canonical_hash(result)
    return result


def common_recipe(pins, profile, source_hash):
    profile = validate_profile(dict(profile))
    return {
        'protocol_version': PROTOCOL_VERSION, **pins,
        'epochs': EPOCHS, 'train_images': 18180, 'val_images': 2020, 'test_images': 25250,
        'python': '3.12', 'search_seed': SEARCH_SEED, 'final_seeds': list(FINAL_SEEDS),
        'lr_grid': list(LR_GRID), 'configurations': configuration_specs(),
        'aliases': deepcopy(ALIASES), 'series': deepcopy(SERIES),
        'recipe': deepcopy(RECIPE), 'statistics': statistics_spec(),
        'transform_version': TRANSFORM_VERSION, 'selection_version': SELECTION_VERSION,
        'profile': profile, 'source_hash': source_hash,
    }


def steps_per_epoch(common):
    batch = common['profile']['microbatch']*common['profile']['accumulation']
    return (common['train_images']+batch-1)//batch


def run_id(common, stage, name, lr, seed, selection_hash=None):
    return 'v2-'+('lr-' if stage == 'lr_search' else 'final-')+canonical_hash({
        'common': common, 'stage': stage, 'config': name, 'lr': float(lr),
        'seed': seed, 'selection_hash': selection_hash})[:20]


def expected_lr_runs(common):
    runs = []
    for i, name in enumerate(CONFIGS):
        for j, lr in enumerate(LR_GRID):
            runs.append({'run_id': run_id(common, 'lr_search', name, lr, SEARCH_SEED),
                         'stage': 'lr_search', 'config': name, 'lr': lr,
                         'seed': SEARCH_SEED, 'executor': EXECUTORS[(j-i) % len(EXECUTORS)]})
    return runs


def expected_final_runs(common, selection, selection_hash):
    names, runs = list(CONFIGS), []
    for offset, (executor, seed) in enumerate(zip(EXECUTORS, FINAL_SEEDS)):
        for name in names[offset:]+names[:offset]:
            lr = selection['selected'][name]['lr']
            runs.append({'run_id': run_id(common, 'final', name, lr, seed, selection_hash),
                         'stage': 'final', 'config': name, 'lr': lr,
                         'seed': seed, 'executor': executor})
    return runs


def make_lr_plan(protocol, profile=None):
    files = source_manifest()
    common = common_recipe(protocol_info(protocol), PROFILE if profile is None else profile,
                           canonical_hash(files))
    return validate_plan(seal_hash({'schema_version': 2, 'stage': 'lr_search',
        'source_files': files, 'common': common, 'runs': expected_lr_runs(common)}, 'plan_hash'))


def portable_selection(selection):
    return deepcopy({**{k: selection[k] for k in ('schema_version', 'stage', 'plan_hash', 'selected')},
            'trials': sorted(({k: t[k] for k in ('run', 'status', 'metrics', 'checksums_sha256')}
                              for t in selection['trials']), key=lambda t: t['run']['run_id'])})


def validate_selection(selection, lr_plan):
    if (selection['schema_version'], selection['stage'], selection['plan_hash']) != (
            2, 'lr_selection', lr_plan['plan_hash']):
        raise ValueError('Expected v2 selection from the same 24-trial LR plan')
    expected = {r['run_id']: r for r in lr_plan['runs']}
    found = {}
    for trial in selection['trials']:
        run, status, metrics = trial['run'], trial['status'], trial['metrics']
        identifier = run['run_id']
        if identifier in found or run != expected.get(identifier):
            raise ValueError('Duplicate/foreign LR evidence')
        if not re.fullmatch('[0-9a-f]{64}', trial['checksums_sha256']):
            raise ValueError('Missing original LR checksums')
        if ((status.get('schema_version'), status.get('stage'), status.get('action')) !=
                (2, 'lr_search', 'train') or status.get('status') not in ('valid', 'numerical_invalid')):
            raise ValueError('Unresolved LR candidate')
        if any(metrics.get(k) is not None for k in ('test_accuracy', 'test_correct_count', 'test_sample_count')):
            raise ValueError('LR evidence contains test metrics')
        if status['status'] == 'valid':
            count = metrics['val_correct_count']
            steps = EPOCHS*steps_per_epoch(lr_plan['common'])
            if (type(count) is not int or not 0 <= count <= 2020 or
                    metrics['epochs_completed'] != EPOCHS or metrics['val_sample_count'] != 2020 or
                    metrics['val_accuracy'] != count/2020 or type(metrics['best_epoch']) is not int or
                    not 1 <= metrics['best_epoch'] <= EPOCHS or
                    metrics.get('global_step') != steps or metrics.get('attempted_steps') != steps or
                    any(type(metrics.get(k)) is not int or metrics[k] < 0
                        for k in ('applied_steps', 'skipped_steps')) or
                    metrics['applied_steps']+metrics['skipped_steps'] != steps):
                raise ValueError('Invalid v2 LR metrics')
        elif not status.get('reason') or any(metrics.get(k) is not None for k in (
                'val_accuracy', 'val_correct_count', 'val_sample_count', 'best_epoch')):
            raise ValueError('Invalid numerical-failure evidence')
        found[identifier] = trial
    if set(found) != set(expected):
        raise ValueError('Need all 24 resolved v2 LR trials')
    winners = {}
    for name in CONFIGS:
        valid = [t for t in found.values() if t['run']['config'] == name and t['status']['status'] == 'valid']
        if not valid:
            raise ValueError('All four LR candidates numerically invalid for '+name)
        best = min(valid, key=lambda t: (-t['metrics']['val_correct_count'], t['run']['lr']))
        winners[name] = {'lr': best['run']['lr'], 'run_id': best['run']['run_id'],
                         'val_correct_count': best['metrics']['val_correct_count'],
                         'val_sample_count': 2020, 'best_epoch': best['metrics']['best_epoch']}
    if winners != selection['selected']:
        raise ValueError('Selected LR differs from maximum correct_count/lower-LR tie-break')
    return selection


def validate_plan(plan):
    if (plan.get('schema_version') != 2 or plan.get('stage') not in ('lr_search', 'final') or
            canonical_hash({k: v for k, v in plan.items() if k != 'plan_hash'}) != plan.get('plan_hash')):
        raise ValueError('V2 plan schema/hash mismatch; pilot v1 cannot be mixed with v2')
    files, common = plan['source_files'], plan['common']
    if set(files) != set(SOURCE_FILES) or any(not re.fullmatch('[0-9a-f]{64}', v) for v in files.values()):
        raise ValueError('Invalid v2 source manifest')
    pins = {k: common[k] for k in ('model', 'model_revision', 'dataset', 'dataset_revision', 'manifest_hash')}
    if pins['model'] != MODEL_ID or pins['dataset'] != DATASET_ID:
        raise ValueError('Unexpected model/dataset')
    for key, size in (('model_revision', 40), ('dataset_revision', 40), ('manifest_hash', 64)):
        if not re.fullmatch('[0-9a-f]{'+str(size)+'}', pins[key]):
            raise ValueError('Invalid immutable pin: '+key)
    if common != common_recipe(pins, common['profile'], canonical_hash(files)):
        raise ValueError('V2 scientific recipe differs from the agreed protocol')
    if plan['stage'] == 'lr_search':
        expected = expected_lr_runs(common)
    else:
        lr_plan = validate_plan(plan['lr_plan'])
        if lr_plan['stage'] != 'lr_search' or lr_plan['common'] != common or lr_plan['source_files'] != files:
            raise ValueError('Final must keep the v2 LR recipe and computational release')
        selection = validate_selection(plan['selection'], lr_plan)
        if selection != portable_selection(selection) or canonical_hash(selection) != plan['selection_hash']:
            raise ValueError('Final selection hash/order mismatch')
        expected = expected_final_runs(common, selection, plan['selection_hash'])
    if plan['runs'] != expected:
        raise ValueError('V2 tasks/assignments/order differ from the declared design')
    return plan


def read_plan(path, protocol=None, check_source=True):
    plan = validate_plan(json.loads(Path(path).read_text()))
    if check_source and plan['source_files'] != source_manifest():
        raise ValueError('Code changed since v2 plan creation; use the frozen release')
    if protocol is not None and any(plan['common'][k] != v for k, v in protocol_info(protocol).items()):
        raise ValueError('V2 plan/protocol pins differ')
    return plan


def make_final_plan(lr_plan, roots, selection=None):
    from study_results import collect_search_results
    validate_plan(lr_plan)
    if lr_plan['stage'] != 'lr_search' or lr_plan['source_files'] != source_manifest():
        raise ValueError('Final requires the original v2 LR release')
    verified = portable_selection(collect_search_results(lr_plan, roots))
    validate_selection(verified, lr_plan)
    if selection is not None and portable_selection(selection) != verified:
        raise ValueError('Selection differs from the verified v2 LR artifacts')
    selection_hash = canonical_hash(verified)
    return validate_plan(seal_hash({
        'schema_version': 2, 'stage': 'final', 'source_files': deepcopy(lr_plan['source_files']),
        'common': deepcopy(lr_plan['common']), 'lr_plan': deepcopy(lr_plan),
        'selection': verified, 'selection_hash': selection_hash,
        'runs': expected_final_runs(lr_plan['common'], verified, selection_hash)}, 'plan_hash'))


def make_freeze(plan):
    validate_plan(plan)
    if plan['stage'] != 'final':
        raise ValueError('Only final plans can be frozen')
    return seal_hash({'schema_version': 2, 'protocol_version': PROTOCOL_VERSION,
        'stage': 'final', 'plan_hash': plan['plan_hash'], 'lr_plan_hash': plan['lr_plan']['plan_hash'],
        'selection_hash': plan['selection_hash'], 'source_hash': plan['common']['source_hash'],
        'selected_lrs': {c: plan['selection']['selected'][c]['lr'] for c in CONFIGS},
        'series': deepcopy(SERIES), 'test_models': len(CONFIGS)*len(FINAL_SEEDS)}, 'freeze_hash')


def validate_freeze(path, plan):
    if json.loads(Path(path).read_text()) != make_freeze(plan):
        raise ValueError('Missing/mismatched v2 final freeze')


def write_final_files(plan, path, freeze):
    path, freeze = Path(path), Path(freeze)
    if path.resolve() == freeze.resolve() or path.exists() or freeze.exists():
        raise FileExistsError('Use new v2 plan/freeze paths')
    record = make_freeze(plan)
    write_json(path, plan)
    try:
        write_json(freeze, record)
    except BaseException:
        path.unlink()
        raise


def config_record(plan, run):
    if run not in plan['runs']:
        raise ValueError('Task is not in the v2 plan')
    return {'schema_version': 2, 'common': plan['common'], 'run': run,
            'plan_hash': plan['plan_hash'],
            'freeze_hash': make_freeze(plan)['freeze_hash'] if plan['stage'] == 'final' else None}
