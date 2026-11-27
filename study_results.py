"""Stage-aware v2 integrity, 24-trial LR selection and 18-model training collection."""
import json
import math
from pathlib import Path

from file_utils import file_hash, write_json
from data import canonical_hash
from study_configs import CONFIGS, EPOCHS, SERIES
from study_protocol import config_record, steps_per_epoch, portable_selection, validate_selection

TEST_FIELDS = ('test_accuracy', 'test_correct_count', 'test_sample_count')


def artifact_names(stage):
    return ('config.json', 'environment.json', 'history.json', 'updates.json', 'metrics.json',
            'status.json' if stage == 'lr_search' else 'training_status.json')


def inventory(root, stage):
    root = Path(root)
    if root.is_symlink() or (root/'best').is_symlink():
        raise ValueError('Result/export must not be symlinks')
    paths = [root/name for name in artifact_names(stage)]
    for p in sorted((root/'best').rglob('*')):
        if p.is_symlink():
            raise ValueError('Export contains a symlink')
        if p.is_file():
            paths.append(p)
    if any(p.is_symlink() or not p.is_file() for p in paths):
        raise ValueError('Missing/unsafe v2 result artifact')
    return {p.relative_to(root).as_posix(): {'sha256': file_hash(p), 'bytes': p.stat().st_size}
            for p in paths}


def validate_result(root, plan, run):
    root, stage = Path(root), plan['stage']
    if json.loads((root/'config.json').read_text()) != config_record(plan, run):
        raise ValueError('Result belongs to another v2 plan/task/freeze')
    status_file = 'status.json' if stage == 'lr_search' else 'training_status.json'
    status = json.loads((root/status_file).read_text())
    success = 'valid' if stage == 'lr_search' else 'trained'
    if (status.get('schema_version'), status.get('stage'), status.get('action')) != (2, stage, 'train'):
        raise ValueError('V2 result stage/schema mismatch')
    if status.get('status') not in (success, 'numerical_invalid'):
        raise ValueError('V2 result is not resolved')
    metrics = json.loads((root/'metrics.json').read_text())
    history = json.loads((root/'history.json').read_text())
    updates = json.loads((root/'updates.json').read_text())
    canonical_hash([metrics, history, updates])  # NaN/Infinity never become accepted results.
    if any(k not in metrics or metrics[k] is not None for k in TEST_FIELDS):
        raise ValueError('Training/search cannot contain test metrics')
    if status['status'] == 'numerical_invalid':
        if not status.get('reason') or any(metrics.get(k) is not None for k in (
                'best_epoch', 'val_correct_count', 'val_sample_count', 'val_accuracy')):
            raise ValueError('Numerical failure cannot publish a usable best metric')
        return {'path': str(root), 'run': run, 'status': status, 'metrics': metrics}
    per_epoch = steps_per_epoch(plan['common'])
    total_steps = EPOCHS*per_epoch
    evaluations = [h for h in history if 'eval_correct_count' in h]
    if ([h['epoch'] for h in evaluations] != list(range(1, EPOCHS+1)) or
            [h['step'] for h in evaluations] != [per_epoch*i for i in range(1, EPOCHS+1)]):
        raise ValueError('Expected ten complete validation epochs and 360 attempted updates')
    for h in evaluations:
        count = h['eval_correct_count']
        if (type(count) is not int or not 0 <= count <= 2020 or h['eval_sample_count'] != 2020 or
                h['eval_accuracy'] != count/2020 or not math.isfinite(h['eval_loss'])):
            raise ValueError('Invalid full validation metrics')
    best = max(evaluations, key=lambda h: (h['eval_correct_count'], -h['epoch']))
    expected = {'epochs_completed': EPOCHS, 'global_step': total_steps,
                'attempted_steps': total_steps, 'best_epoch': int(best['epoch']),
                'val_sample_count': 2020, 'val_correct_count': best['eval_correct_count'],
                'val_accuracy': best['eval_accuracy']}
    if any(metrics.get(k) != v for k, v in expected.items()) or len(updates) != total_steps:
        raise ValueError('Incomplete training or wrong best checkpoint selection')
    applied = consecutive = 0
    for step, event in enumerate(updates, 1):
        if event['step'] != step or type(event['skipped']) is not bool:
            raise ValueError('Invalid optimizer audit sequence')
        applied += not event['skipped']
        consecutive = consecutive+1 if event['skipped'] else 0
        expected_lr = run['lr']*(1-applied/total_steps)
        if (consecutive >= 10 or not event['lr_after_update'] or
                any(not math.isclose(v, expected_lr, rel_tol=1e-8, abs_tol=1e-12)
                    for v in event['lr_after_update'])):
            raise ValueError('Incorrect applied/skipped scheduler audit')
        if event.get('scale') is not None and (not math.isfinite(event['scale']) or event['scale'] <= 0):
            raise ValueError('Invalid AMP scale')
    if metrics.get('applied_steps') != applied or metrics.get('skipped_steps') != total_steps-applied:
        raise ValueError('Optimizer totals differ from audit')
    metadata = json.loads((root/'best/metadata.json').read_text())
    if metadata != {**config_record(plan, run), 'best_epoch': expected['best_epoch'],
                    'val_correct_count': expected['val_correct_count'], 'val_sample_count': 2020}:
        raise ValueError('Best export metadata differs from training')
    required = ('head.safetensors',) if run['config'] == 'LP' else (
        'adapter_model.safetensors', 'adapter_config.json')
    if any(not (root/'best'/name).is_file() for name in required):
        raise ValueError('Missing best model weights')
    return {'path': str(root), 'run': run, 'status': status, 'metrics': metrics}


def result_paths(stage, status):
    if stage == 'lr_search':
        return 'checksums.json', 'COMPLETE'
    return 'training_checksums.json', 'TRAINED' if status == 'trained' else 'TRAINING_FAILED'


def seal_result(root, plan, run):
    root = Path(root)
    result = validate_result(root, plan, run)
    sums, marker = result_paths(plan['stage'], result['status']['status'])
    if any((root/n).exists() for n in ('checksums.json', 'training_checksums.json',
                                      'COMPLETE', 'TRAINED', 'TRAINING_FAILED')):
        raise FileExistsError('Resolved v2 result cannot be replaced')
    write_json(root/sums, {'schema_version': 2, 'stage': plan['stage'],
                           'files': inventory(root, plan['stage'])})
    write_json(root/marker, {'checksums_sha256': file_hash(root/sums)})
    return result


def read_result(root, plan, run):
    root = Path(root)
    stage = plan['stage']
    names = ('COMPLETE',) if stage == 'lr_search' else ('TRAINED', 'TRAINING_FAILED')
    markers = [root/n for n in names if (root/n).exists()]
    sums = root/('checksums.json' if stage == 'lr_search' else 'training_checksums.json')
    if len(markers) != 1 or markers[0].is_symlink() or sums.is_symlink():
        raise ValueError('Missing/ambiguous v2 result marker')
    if json.loads(markers[0].read_text()) != {'checksums_sha256': file_hash(sums)}:
        raise ValueError('Invalid v2 completion marker')
    if json.loads(sums.read_text()) != {'schema_version': 2, 'stage': stage,
                                       'files': inventory(root, stage)}:
        raise ValueError('V2 artifact checksum mismatch')
    result = validate_result(root, plan, run)
    if markers[0].name != result_paths(stage, result['status']['status'])[1]:
        raise ValueError('V2 marker/status mismatch')
    return {**result, 'checksums_sha256': file_hash(sums)}


def collect_results(plan, roots):
    by_id = {r['run_id']: r for r in plan['runs']}
    found = {}
    markers = ('COMPLETE',) if plan['stage'] == 'lr_search' else ('TRAINED', 'TRAINING_FAILED')
    for root in roots:
        for name in markers:
            for marker in Path(root).rglob(name):
                identifier = json.loads((marker.parent/'config.json').read_text())['run']['run_id']
                if identifier not in by_id or identifier in found:
                    raise ValueError('Foreign/duplicate v2 result: '+identifier)
                found[identifier] = read_result(marker.parent, plan, by_id[identifier])
    if set(found) != set(by_id):
        raise ValueError(f'Need all {len(by_id)} resolved v2 tasks; missing '+str(sorted(set(by_id)-set(found))))
    return [found[r['run_id']] for r in plan['runs']]


def collect_search_results(plan, roots):
    if plan['stage'] != 'lr_search':
        raise ValueError('Expected a v2 LR plan')
    trials = collect_results(plan, roots)
    selected = {}
    for name in CONFIGS:
        valid = [t for t in trials if t['run']['config'] == name and t['status']['status'] == 'valid']
        if not valid:
            raise ValueError('All four LR candidates numerically invalid for '+name)
        best = min(valid, key=lambda t: (-t['metrics']['val_correct_count'], t['run']['lr']))
        selected[name] = {'lr': best['run']['lr'], 'run_id': best['run']['run_id'],
                          'val_correct_count': best['metrics']['val_correct_count'],
                          'val_sample_count': 2020, 'best_epoch': best['metrics']['best_epoch']}
    result = {'schema_version': 2, 'stage': 'lr_selection', 'plan_hash': plan['plan_hash'],
              'selected': selected, 'trials': trials}
    validate_selection(portable_selection(result), plan)
    return result


def collect_final_results(plan, roots):
    if plan['stage'] != 'final':
        raise ValueError('Expected a v2 final plan')
    results = collect_results(plan, roots)
    if any(t['status']['status'] != 'trained' for t in results):
        raise ValueError('Numerically invalid final blocks the complete v2 series')
    return {'schema_version': 2, 'stage': 'final', 'status': 'trained',
            'plan_hash': plan['plan_hash'], 'test_evaluated': False,
            'series': SERIES, 'statistics': plan['common']['statistics'], 'runs': results}
