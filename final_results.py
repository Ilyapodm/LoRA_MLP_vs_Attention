"""Immutable trained artifacts, distinct from the future completed test result."""
import json
import math
from pathlib import Path

from checkpoints import file_hash, write_json
from final_protocol import config_record

FILES = ('config.json', 'environment.json', 'training_status.json', 'history.json',
         'updates.json', 'metrics.json')
TEST_FIELDS = ('test_accuracy', 'test_correct_count', 'test_sample_count')


def inventory(root):
    root = Path(root)
    if root.is_symlink():
        raise ValueError('Training artifacts must not be symlinks')
    paths = [root/name for name in FILES]
    best = root/'best'
    if best.is_symlink():
        raise ValueError('Best export must not be a symlink')
    for p in sorted(best.rglob('*')):
        if p.is_symlink():
            raise ValueError('Best export contains a symlink')
        if p.is_file():
            paths.append(p)
    if any(p.is_symlink() or not p.is_file() for p in paths):
        raise ValueError('Missing/unsafe final training artifact')
    return {p.relative_to(root).as_posix(): {'sha256': file_hash(p), 'bytes': p.stat().st_size}
            for p in paths}


def validate_training(root, plan, run):
    root = Path(root)
    if json.loads((root/'config.json').read_text()) != config_record(plan, run):
        raise ValueError('Training result belongs to another final run/freeze')
    status = json.loads((root/'training_status.json').read_text())
    if (status.get('schema_version'), status.get('stage'), status.get('action')) != (1, 'final', 'train'):
        raise ValueError('Invalid final training status')
    if status['status'] not in ('trained', 'numerical_invalid'):
        raise ValueError('Unresolved final training')
    metrics = json.loads((root/'metrics.json').read_text())
    if any(k not in metrics or metrics[k] is not None for k in TEST_FIELDS):
        raise ValueError('Training must not contain test metrics')
    history = json.loads((root/'history.json').read_text())
    updates = json.loads((root/'updates.json').read_text())
    # Reject NaN/Infinity even in diagnostic fields rather than sealing misleading JSON.
    from data import canonical_hash
    canonical_hash([metrics, history, updates])
    if status['status'] == 'trained':
        evaluations = [h for h in history if 'eval_correct_count' in h]
        if (len(evaluations) != 5 or [h['epoch'] for h in evaluations] != [1, 2, 3, 4, 5] or
                [h['step'] for h in evaluations] != [36, 72, 108, 144, 180]):
            raise ValueError('Expected five complete validation epochs')
        for h in evaluations:
            count = h['eval_correct_count']
            if (type(count) is not int or not 0 <= count <= 2020 or h['eval_sample_count'] != 2020 or
                    h['eval_accuracy'] != count/2020):
                raise ValueError('Invalid validation counts/accuracy')
        best = max(evaluations, key=lambda h: (h['eval_correct_count'], -h['epoch']))
        expected = {'epochs_completed': 5, 'global_step': 180, 'attempted_steps': 180,
                    'val_correct_count': best['eval_correct_count'], 'val_sample_count': 2020,
                    'val_accuracy': best['eval_accuracy'], 'best_epoch': int(best['epoch'])}
        if any(metrics.get(k) != v for k, v in expected.items()) or len(updates) != 180:
            raise ValueError('Incomplete training or incorrect best selection')
        applied, consecutive = 0, 0
        for step, event in enumerate(updates, 1):
            if event['step'] != step or type(event['skipped']) is not bool:
                raise ValueError('Invalid update audit')
            applied += not event['skipped']
            consecutive = consecutive+1 if event['skipped'] else 0
            lr = run['lr'] * (1-applied/180)
            if (consecutive >= 10 or not event['lr_after_update'] or
                    any(not math.isclose(v, lr, rel_tol=1e-8, abs_tol=1e-12)
                        for v in event['lr_after_update'])):
                raise ValueError('Scheduler/update audit differs from the scientific profile')
        if metrics.get('applied_steps') != applied or metrics.get('skipped_steps') != 180-applied:
            raise ValueError('Update totals differ')
        metadata = json.loads((root/'best/metadata.json').read_text())
        if metadata != {**config_record(plan, run), 'best_epoch': expected['best_epoch'],
                        'val_correct_count': expected['val_correct_count'], 'val_sample_count': 2020}:
            raise ValueError('Best export metadata differs from training')
        required = ('head.safetensors',) if run['config'] == 'LP' else (
            'adapter_model.safetensors', 'adapter_config.json')
        if any(not (root/'best'/name).is_file() for name in required):
            raise ValueError('Missing best model weights')
    else:
        if not status.get('reason') or any(metrics.get(k) is not None for k in (
                'val_accuracy', 'val_correct_count', 'val_sample_count', 'best_epoch')):
            raise ValueError('Numerically invalid final cannot publish a usable best result')
    return {'path': str(root), 'run': run, 'status': status, 'metrics': metrics}


def seal_training(root, plan, run):
    root = Path(root)
    result = validate_training(root, plan, run)
    if any((root/name).exists() for name in ('TRAINED', 'TRAINING_FAILED', 'training_checksums.json')):
        raise FileExistsError('Training result is already sealed')
    write_json(root/'training_checksums.json', {'schema_version': 1, 'files': inventory(root)})
    marker = 'TRAINED' if result['status']['status'] == 'trained' else 'TRAINING_FAILED'
    write_json(root/marker, {'checksums_sha256': file_hash(root/'training_checksums.json')})
    return result


def read_training_result(root, plan, run):
    root = Path(root)
    markers = [root/name for name in ('TRAINED', 'TRAINING_FAILED') if (root/name).exists()]
    sums = root/'training_checksums.json'
    if len(markers) != 1 or markers[0].is_symlink() or sums.is_symlink():
        raise ValueError('Missing/ambiguous final training marker')
    if json.loads(markers[0].read_text()) != {'checksums_sha256': file_hash(sums)}:
        raise ValueError('Invalid final training marker')
    if json.loads(sums.read_text()) != {'schema_version': 1, 'files': inventory(root)}:
        raise ValueError('Final training artifact checksum mismatch')
    result = validate_training(root, plan, run)
    expected = 'TRAINED' if result['status']['status'] == 'trained' else 'TRAINING_FAILED'
    if markers[0].name != expected:
        raise ValueError('Final training marker/status mismatch')
    return result


def collect_training_results(plan, roots):
    expected = {r['run_id']: r for r in plan['runs']}
    found = {}
    for root in roots:
        for name in ('TRAINED', 'TRAINING_FAILED'):
            for marker in Path(root).rglob(name):
                identifier = json.loads((marker.parent/'config.json').read_text())['run']['run_id']
                if identifier not in expected or identifier in found:
                    raise ValueError('Foreign/duplicate final training result: '+identifier)
                result = read_training_result(marker.parent, plan, expected[identifier])
                if result['status']['status'] != 'trained':
                    raise ValueError('Numerically invalid final blocks the complete series: '+identifier)
                found[identifier] = result
    if set(found) != set(expected):
        raise ValueError('Need all 12 trained finals; missing '+str(sorted(set(expected)-set(found))))
    return {'schema_version': 1, 'stage': 'final', 'status': 'trained',
            'plan_hash': plan['plan_hash'], 'test_evaluated': False,
            'runs': [found[r['run_id']] for r in plan['runs']]}
