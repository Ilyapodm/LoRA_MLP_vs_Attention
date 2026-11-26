"""Immutable local experiment plan, result integrity and LR selection. No ML imports."""
import hashlib
import json
from pathlib import Path
import re

from configs import CONFIGS, EPOCHS, LR_GRID, SEARCH_SEED, MODEL_ID
from data import DATASET_ID, canonical_hash, read_manifest
from checkpoints import file_hash, write_json

EXECUTORS = ('rtx4070ti', 'rtx3090', 'rtx3060')
PROFILE = {'microbatch': 16, 'accumulation': 32, 'eval_batch': 32,
           'gradient_checkpointing': False, 'precision': 'fp32-amp-fp16',
           'attention': 'sdpa', 'tf32': False, 'workers': 0}
SOURCE_FILES = ('configs.py', 'data.py', 'train.py', 'checkpoints.py',
                'experiment.py', 'run_queue.py', 'requirements.txt')


def source_hash():
    root = Path(__file__).resolve().parent
    return canonical_hash({name: file_hash(root/name) for name in SOURCE_FILES})


def validate_profile(profile):
    fixed = {k: v for k, v in PROFILE.items() if k not in
             ('microbatch', 'accumulation', 'eval_batch', 'gradient_checkpointing')}
    if set(profile) != set(PROFILE) or any(profile[k] != v for k, v in fixed.items()):
        raise ValueError('Unsupported math/data-loader profile')
    if (profile['microbatch'], profile['accumulation']) not in ((16, 32), (8, 64), (4, 128)):
        raise ValueError('Expected a common effective batch of 512')
    if profile['eval_batch'] not in (32, 16, 8) or type(profile['gradient_checkpointing']) is not bool:
        raise ValueError('Invalid evaluation/checkpointing profile')
    return profile


def protocol_info(directory):
    root = Path(directory)
    manifest = read_manifest(root/'data_manifest.json')
    model = json.loads((root/'model_revision.json').read_text())
    dataset = json.loads((root/'dataset_revision.json').read_text())
    if dataset != {'dataset': DATASET_ID, 'revision': manifest['dataset_revision']}:
        raise ValueError('Dataset pin differs from manifest')
    if model.get('model') != MODEL_ID or not re.fullmatch('[0-9a-f]{40}', model.get('revision', '')):
        raise ValueError('Expected the agreed model with a full commit SHA')
    return {'model': MODEL_ID, 'model_revision': model['revision'],
            'dataset': DATASET_ID, 'dataset_revision': manifest['dataset_revision'],
            'manifest_hash': manifest['content_hash']}


def run_id(common, config, lr):
    return 'lr-'+canonical_hash({'common': common, 'config': config, 'lr': float(lr),
                                 'seed': SEARCH_SEED, 'stage': 'lr_search'})[:20]


def make_plan(protocol, profile=None):
    profile = validate_profile(dict(PROFILE if profile is None else profile))
    common = {'protocol_version': 'food101-local-v1', **protocol_info(protocol),
              'epochs': EPOCHS, 'seed': SEARCH_SEED, 'train_images': 18180,
              'val_images': 2020, 'python': '3.12', 'profile': profile,
              'source_hash': source_hash()}
    runs = []
    # Each method sees each GPU once. Each GPU receives all four methods.
    for method_index, config in enumerate(CONFIGS):
        for lr_index, lr in enumerate(LR_GRID):
            executor = EXECUTORS[(lr_index-method_index) % 3]
            runs.append({'run_id': run_id(common, config, lr), 'stage': 'lr_search',
                         'config': config, 'lr': lr, 'seed': SEARCH_SEED, 'executor': executor})
    plan = {'schema_version': 1, 'common': common, 'runs': runs}
    plan['plan_hash'] = canonical_hash(plan)
    return validate_plan(plan)


def validate_plan(plan):
    if plan.get('schema_version') != 1 or canonical_hash({k: v for k, v in plan.items() if k != 'plan_hash'}) != plan.get('plan_hash'):
        raise ValueError('Plan schema/hash mismatch')
    common = plan['common']
    validate_profile(common['profile'])
    if (common['epochs'], common['seed'], common['train_images'], common['val_images'],
        common['python'], common['model'], common['dataset'], common['protocol_version']) != (
            5, 142, 18180, 2020, '3.12', MODEL_ID, DATASET_ID, 'food101-local-v1'):
        raise ValueError('Scientific protocol differs from the agreed local plan')
    for field, size in (('model_revision', 40), ('dataset_revision', 40),
                        ('manifest_hash', 64), ('source_hash', 64)):
        if not re.fullmatch('[0-9a-f]{'+str(size)+'}', common[field]):
            raise ValueError('Invalid immutable pin: '+field)
    expected = {(c, float(lr)) for c in CONFIGS for lr in LR_GRID}
    pairs, ids = set(), set()
    for run in plan['runs']:
        pair = (run['config'], float(run['lr']))
        if pair not in expected or pair in pairs or run['executor'] not in EXECUTORS:
            raise ValueError('Duplicate/unsupported LR task or executor')
        if run['stage'] != 'lr_search' or run['seed'] != SEARCH_SEED or run['run_id'] != run_id(common, *pair):
            raise ValueError('Run identity differs from the plan')
        pairs.add(pair);ids.add(run['run_id'])
    if pairs != expected or len(ids) != 12:
        raise ValueError('The plan must contain exactly all 12 LR trials')
    return plan


def read_plan(path, protocol=None, check_source=True):
    plan = validate_plan(json.loads(Path(path).read_text()))
    if check_source and plan['common']['source_hash'] != source_hash():
        raise ValueError('Computational code differs; create a new plan before training')
    if protocol is not None:
        for key, value in protocol_info(protocol).items():
            if plan['common'][key] != value:
                raise ValueError('Protocol/plan mismatch: '+key)
    return plan


def finish_result(directory):
    root = Path(directory)
    names = ('config.json', 'environment.json', 'status.json', 'history.json', 'updates.json', 'metrics.json')
    for name in names:
        if not (root/name).is_file():
            raise ValueError('Incomplete LR result: '+name)
    status = json.loads((root/'status.json').read_text())
    if status['stage'] != 'lr_search' or status['status'] not in ('valid', 'numerical_invalid'):
        raise ValueError('Only resolved LR trials receive COMPLETE')
    best = root/'best'
    if status['status'] == 'valid' and not (best/'metadata.json').is_file():
        raise ValueError('Valid result needs a best export')
    files = [root/name for name in names]+[p for p in best.rglob('*') if p.is_file()]
    sums = {p.relative_to(root).as_posix(): {'sha256': file_hash(p), 'bytes': p.stat().st_size} for p in files}
    write_json(root/'checksums.json', {'schema_version': 1, 'files': sums})
    write_json(root/'COMPLETE', {'checksums_sha256': file_hash(root/'checksums.json')})


def read_result(directory, plan, run):
    root = Path(directory)
    marker = json.loads((root/'COMPLETE').read_text())
    if marker != {'checksums_sha256': file_hash(root/'checksums.json')}:
        raise ValueError('Invalid COMPLETE marker')
    sums = json.loads((root/'checksums.json').read_text())
    mandatory = {'config.json', 'environment.json', 'status.json', 'history.json', 'updates.json', 'metrics.json'}
    if sums.get('schema_version') != 1 or not mandatory.issubset(sums['files']):
        raise ValueError('Incomplete result checksums')
    for name, expected in sums['files'].items():
        p = (root/name).resolve()
        if not p.is_relative_to(root.resolve()) or p.is_symlink() or p.stat().st_size != expected['bytes'] or file_hash(p) != expected['sha256']:
            raise ValueError('LR artifact checksum mismatch: '+name)
    config = json.loads((root/'config.json').read_text())
    if config['common'] != plan['common'] or config['run'] != run:
        raise ValueError('Result belongs to another scientific run')
    status = json.loads((root/'status.json').read_text())
    metrics = json.loads((root/'metrics.json').read_text())
    if status['stage'] != 'lr_search' or status['status'] not in ('valid', 'numerical_invalid'):
        raise ValueError('LR result is not resolved')
    if any(metrics.get(k) is not None for k in ('test_accuracy', 'test_correct_count', 'test_sample_count')):
        raise ValueError('LR results must not contain test metrics')
    if status['status'] == 'valid':
        if metrics['epochs_completed'] != 5 or metrics['val_sample_count'] != 2020 or not 0 <= metrics['val_correct_count'] <= 2020:
            raise ValueError('Incomplete scientific training/validation')
        if metrics['val_accuracy'] != metrics['val_correct_count']/2020 or not 1 <= metrics['best_epoch'] <= 5:
            raise ValueError('Invalid best validation metrics')
        required = {'best/metadata.json', 'best/head.safetensors' if run['config'] == 'LP' else 'best/adapter_model.safetensors'}
        if run['config'] != 'LP': required.add('best/adapter_config.json')
        if not required.issubset(sums['files']):
            raise ValueError('Missing best weights')
    return {'path': str(root), 'run': run, 'status': status, 'metrics': metrics}


def collect_results(plan, roots):
    by_id = {r['run_id']: r for r in plan['runs']}
    found = {}
    for root in roots:
        for marker in Path(root).rglob('COMPLETE'):
            config = json.loads((marker.parent/'config.json').read_text())
            identifier = config.get('run', {}).get('run_id')
            if identifier not in by_id:
                raise ValueError('Foreign result in collection: '+str(marker))
            result = read_result(marker.parent, plan, by_id[identifier])
            if identifier in found:
                raise ValueError('Duplicate resolved run: '+identifier)
            found[identifier] = result
    if set(found) != set(by_id):
        raise ValueError('Need all 12 resolved LR trials; missing '+str(sorted(set(by_id)-set(found))))
    selected = {}
    for config in CONFIGS:
        valid = [r for r in found.values() if r['run']['config'] == config and r['status']['status'] == 'valid']
        if not valid:
            raise ValueError('All LR candidates numerically invalid for '+config+'; stop and diagnose')
        winner = min(valid, key=lambda r: (-r['metrics']['val_correct_count'], r['run']['lr']))
        selected[config] = {'lr': winner['run']['lr'], 'run_id': winner['run']['run_id'],
                            'val_correct_count': winner['metrics']['val_correct_count'],
                            'val_sample_count': 2020, 'best_epoch': winner['metrics']['best_epoch']}
    return {'schema_version': 1, 'stage': 'lr_selection', 'plan_hash': plan['plan_hash'],
            'selected': selected, 'trials': [found[r['run_id']] for r in plan['runs']]}
