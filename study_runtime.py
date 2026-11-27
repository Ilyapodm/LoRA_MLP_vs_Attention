"""Local GPU environment, shared numerical profile, data pins and queue locking."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import platform
import re

from data import DATASET_ID, read_manifest
from study_configs import MODEL_ID

EXECUTORS = ('rtx4070ti', 'rtx3090', 'rtx3060')
PROFILE = {'microbatch': 16, 'accumulation': 32, 'eval_batch': 32,
           'gradient_checkpointing': False, 'precision': 'fp32-amp-fp16',
           'attention': 'sdpa', 'tf32': False, 'workers': 0}
GPU_PATTERNS = {'rtx4070ti': r'RTX\s*4070\s*TI(?!\s*SUPER)',
                'rtx3090': r'RTX\s*3090(?!\s*TI)', 'rtx3060': r'RTX\s*3060(?!\s*TI)'}


def validate_profile(profile):
    fixed = {k: v for k, v in PROFILE.items() if k not in
             ('microbatch', 'accumulation', 'eval_batch', 'gradient_checkpointing')}
    if set(profile) != set(PROFILE) or any(profile[k] != v for k, v in fixed.items()):
        raise ValueError('Unsupported math/data-loader profile')
    if any(type(profile[k]) is not int for k in ('microbatch', 'accumulation', 'eval_batch', 'workers')):
        raise ValueError('Runtime batch/worker values must be integers')
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


def environment():
    import importlib.metadata
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('Expose exactly one CUDA GPU (CUDA_VISIBLE_DEVICES)')
    packages = {name: importlib.metadata.version(name) for name in (
        'torch', 'torchvision', 'transformers', 'peft', 'accelerate', 'datasets',
        'numpy', 'Pillow', 'huggingface-hub', 'safetensors')}
    for line in Path(__file__).with_name('requirements.txt').read_text().splitlines():
        if '==' in line and not line.lstrip().startswith('#'):
            name, version = line.strip().split('==')
            actual = packages[name]
            if actual.split('+')[0] != version:
                raise ValueError(f'Package mismatch: {name}: {actual} != {version}')
    if platform.python_version_tuple()[:2] != ('3', '12') or torch.version.cuda != '12.6':
        raise ValueError('Scientific PC profile requires Python 3.12 and torch CUDA build 12.6')
    device = torch.cuda.get_device_properties(0)
    return {'python': platform.python_version(), 'packages': packages,
            'cuda_build': torch.version.cuda, 'gpu': device.name,
            'vram_gib': device.total_memory/2**30, 'platform': platform.platform()}


def validate_environment_receipt(path, plan, executor, current=None):
    receipt = json.loads(Path(path).read_text())
    if (receipt.get('status') != 'passed' or receipt.get('plan_hash') != plan['plan_hash'] or
            receipt.get('executor') != executor):
        raise ValueError('Missing/mismatched preflight; run --preflight first')
    if current is not None:
        if not re.search(GPU_PATTERNS[executor], current['gpu'], re.I):
            raise ValueError('Executor does not match the actual GPU')
        if receipt['environment'] != current:
            raise ValueError('Environment/device changed since preflight')
    return receipt


@contextmanager
def queue_lock(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    lock = root/'queue.lock'
    with lock.open('x') as stream:
        json.dump({'pid': os.getpid(), 'host': platform.node()}, stream)
    try:
        yield
    finally:
        lock.unlink()
