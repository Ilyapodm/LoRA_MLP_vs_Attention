"""Export/compare CPU initial heads without training, downloads or changing the v2 release."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
from itertools import combinations
import json
import os
from pathlib import Path
import platform
import zipfile

from file_utils import file_hash, write_json
from study_configs import CONFIGS, configure_model, active_head, head_hash, tensor_hash
from study_protocol import read_plan, source_manifest
from study_queue import validate_admission
from study_runtime import EXECUTORS

SEEDS = (42, 142)
FILES = ('heads.safetensors', 'report.json', 'torch-build.txt')


def head_digest(weight, bias):
    return hashlib.sha256((tensor_hash(weight)+tensor_hash(bias)).encode()).hexdigest()


def verify_export(directory):
    """Verify original bytes and tensor fingerprints before comparing shipments."""
    from safetensors.torch import load_file
    import torch
    directory = Path(directory)
    paths = [directory/name for name in (*FILES, 'checksums.json')]
    if any(p.is_symlink() or not p.is_file() for p in paths):
        raise ValueError('Missing/unsafe head-check file: '+str(directory))
    expected = {name: {'sha256': file_hash(directory/name), 'bytes': (directory/name).stat().st_size}
                for name in FILES}
    if json.loads((directory/'checksums.json').read_text()) != expected:
        raise ValueError('Head-check checksum mismatch: '+str(directory))
    report = json.loads((directory/'report.json').read_text())
    tensors = load_file(str(directory/'heads.safetensors'), device='cpu')
    required = {f'seed{seed}.{kind}' for seed in SEEDS for kind in ('weight', 'bias', 'uniform')}
    if set(tensors) != required or report.get('seeds') != list(SEEDS):
        raise ValueError('Unexpected diagnostic tensor inventory')
    for seed in SEEDS:
        for kind, shape in (('weight', (101, 768)), ('bias', (101,)), ('uniform', (101, 768))):
            value = tensors[f'seed{seed}.{kind}']
            if value.dtype != torch.float32 or tuple(value.shape) != shape or not torch.isfinite(value).all().item():
                raise ValueError(f'Invalid diagnostic tensor seed{seed}.{kind}')
        record = report['heads'][str(seed)]
        if head_digest(tensors[f'seed{seed}.weight'], tensors[f'seed{seed}.bias']) != record['head_hash']:
            raise ValueError('Head fingerprint differs from exported tensor')
        if tensor_hash(tensors[f'seed{seed}.uniform']) != record['uniform_hash']:
            raise ValueError('Uniform fingerprint differs from exported tensor')
        if torch.count_nonzero(tensors[f'seed{seed}.bias']).item():
            raise ValueError('Expected a zero initial classifier bias')
    return report, tensors


def export_heads(args, plan):
    import torch
    from safetensors.torch import save_file
    from transformers import ViTConfig, ViTForImageClassification
    from study_train import configure_runtime

    directory = args.output or Path('artifacts/head-check')/args.executor
    archive = directory.parent/(directory.name+'.zip')
    if directory.exists() or archive.exists():
        raise FileExistsError('Output already exists; preserve it or use a new --output path')
    if args.admission:
        receipt_path = args.admission
    else:
        candidates = list((Path('results/v2/lr')/args.executor).glob('preflight/*/admission.json'))
        if len(candidates) != 1:
            raise ValueError('Expected exactly one v2 preflight; provide its --admission PATH')
        receipt_path = candidates[0]
    receipt = validate_admission(receipt_path, plan, args.executor)
    parameter_path = receipt_path.parent/'parameters.json'
    parameters = json.loads(parameter_path.read_text())['runs']
    if Counter(row['config'] for row in parameters) != Counter(CONFIGS.keys()):
        raise ValueError('Preflight must contain all six configurations exactly once')
    previous_hashes = {row['head_hash'] for row in parameters}
    if len(previous_hashes) != 1:
        raise ValueError('Preflight methods disagree on the initial head')

    # Match study_check.parameter_check exactly up to its first audit_model call.
    # Model construction, classifier initialization and RNG probes all stay on CPU.
    configure_runtime(42)
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(12345)
        base = ViTForImageClassification(ViTConfig(num_labels=101, attn_implementation='sdpa'))
    tensors, records = {}, {}
    for seed in SEEDS:
        model = configure_model(base, CONFIGS['LP'], seed)
        head = active_head(model)
        if head.weight.device.type != 'cpu' or head.weight.dtype != torch.float32:
            raise ValueError('Expected original FP32 CPU head initialization')
        tensors[f'seed{seed}.weight'] = head.weight.detach().clone().contiguous()
        tensors[f'seed{seed}.bias'] = head.bias.detach().clone().contiguous()
        generator = torch.Generator(device='cpu').manual_seed(seed)
        uniform = torch.empty_like(head.weight).uniform_(-1, 1, generator=generator)
        tensors[f'seed{seed}.uniform'] = uniform
        records[str(seed)] = {'head_hash': head_hash(model), 'uniform_hash': tensor_hash(uniform),
                              'weight_std': head.weight.detach().double().std().item(),
                              'weight_first8': head.weight.detach().flatten()[:8].tolist(),
                              'uniform_first8': uniform.flatten()[:8].tolist()}
        print(f'SEED {seed}: {records[str(seed)]["head_hash"]}', flush=True)
    if plan['source_files'] != source_manifest():
        raise ValueError('Computational source changed while exporting heads')
    packages = {name: importlib.metadata.version(name) for name in receipt['environment']['packages']}
    current = {'python': platform.python_version(), 'packages': packages,
               'cuda_build': torch.version.cuda, 'platform': platform.platform()}
    differences = {key: {'preflight': receipt['environment'][key], 'current': value}
                   for key, value in current.items() if receipt['environment'][key] != value}
    matches = records['42']['head_hash'] == next(iter(previous_hashes))
    report = {'schema_version': 1, 'stage': 'initial-head-diagnostic', 'scientific_result': False,
              'executor': args.executor, 'seeds': list(SEEDS), 'plan_hash': plan['plan_hash'],
              'source_hash': plan['common']['source_hash'], 'source_files': source_manifest(),
              'diagnostic_source_sha256': file_hash(__file__), 'device': 'cpu',
              'environment': current, 'environment_differences_since_preflight': differences,
              'cpu': {'machine': platform.machine(), 'processor': platform.processor(),
                      'capability': torch.backends.cpu.get_cpu_capability(),
                      'threads': torch.get_num_threads(), 'default_dtype': str(torch.get_default_dtype()),
                      'math_environment': {key: os.environ.get(key) for key in
                                           ('ATEN_CPU_CAPABILITY', 'MKL_CBWR', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')}},
              'initializer_range': base.config.initializer_range,
              'preflight': {'admission_sha256': file_hash(receipt_path),
                            'parameters_sha256': file_hash(parameter_path),
                            'expected_seed42_head_hash': next(iter(previous_hashes)),
                            'seed42_head_matches': matches}, 'heads': records}
    directory.mkdir(parents=True, exist_ok=False)
    save_file(tensors, str(directory/'heads.safetensors'))
    write_json(directory/'report.json', report)
    (directory/'torch-build.txt').write_text(torch.__config__.show(), encoding='utf-8')
    write_json(directory/'checksums.json', {
        name: {'sha256': file_hash(directory/name), 'bytes': (directory/name).stat().st_size} for name in FILES})
    verify_export(directory)
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as bundle:
        for name in (*FILES, 'checksums.json'):
            bundle.write(directory/name, directory.name+'/'+name)
    print('PREFLIGHT_HEAD_MATCH:', 'YES' if matches else 'NO')
    print('ENVIRONMENT_MATCH:', 'YES' if not differences else 'NO (see report.json)')
    print('SEND_ARCHIVE:', archive.resolve())
    print('ARCHIVE_SHA256:', file_hash(archive))


def tensor_difference(left, right):
    import torch
    left64, right64 = left.double(), right.double()
    difference = left64-right64
    denominator = max(left64.norm().item(), right64.norm().item())
    return {'exact_equal': torch.equal(left, right), 'different_values': int((left != right).sum().item()),
            'total_values': left.numel(), 'max_abs': difference.abs().max().item(),
            'mean_abs': difference.abs().mean().item(),
            'relative_l2': difference.norm().item()/denominator if denominator else 0.0}


def compare_heads(args, plan):
    exports = [verify_export(path) for path in args.compare]
    if Counter(report['executor'] for report, _ in exports) != Counter(EXECUTORS):
        raise ValueError('Provide exactly one export from each of rtx4070ti, rtx3090 and rtx3060')
    for report, _ in exports:
        if (report['plan_hash'], report['source_hash'], report['source_files']) != (
                plan['plan_hash'], plan['common']['source_hash'], plan['source_files']):
            raise ValueError('Export belongs to a different computational release/plan')
        if report['diagnostic_source_sha256'] != file_hash(__file__):
            raise ValueError('Export came from another version of this diagnostic script')
    comparisons = []
    for (left_report, left), (right_report, right) in combinations(exports, 2):
        for seed in SEEDS:
            entry = {'left': left_report['executor'], 'right': right_report['executor'], 'seed': seed,
                     **{kind: tensor_difference(left[f'seed{seed}.{kind}'], right[f'seed{seed}.{kind}'])
                        for kind in ('weight', 'bias', 'uniform')}}
            comparisons.append(entry)
            print(f'{entry["left"]} / {entry["right"]}, seed={seed}: '
                  f'head max_abs={entry["weight"]["max_abs"]:.9g}, '
                  f'rel_L2={entry["weight"]["relative_l2"]:.9g}, '
                  f'uniform_equal={entry["uniform"]["exact_equal"]}')
    output = args.output or Path('artifacts/head-check/comparison')
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'comparison.json', {'schema_version': 1, 'stage': 'initial-head-comparison',
        'scientific_result': False, 'plan_hash': plan['plan_hash'],
        'exports': [report for report, _ in exports], 'comparisons': comparisons,
        'interpretation': 'No automatic tolerance or acceptance verdict. Check preflight reproduction and environment first; compare uniform RNG and weight differences to establish magnitude/cause.'})
    print('COMPARISON:', (output/'comparison.json').resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--export', action='store_true')
    modes.add_argument('--compare', nargs=3, type=Path, metavar='DIRECTORY')
    parser.add_argument('--executor', choices=EXECUTORS)
    parser.add_argument('--admission', type=Path)
    parser.add_argument('--plan', type=Path, default=Path('runs-v2.json'))
    parser.add_argument('--protocol', type=Path, default=Path('artifacts/protocol'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.export and not args.executor:
        parser.error('--export requires --executor')
    plan = read_plan(args.plan, args.protocol)
    if plan['stage'] != 'lr_search':
        parser.error('Use the original v2 LR plan')
    if args.export:
        export_heads(args, plan)
    else:
        compare_heads(args, plan)


if __name__ == '__main__':
    main()
