"""Fresh-process acceptance: full ViT LP/PEFT, interrupted epoch, relocation, resume."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from checkpoints import publish_bundle, verify_bundle, write_json


def assert_nested_close(a, b):
    import numpy as np
    import torch
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_nested_close(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            assert_nested_close(left, right)
    else:
        assert a == b, (a, b)


def compare_weights(left, right):
    import torch
    from safetensors import safe_open
    maximum = 0.0
    with safe_open(str(left), framework='pt', device='cpu') as a, safe_open(str(right), framework='pt', device='cpu') as b:
        assert set(a.keys()) == set(b.keys())
        for key in a.keys():
            first, second = a.get_tensor(key), b.get_tensor(key)
            torch.testing.assert_close(first, second, rtol=2e-5, atol=2e-6, msg=lambda msg: key+': '+msg)
            maximum = max(maximum, float((first-second).abs().max()))
    return maximum


def main():
    import torch
    from safetensors import safe_open

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--model-revision', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--config', choices=('LP', 'B-iso', 'all'), default='all')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    code = Path(__file__).resolve().parent
    reports = []
    for config in ('LP', 'B-iso') if args.config == 'all' else (args.config,):
        root = args.output/config;root.mkdir()
        store = root/'persistent-fixture';store.mkdir()

        def worker(name, extra=()):
            destination = root/name
            command = [sys.executable, str(code/'train.py'), '--stage', 'technical',
                       '--probe', '--force-best-first', '--config', config,
                       '--manifest', str(args.manifest.resolve()), '--model-revision', args.model_revision,
                       '--output', str(destination.resolve()), '--storage-root', str(store.resolve()), *extra]
            with (root/(name+'.log')).open('x') as log:
                result = subprocess.run(command, cwd=code, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f'{config}/{name} failed; see {log.name}')
            return json.loads((destination/'technical-result.json').read_text())

        print(config+': continuous three-epoch technical run', flush=True)
        full = worker('continuous')
        print(config+': interrupt after epoch 2, best deliberately stays at epoch 1', flush=True)
        partial = worker('interrupted', ['--stop-after-epoch', '2'])
        saved = Path(partial['published_bundle'])
        meta = verify_bundle(saved)
        assert meta['global_step'] == 4 and meta['best_global_step'] == 2
        relocated = publish_bundle(saved, root/'transport-fixture')
        # Move the original working directory away: stale absolute best paths cannot work.
        shutil.move(str(root/'interrupted'), str(root/'original-path-moved'))
        print(config+': new process, new output path, restore verified bundle', flush=True)
        resumed = worker('resumed', ['--resume-bundle', str(relocated.resolve())])
        first, second = Path(full['published_bundle']), Path(resumed['published_bundle'])
        a, b = verify_bundle(first), verify_bundle(second)
        assert a['identity'] == b['identity'] == meta['identity']
        assert a['global_step'] == b['global_step'] == 6
        assert a['epoch'] == b['epoch'] == 3
        assert a['best_global_step'] == b['best_global_step'] == 2
        assert a['audit'] == b['audit']
        full_trace = json.loads((root/'continuous'/'probe-trace.json').read_text())
        resumed_trace = json.loads((root/'resumed'/'probe-trace.json').read_text())
        expected_trace = [event for event in full_trace if event['step_before'] >= 4]
        if expected_trace != resumed_trace:
            write_json(root/'trace-difference.json', {'continuous': expected_trace, 'resumed': resumed_trace})
            raise AssertionError('Resume trace differs; see trace-difference.json (inputs/RNG/parameters/optimizer/gradients)')
        name = 'model.safetensors' if config == 'LP' else 'adapter_model.safetensors'
        diffs = {part: compare_weights(first/part/name, second/part/name)
                 for part in ('resume', 'best_model')}
        for file in ('optimizer.pt', 'scheduler.pt', 'scaler.pt'):
            assert_nested_close(torch.load(first/'resume'/file, map_location='cpu', weights_only=True),
                                torch.load(second/'resume'/file, map_location='cpu', weights_only=True))
        # These are our own SHA-verified bundles, produced immediately above.
        assert_nested_close(torch.load(first/'resume'/'rng_state.pth', map_location='cpu', weights_only=False),
                            torch.load(second/'resume'/'rng_state.pth', map_location='cpu', weights_only=False))
        export = 'head.safetensors' if config == 'LP' else name
        diffs['best_export'] = compare_weights(first/'best'/export, second/'best'/export)
        with safe_open(str(second/'best'/export), framework='pt', device='cpu') as f:
            if config == 'LP':
                assert set(f.keys()) == {'classifier.weight', 'classifier.bias'}
            assert all(f.get_tensor(key).dtype == torch.float32 for key in f.keys())
            count = sum(f.get_tensor(key).numel() for key in f.keys())
        assert count == (77669 if config == 'LP' else 1552229), count
        report = {'config': config, 'stage': 'technical', 'scientific_result': False,
                  'same_gpu_relocated_resume': True, 'global_step': 6, 'best_global_step': 2,
                  'max_weight_abs_diff': diffs, 'export_parameters': count,
                  'optimizer_scheduler_scaler_rng_equal': True,
                  'identity': a['identity'], 'relocated_bundle': str(relocated),
                  'full_bundle': str(first), 'resumed_bundle': str(second)}
        write_json(root/'resume-comparison.json', report)
        reports.append(report)
        print(config+': resume accepted, max weight difference '+str(diffs), flush=True)
    write_json(args.output/'resume-acceptance.json', {'stage': 'technical', 'runs': reports})


if __name__ == '__main__':
    main()
