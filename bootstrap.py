"""Read-only environment report and optional verified storage write probe."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys
import tempfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('artifacts/environment.json'))
    p.add_argument('--storage', type=Path, help='Existing persistent directory to probe; no mount/login automation')
    p.add_argument('--require-cuda', action='store_true')
    a = p.parse_args()
    info = {'python':sys.version, 'platform':platform.platform(), 'packages':{},
            'disk_free_gib':shutil.disk_usage(Path.cwd()).free/2**30,
            'note':'bootstrap evidence, not an accepted full experiment lock'}
    for name in ['torch','torchvision','transformers','peft','accelerate','numpy','Pillow','safetensors','huggingface-hub','datasets']:
        try:
            info['packages'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            info['packages'][name] = None
    try:
        import torch
        info['cuda_available'] = torch.cuda.is_available()
        info['torch_cuda_build'] = torch.version.cuda
        info['gpus'] = [{'name':torch.cuda.get_device_properties(i).name,
                         'vram_gib':torch.cuda.get_device_properties(i).total_memory/2**30}
                        for i in range(torch.cuda.device_count())]
    except ImportError:
        info['cuda_available'] = False
    if a.storage:
        if not a.storage.is_dir():
            p.error('Storage directory must already exist; mount/check it first')
        payload = b'LoRA storage read/write probe v1\n'
        with tempfile.TemporaryDirectory(prefix='lora-probe-', dir=a.storage) as d:
            f=Path(d)/'probe.txt'
            f.write_bytes(payload)
            if hashlib.sha256(f.read_bytes()).digest()!=hashlib.sha256(payload).digest():
                raise RuntimeError('Storage readback mismatch')
        info['storage_probe'] = {'path':str(a.storage.resolve()), 'readback_verified':True,
                                 'note':'local disk write/read integrity only'}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(info, indent=2)+'\n')
    print(json.dumps(info, indent=2))
    if a.require_cuda and not info['cuda_available']:
        raise SystemExit('CUDA required but unavailable')


if __name__ == '__main__':
    main()
