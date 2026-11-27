"""Local artifact hashing, exclusive JSON writes and compact best-model exports."""
import hashlib
import json
from pathlib import Path


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def export_best(checkpoint, destination, identity, best_epoch, correct_count):
    import shutil
    from safetensors import safe_open
    from safetensors.torch import save_file
    destination, checkpoint = Path(destination), Path(checkpoint)
    destination.mkdir(exist_ok=False)
    if identity['run']['config'] == 'LP':
        with safe_open(str(checkpoint/'model.safetensors'), framework='pt', device='cpu') as weights:
            tensors = {key: weights.get_tensor(key).float().contiguous()
                       for key in ('classifier.weight', 'classifier.bias')}
        save_file(tensors, str(destination/'head.safetensors'))
    else:
        for name in ('adapter_model.safetensors', 'adapter_config.json'):
            shutil.copy2(checkpoint/name, destination/name)
    write_json(destination/'metadata.json', {**identity, 'best_epoch': best_epoch,
        'val_correct_count': correct_count, 'val_sample_count': 2020})
