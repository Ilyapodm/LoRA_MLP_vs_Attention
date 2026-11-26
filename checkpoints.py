"""Verified epoch bundles. A directory without a valid READY is never resumable."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import time
import uuid


class StorageSyncError(RuntimeError):
    pass


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def inventory(root):
    root = Path(root)
    result = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError('Bundles must not contain symlinks')
        if path.is_file():
            name = path.relative_to(root).as_posix()
            if name not in ('READY', 'checksums.json'):
                result[name] = {'sha256': file_hash(path), 'bytes': path.stat().st_size}
    return result


def verify_bundle(root, identity=None, require_ready=True):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('Expected a real bundle directory')
    sums = root/'checksums.json'
    if sums.is_symlink():
        raise ValueError('Symlink checksum file')
    expected = json.loads(sums.read_text())
    if expected.get('schema_version') != 1 or not isinstance(expected.get('files'), dict):
        raise ValueError('Unsupported bundle checksums')
    for name in expected['files']:
        path = PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or str(path) != name:
            raise ValueError('Unsafe checksum path')
    if inventory(root) != expected['files']:
        raise ValueError('Bundle file/checksum mismatch')
    if require_ready:
        marker = root/'READY'
        if marker.is_symlink() or json.loads(marker.read_text()) != {'checksums_sha256': file_hash(sums)}:
            raise ValueError('Invalid READY')
    metadata = json.loads((root/'metadata.json').read_text())
    if metadata.get('schema_version') != 1:
        raise ValueError('Unsupported bundle metadata')
    if identity is not None and metadata['identity'] != identity:
        raise ValueError('Resume identity differs (code/data/model/profile/packages)')
    for name in ('optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'trainer_state.json'):
        if not (root/'resume'/name).is_file():
            raise ValueError('Missing resume state: '+name)
    if metadata['amp'] and not (root/'resume'/'scaler.pt').is_file():
        raise ValueError('Missing AMP scaler state')
    state = json.loads((root/'resume'/'trainer_state.json').read_text())
    if state['global_step'] != metadata['global_step'] or state['epoch'] != metadata['epoch']:
        raise ValueError('Trainer state/metadata progress mismatch')
    if state['best_model_checkpoint'] != 'best_model':
        raise ValueError('Stored best path must be relative')
    model_file = 'model.safetensors' if metadata['identity']['config'] == 'LP' else 'adapter_model.safetensors'
    config_file = 'config.json' if metadata['identity']['config'] == 'LP' else 'adapter_config.json'
    for folder in ('resume', 'best_model'):
        if not (root/folder/model_file).is_file() or not (root/folder/config_file).is_file():
            raise ValueError('Missing Trainer model: '+folder)
    export_file = 'head.safetensors' if metadata['identity']['config'] == 'LP' else model_file
    if not (root/'best'/export_file).is_file() or not (root/'best'/'metadata.json').is_file():
        raise ValueError('Missing compact best export')
    return metadata


def seal_bundle(root):
    root = Path(root)
    write_json(root/'checksums.json', {'schema_version': 1, 'files': inventory(root)})
    verify_bundle(root, require_ready=False)
    write_json(root/'READY', {'checksums_sha256': file_hash(root/'checksums.json')})
    verify_bundle(root)


def create_bundle(checkpoint, best_checkpoint, destination, metadata):
    """Keep a separate Trainer-compatible best model, even if it is older."""
    checkpoint, best_checkpoint, destination = map(Path, (checkpoint, best_checkpoint, destination))
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copytree(checkpoint, destination/'resume')
    # Paths in stored trainer state are relative; restore_bundle rewrites a local copy.
    state_path = destination/'resume'/'trainer_state.json'
    state = json.loads(state_path.read_text())
    state['best_model_checkpoint'] = 'best_model'
    state_path.write_text(json.dumps(state, indent=2)+'\n')
    best = destination/'best_model'
    best.mkdir()
    for source in best_checkpoint.iterdir():
        if source.is_file() and source.suffix in ('.json', '.safetensors') and source.name not in ('trainer_state.json',):
            shutil.copy2(source, best/source.name)
    export = destination/'best'
    export.mkdir()
    if metadata['identity']['config'] == 'LP':
        from safetensors import safe_open
        from safetensors.torch import save_file
        with safe_open(str(best/'model.safetensors'), framework='pt', device='cpu') as f:
            head = {name: f.get_tensor(name).float().contiguous()
                    for name in ('classifier.weight', 'classifier.bias')}
        save_file(head, str(export/'head.safetensors'))
    else:
        for name in ('adapter_config.json', 'adapter_model.safetensors'):
            shutil.copy2(best/name, export/name)
    write_json(export/'metadata.json', {'schema_version': 1, 'identity': metadata['identity'],
                                       'best_global_step': metadata['best_global_step'],
                                       'best_correct_count': metadata['best_correct_count']})
    write_json(destination/'metadata.json', metadata)
    seal_bundle(destination)
    return destination


def publish_bundle(source, storage, copy_file=shutil.copy2, wait=time.sleep):
    """Three attempts, fresh directories, READY last. Existing versions are untouched."""
    source, storage = Path(source), Path(storage)
    metadata = verify_bundle(source)
    last_error = None
    for attempt in range(3):
        target = storage/(source.name+'-'+uuid.uuid4().hex[:12])
        try:
            storage.mkdir(parents=True, exist_ok=True)
            target.mkdir(exist_ok=False)
            for item in sorted(source.rglob('*')):
                name = item.relative_to(source)
                if name.as_posix() == 'READY':
                    continue
                dest = target/name
                if item.is_dir():
                    dest.mkdir(parents=True, exist_ok=True)
                else:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    copy_file(item, dest)
            verify_bundle(target, metadata['identity'], require_ready=False)
            write_json(target/'READY', {'checksums_sha256': file_hash(target/'checksums.json')})
            verify_bundle(target, metadata['identity'])
            return target
        except (OSError, ValueError, KeyError) as error:
            last_error = error
            try:
                (target/'READY').unlink(missing_ok=True)
            except OSError:
                pass  # Readers still reject invalid/incomplete bundles.
            if attempt < 2:
                wait((5, 15)[attempt])
    raise StorageSyncError(f'storage_sync_failed after 3 attempts: {last_error}') from last_error


def restore_bundle(source, destination, identity):
    """Verify before copying/loading; never mutate the confirmed source bundle."""
    source, destination = Path(source), Path(destination)
    metadata = verify_bundle(source, identity)
    if destination.exists():
        raise FileExistsError('Restore destination must be new')
    shutil.copytree(source, destination)
    verify_bundle(destination, identity)
    state_path = destination/'resume'/'trainer_state.json'
    state = json.loads(state_path.read_text())
    state['best_model_checkpoint'] = str((destination/'best_model').resolve())
    state_path.write_text(json.dumps(state, indent=2)+'\n')
    # This is now a mutable local working copy, not another confirmed bundle.
    (destination/'READY').unlink()
    return destination/'resume', metadata


def prune_bundles(storage, identity, keep=2):
    """Only delete verified bundles of this identity after a newer copy was verified."""
    if keep < 2:
        raise ValueError('Keep at least two verified epoch bundles')
    versions = []
    for directory in Path(storage).iterdir():
        if not directory.is_dir() or directory.is_symlink():
            continue
        try:
            meta = verify_bundle(directory, identity)
        except (OSError, ValueError, KeyError):
            continue
        versions.append((meta['global_step'], directory.name, directory))
    for _, _, directory in sorted(versions, reverse=True)[keep:]:
        shutil.rmtree(directory)
