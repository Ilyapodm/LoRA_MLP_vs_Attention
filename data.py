"""Food-101 train/val preparation. This module never constructs a test loader."""
import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import io
import json
from pathlib import Path
import random
import re

DATASET_ID = 'ethz/food101'
SELECTION_VERSION = 'pcg64-class-permutation-v1'
TRANSFORM_VERSION = 'rgb-bilinear-crop224-v1'


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def validate_manifest(manifest):
    if manifest['schema_version'] != 1 or manifest['dataset_id'] != DATASET_ID:
        raise ValueError('Unsupported manifest schema/dataset')
    if not re.fullmatch('[0-9a-f]{40}', manifest['dataset_revision']):
        raise ValueError('Dataset revision must be a full commit SHA')
    if manifest['selection_version'] != SELECTION_VERSION or manifest['split_seed'] != 42:
        raise ValueError('Unexpected selection protocol')
    if len(manifest['label_names']) != 101 or len(set(manifest['label_names'])) != 101:
        raise ValueError('Expected 101 unique class names')
    payload = {k: v for k, v in manifest.items() if k != 'content_hash'}
    if canonical_hash(payload) != manifest['content_hash']:
        raise ValueError('Manifest content hash mismatch')
    seen = set()
    for split, per_class in [('train', 180), ('val', 20)]:
        entries = manifest[split]
        if len(entries) != 101 * per_class or Counter(e['label'] for e in entries) != Counter({i: per_class for i in range(101)}):
            raise ValueError(f'Wrong class counts for {split}')
        for entry in entries:
            i = entry['row_index']
            expected = sample_id(manifest['dataset_revision'], i)
            if type(i) is not int or not 0 <= i < 75750 or entry['sample_id'] != expected:
                raise ValueError('Invalid source row/ID')
            if expected in seen:
                raise ValueError('Duplicate image or train/val overlap')
            seen.add(expected)
            if not re.fullmatch('[0-9a-f]{64}', entry['image_sha256']):
                raise ValueError('Invalid image hash')
    return manifest


def sample_id(revision, index):
    return f'{DATASET_ID}@{revision}/train/{index}'


def select_indices(labels):
    import numpy as np
    if len(labels) != 75750 or Counter(labels) != Counter({i: 750 for i in range(101)}):
        raise ValueError('Unexpected official Food-101 train counts')
    rng = np.random.Generator(np.random.PCG64(42))
    groups = {i: [] for i in range(101)}
    for index, label in enumerate(labels):
        groups[int(label)].append(index)
    train, val = [], []
    for label in range(101):
        chosen = rng.permutation(groups[label])[:200].tolist()
        train.extend(chosen[:180])
        val.extend(chosen[180:])
    return train, val


def image_bytes(value):
    if value.get('bytes') is not None:
        return value['bytes']
    if value.get('path'):
        return Path(value['path']).read_bytes()
    raise ValueError('Image has neither bytes nor readable path')


def official_train(revision):
    from datasets import Image, load_dataset
    if not re.fullmatch('[0-9a-f]{40}', revision or ''):
        raise ValueError('Pass an immutable dataset commit SHA')
    return load_dataset(DATASET_ID, revision=revision, split='train').cast_column('image', Image(decode=False))


def prepare_manifest(dataset, revision, output):
    from PIL import Image
    output = Path(output)
    if output.exists():
        raise FileExistsError('Manifest already exists; verify/reuse it instead of overwriting')
    label_names = dataset.features['label'].names
    indices = select_indices([int(x) for x in dataset['label']])
    result = {'schema_version': 1, 'dataset_id': DATASET_ID, 'dataset_revision': revision,
              'selection_version': SELECTION_VERSION, 'split_seed': 42,
              'label_names': label_names}
    for split, chosen in zip(('train', 'val'), indices):
        entries = []
        for index in chosen:
            row = dataset[index]
            raw = image_bytes(row['image'])
            with Image.open(io.BytesIO(raw)) as im:
                im.convert('RGB').load()  # Decode every selected image before acceptance.
            entries.append({'sample_id': sample_id(revision, index), 'row_index': index,
                            'label': int(row['label']), 'image_sha256': hashlib.sha256(raw).hexdigest()})
        result[split] = entries
    result['content_hash'] = canonical_hash(result)
    validate_manifest(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create: two executors must not silently replace the shared manifest.
    with output.open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write('\n')
    return result


def read_manifest(path):
    return validate_manifest(json.loads(Path(path).read_text()))


def augmentation_seed(seed, epoch, sample):
    value = json.dumps([TRANSFORM_VERSION, seed, epoch, sample], separators=(',', ':'))
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], 'big')


@contextmanager
def image_rng(seed):
    import numpy as np
    import torch
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            # torch.manual_seed also changes CUDA generators: avoid disturbing dropout.
            torch.default_generator.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed % 2**32)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def make_transform(processor, training):
    from torchvision import transforms as T
    interpolation = T.InterpolationMode.BILINEAR
    spatial = [T.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(3/4, 4/3),
                                  interpolation=interpolation, antialias=True),
               T.RandomHorizontalFlip(p=0.5)] if training else [
                   T.Resize(224, interpolation=interpolation, antialias=True), T.CenterCrop(224)]
    return T.Compose([*spatial, T.ToTensor(), T.Normalize(processor.image_mean, processor.image_std)])


class ManifestDataset:
    def __init__(self, source, manifest, split, processor, seed):
        if split not in ('train', 'val'):
            raise ValueError('Only train and val are allowed here; test requires the final-stage gate')
        validate_manifest(manifest)
        self.source, self.entries = source, manifest[split]
        self.training, self.seed, self.epoch = split == 'train', seed, 0
        self.transform = make_transform(processor, self.training)

    def __len__(self):
        return len(self.entries)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __getitem__(self, index):
        from PIL import Image
        entry = self.entries[index]
        row = self.source[entry['row_index']]
        raw = image_bytes(row['image'])
        if int(row['label']) != entry['label'] or hashlib.sha256(raw).hexdigest() != entry['image_sha256']:
            raise ValueError(f'Source/manifest mismatch: {entry["sample_id"]}')
        with Image.open(io.BytesIO(raw)) as im:
            rgb = im.convert('RGB')
            if self.training:
                with image_rng(augmentation_seed(self.seed, self.epoch, entry['sample_id'])):
                    pixels = self.transform(rgb)
            else:
                pixels = self.transform(rgb)
        return {'pixel_values': pixels, 'labels': entry['label']}


class EpochSampler:
    """Duck-typed torch sampler; the Trainer adapter must call set_epoch each epoch."""
    def __init__(self, dataset, seed):
        self.dataset, self.seed, self.epoch = dataset, seed, 0

    def __len__(self):
        return len(self.dataset)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        self.dataset.set_epoch(epoch)

    def __iter__(self):
        import torch
        self.dataset.set_epoch(self.epoch)
        generator = torch.Generator().manual_seed(augmentation_seed(self.seed, self.epoch, 'sampler'))
        return iter(torch.randperm(len(self), generator=generator).tolist())


def collate_fn(examples):
    import torch
    return {'pixel_values': torch.stack([e['pixel_values'] for e in examples]),
            'labels': torch.tensor([e['labels'] for e in examples], dtype=torch.long)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--dataset-revision', help='Required only when creating a manifest')
    p.add_argument('--verify', action='store_true', help='Offline schema/hash/count checks; no image download')
    args = p.parse_args()
    if args.verify:
        manifest = read_manifest(args.manifest)
    else:
        if not args.dataset_revision:
            p.error('Creation requires --dataset-revision SHA')
        manifest = prepare_manifest(official_train(args.dataset_revision), args.dataset_revision, args.manifest)
    print(json.dumps({'manifest_hash':manifest['content_hash'], 'train':len(manifest['train']),
                      'val':len(manifest['val']), 'classes':len(manifest['label_names'])}))


if __name__ == '__main__':
    main()
