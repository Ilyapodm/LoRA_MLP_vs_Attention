"""CPU acceptance of real train/val images; no model training or test loader."""
import argparse
import json
from pathlib import Path
import re
import time


def main():
    import torch
    from transformers import AutoImageProcessor
    from configs import MODEL_ID
    from data import (read_manifest, official_train, ManifestDataset,
                      EpochSampler, collate_fn)

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--model-revision', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if not re.fullmatch('[0-9a-f]{40}', args.model_revision):
        p.error('Pass the full immutable model SHA')
    if args.output.exists():
        p.error('Output already exists; preserve the previous acceptance report')
    torch.set_num_threads(2)
    manifest = read_manifest(args.manifest)
    source = official_train(manifest['dataset_revision'])
    if source.features['label'].names != manifest['label_names']:
        raise AssertionError('Source/manifest class names or label order differ')
    processor = AutoImageProcessor.from_pretrained(
        MODEL_ID, revision=args.model_revision, use_fast=False)
    datasets = {split: ManifestDataset(source, manifest, split, processor, 42)
                for split in ('train', 'val')}
    report = {'stage': 'technical', 'scientific_result': False,
              'dataset_revision': manifest['dataset_revision'],
              'manifest_hash': manifest['content_hash'], 'model': MODEL_ID,
              'model_revision': args.model_revision,
              'image_mean': processor.image_mean, 'image_std': processor.image_std,
              'device': 'cpu', 'batch_size': 16, 'splits': {}}
    for split, dataset in datasets.items():
        start = time.perf_counter()
        count = 0
        for offset in range(0, len(dataset), 16):
            batch = collate_fn([dataset[i] for i in range(offset, min(offset+16, len(dataset)))])
            pixels = batch['pixel_values']
            n = len(batch['labels'])
            assert tuple(pixels.shape) == (n, 3, 224, 224)
            assert pixels.dtype == torch.float32 and torch.isfinite(pixels).all().item()
            expected = [e['label'] for e in manifest[split][offset:offset+n]]
            assert batch['labels'].tolist() == expected
            count += n
            if count % 1600 == 0:
                print(f'{split}: {count}/{len(dataset)} images checked', flush=True)
        report['splits'][split] = {'images_checked': count,
                                  'cpu_decode_transform_seconds': time.perf_counter()-start}
        print(f'{split}: all {count} images passed', flush=True)

    train, val = datasets['train'], datasets['val']
    # Same seed/epoch/sample must yield identical crops without sharing dropout RNG.
    for i in (0, 179, 180, len(train)-1):
        train.set_epoch(0)
        first = train[i]['pixel_values']
        train.set_epoch(1)
        _ = train[i]
        train.set_epoch(0)
        assert torch.equal(first, train[i]['pixel_values'])
    for i in (0, len(val)-1):
        val.set_epoch(0)
        first = val[i]['pixel_values']
        val.set_epoch(3)
        assert torch.equal(first, val[i]['pixel_values'])
    sampler = EpochSampler(train, 42)
    sampler.set_epoch(0)
    first_order = list(sampler)
    sampler.set_epoch(1)
    assert first_order != list(sampler)
    sampler.set_epoch(0)
    assert first_order == list(sampler)
    assert sorted(first_order) == list(range(len(train)))
    report['checks'] = ['all_selected_bytes_and_labels_match_manifest',
                        'all_images_decode_transform_collate_finite',
                        'class_names_and_order', 'repeatable_train_crops',
                        'validation_epoch_invariance', 'epoch_sampler_permutation']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(report, f, indent=2)
        f.write('\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
