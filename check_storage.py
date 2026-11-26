"""Publish actual acceptance bundles to a mounted store and restore verified copies."""
import argparse
import json
from pathlib import Path
import uuid

from checkpoints import publish_bundle, restore_bundle, verify_bundle, file_hash, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--resume-report', type=Path, required=True)
    p.add_argument('--storage-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if not args.storage_root.is_dir():
        p.error('Mount/create and check the persistent directory first')
    report = json.loads(args.resume_report.read_text())
    if report.get('stage') != 'technical' or not report.get('runs'):
        p.error('Expected a completed technical resume acceptance report')
    args.output.mkdir(parents=True, exist_ok=False)
    storage = args.storage_root/('checkpoint-acceptance-'+uuid.uuid4().hex[:12])
    results = []
    for run in report['runs']:
        source = Path(run['resumed_bundle'])
        metadata = verify_bundle(source, run['identity'])
        print(run['config']+': publish to '+str(storage), flush=True)
        target = publish_bundle(source, storage/run['config'])
        verify_bundle(target, run['identity'])
        restored, _ = restore_bundle(target, args.output/run['config'], run['identity'])
        # trainer_state path is intentionally rewritten; all weight/optimizer bytes stay identical.
        sums = json.loads((source/'checksums.json').read_text())['files']
        for name, expected in sums.items():
            if name != 'resume/trainer_state.json':
                if file_hash(restored.parent/name) != expected['sha256']:
                    raise ValueError('Restored content differs: '+name)
        results.append({'config': run['config'], 'global_step': metadata['global_step'],
                        'published_bundle': str(target), 'restored_checkpoint': str(restored),
                        'checksums_sha256': file_hash(target/'checksums.json'),
                        'mounted_filesystem_readback_verified': True})
    write_json(args.output/'storage-acceptance.json', {
        'stage': 'technical', 'scientific_result': False, 'runs': results,
        'note': 'Verified through the mounted filesystem; independent provider-side visibility is a separate check'})
    print('STORAGE_ACCEPTANCE_FINISHED', args.output/'storage-acceptance.json', flush=True)


if __name__ == '__main__':
    main()
