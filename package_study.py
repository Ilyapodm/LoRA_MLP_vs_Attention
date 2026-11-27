"""Package the accepted v2 computational release and LR plan, without data/results/secrets."""
import argparse
import json
from pathlib import Path
import zipfile

from checkpoints import file_hash
from data import canonical_hash
from package_bootstrap import FILES as BOOTSTRAP_FILES
from study_protocol import SOURCE_FILES, read_plan


def main():
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, default=root/'runs-v2.json')
    p.add_argument('--protocol', type=Path, default=root/'artifacts/protocol')
    p.add_argument('--output', type=Path, default=root/'artifacts/study-v2-source.zip')
    args = p.parse_args()
    plan = read_plan(args.plan, args.protocol)
    if plan['stage'] != 'lr_search':
        p.error('Source bundle requires the v2 LR plan')
    if args.plan.resolve().parent != root:
        p.error('Keep the shared LR plan in the project root before packaging')
    files = set(BOOTSTRAP_FILES) | set(SOURCE_FILES) | {
        '.gitattributes', '.gitignore', 'package_study.py', args.plan.name,
        'final_protocol.py', 'final_train.py', 'final_results.py', 'final_queue.py',
        'tests/test_final.py', 'tests/test_final_training.py',
        'tests/test_study.py', 'tests/test_study_training.py'}
    files.update(path.relative_to(root).as_posix() for path in (root/'lora_docs_updated').rglob('*.md'))
    hashes = {name: file_hash(root/name) for name in sorted(files)}
    manifest = {'schema_version': 2, 'stage': 'source-release', 'protocol': 'food101-local-v2',
                'plan_hash': plan['plan_hash'], 'source_hash': plan['common']['source_hash'],
                'files': hashes, 'release_hash': canonical_hash(hashes)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, mode='x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(files):
            archive.write(root/name, name)
        archive.writestr('SOURCE_MANIFEST.json', json.dumps(manifest, indent=2)+'\n')
    print(args.output.resolve())
    print('SHA256', file_hash(args.output))


if __name__ == '__main__':
    main()
