"""Package the coordinator's common data manifest and immutable Hub revisions."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

from configs import MODEL_ID
from data import DATASET_ID, read_manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', type=Path, default=Path('artifacts/protocol'))
    p.add_argument('--output', type=Path, default=Path('artifacts/data-protocol.zip'))
    args = p.parse_args()
    manifest = read_manifest(args.protocol/'data_manifest.json')
    dataset = json.loads((args.protocol/'dataset_revision.json').read_text())
    model = json.loads((args.protocol/'model_revision.json').read_text())
    if dataset != {'dataset': DATASET_ID, 'revision': manifest['dataset_revision']}:
        raise ValueError('Dataset revision file and manifest differ')
    if model.get('model') != MODEL_ID or not re.fullmatch('[0-9a-f]{40}', model.get('revision', '')):
        raise ValueError('Expected the agreed model with an immutable commit SHA')
    names = ('data_manifest.json', 'dataset_revision.json', 'model_revision.json')
    payloads = {n: (args.protocol/n).read_bytes() for n in names}
    checksums = {n: hashlib.sha256(value).hexdigest() for n, value in payloads.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, mode='x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in payloads.items():
            archive.writestr(name, payload)
        archive.writestr('checksums.json', json.dumps(checksums, indent=2)+'\n')
    print(json.dumps({'archive': str(args.output), 'manifest_hash': manifest['content_hash'],
                      'archive_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
