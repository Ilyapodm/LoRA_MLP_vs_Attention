"""Build the small source-only bundle loaded by colab.ipynb; never include secrets/results."""
import hashlib
import json
from pathlib import Path
import zipfile

FILES = ('configs.py','bootstrap.py','test_params.py','requirements.txt',
         'README.md','tests/test_configs.py')


def main():
    root=Path(__file__).resolve().parent
    hashes={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in FILES}
    release=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    manifest={'files':hashes,'release_hash':release,'stage':'bootstrap'}
    output=root/'artifacts'/'colab-bootstrap.zip'
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name in FILES:
            z.write(root/name,name)
        z.writestr('SOURCE_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    print(output)
    print('release_hash:',release)


if __name__=='__main__':
    main()
