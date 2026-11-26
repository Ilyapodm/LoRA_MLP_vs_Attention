"""Build the small source-only bundle loaded by colab.ipynb; never include secrets/results."""
import hashlib
import json
from pathlib import Path
import zipfile

FILES = ('configs.py','bootstrap.py','test_params.py','check_data.py','check_resume.py','check_storage.py','checkpoints.py',
         'train.py','experiment.py','run_queue.py','package_protocol.py','requirements.txt','README.md','data.py','runs.json',
         'tests/test_configs.py','tests/test_data.py','tests/test_trainer.py','tests/test_checkpoints.py','tests/test_experiment.py','tests/test_lr_training.py')


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
