"""Failure injection for bundles, independent of CUDA or model dependencies."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from checkpoints import (seal_bundle, verify_bundle, publish_bundle, restore_bundle,
                         prune_bundles, StorageSyncError, file_hash)


def fixture(root, step=2):
    root.mkdir()
    (root/'resume').mkdir()
    (root/'best_model').mkdir()
    (root/'best').mkdir()
    identity = {'config': 'LP', 'code_hash': 'same-code', 'manifest_hash': 'same-data'}
    for name in ('optimizer.pt', 'scheduler.pt', 'rng_state.pth'):
        (root/'resume'/name).write_bytes(b'fixture state')
    for folder in ('resume', 'best_model'):
        (root/folder/'model.safetensors').write_bytes(b'fixture model')
        (root/folder/'config.json').write_text('{}')
    (root/'best'/'head.safetensors').write_bytes(b'fixture head')
    (root/'best'/'metadata.json').write_text('{}')
    (root/'resume'/'trainer_state.json').write_text(json.dumps({
        'global_step': step, 'epoch': step/2, 'best_model_checkpoint': 'best_model'}))
    (root/'metadata.json').write_text(json.dumps({'schema_version': 1, 'identity': identity,
                                                  'global_step': step, 'epoch': step/2, 'amp': False}))
    seal_bundle(root)
    return identity


class BundleTests(unittest.TestCase):
    def test_tamper_missing_marker_and_identity_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)/'bundle'
            identity = fixture(root)
            other = deepcopy(identity);other['manifest_hash'] = 'different-data'
            with self.assertRaisesRegex(ValueError, 'identity differs'):
                restore_bundle(root, Path(d)/'restore', other)
            self.assertFalse((Path(d)/'restore').exists())
            marker = (root/'READY').read_bytes()
            (root/'READY').unlink()
            with self.assertRaises(OSError):
                verify_bundle(root)
            (root/'READY').write_bytes(marker)
            (root/'resume'/'optimizer.pt').write_bytes(b'truncated')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                verify_bundle(root)

    def test_publish_retries_and_marks_only_verified_copy(self):
        with tempfile.TemporaryDirectory() as d:
            root, storage = Path(d)/'bundle', Path(d)/'storage'
            identity = fixture(root)
            failed, delays = False, []

            def copy(source, target):
                nonlocal failed
                if not failed:
                    failed = True
                    Path(target).write_bytes(b'partial')
                    raise OSError('simulated dropped connection')
                return shutil.copy2(source, target)

            target = publish_bundle(root, storage, copy_file=copy, wait=delays.append)
            verify_bundle(target, identity)
            self.assertEqual(delays, [5])
            self.assertEqual(len(list(storage.glob('*/READY'))), 1)
            self.assertEqual(len(list(storage.iterdir())), 2)

    def test_silent_copy_corruption_and_repeated_failure_preserve_previous(self):
        with tempfile.TemporaryDirectory() as d:
            root, storage = Path(d)/'bundle', Path(d)/'storage'
            identity = fixture(root)
            previous = publish_bundle(root, storage)
            before = file_hash(previous/'READY')
            delays = []

            def corrupt(source, target):
                shutil.copy2(source, target)
                if Path(source).name == 'optimizer.pt':
                    Path(target).write_bytes(b'corrupt')

            with self.assertRaises(StorageSyncError):
                publish_bundle(root, storage, copy_file=corrupt, wait=delays.append)
            self.assertEqual(delays, [5, 15])
            self.assertEqual(len(list(storage.glob('*/READY'))), 1)
            self.assertEqual(file_hash(previous/'READY'), before)
            verify_bundle(previous, identity)

    def test_relocation_only_changes_working_copy_and_pruning_keeps_two(self):
        with tempfile.TemporaryDirectory() as d:
            storage = Path(d)/'storage';storage.mkdir()
            for step in (2, 4, 6):
                identity = fixture(storage/f'epoch-{step}', step)
            partial = storage/'partial';partial.mkdir()
            (partial/'unfinished').write_text('unconfirmed')
            newest = storage/'epoch-6'
            source_hash = file_hash(newest/'resume'/'trainer_state.json')
            restored, metadata = restore_bundle(newest, Path(d)/'new-machine', identity)
            state = json.loads((restored/'trainer_state.json').read_text())
            self.assertEqual(state['best_model_checkpoint'], str((Path(d)/'new-machine'/'best_model').resolve()))
            self.assertEqual(file_hash(newest/'resume'/'trainer_state.json'), source_hash)
            verify_bundle(newest, identity)
            self.assertFalse((restored.parent/'READY').exists())
            prune_bundles(storage, identity)
            self.assertFalse((storage/'epoch-2').exists())
            self.assertTrue((storage/'epoch-4'/'READY').exists())
            self.assertTrue((storage/'epoch-6'/'READY').exists())
            self.assertTrue(partial.exists())


if __name__ == '__main__':
    unittest.main()
