"""Shared protocol fixture for study plan, queue and Trainer integration tests."""
import json

from test_data import fixture as manifest_fixture


def protocol(root):
    root.mkdir()
    (root/'data_manifest.json').write_text(json.dumps(manifest_fixture()))
    (root/'dataset_revision.json').write_text(json.dumps({
        'dataset': 'ethz/food101', 'revision': 'a'*40}))
    (root/'model_revision.json').write_text(json.dumps({
        'model': 'google/vit-base-patch16-224-in21k', 'revision': 'b'*40}))
    return root
