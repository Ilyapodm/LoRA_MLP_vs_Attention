"""Manifest checks without network; tensor checks run when the ML stack is installed."""
from copy import deepcopy
import importlib.util
import unittest

from data import (DATASET_ID, SELECTION_VERSION, canonical_hash, sample_id,
                  validate_manifest, augmentation_seed)


def fixture():
    revision = 'a' * 40
    m = {'schema_version':1, 'dataset_id':DATASET_ID, 'dataset_revision':revision,
         'selection_version':SELECTION_VERSION, 'split_seed':42,
         'label_names':[f'class_{i}' for i in range(101)], 'train':[], 'val':[]}
    for label in range(101):
        for j in range(200):
            i=label*750+j
            m['train' if j<180 else 'val'].append({
                'row_index':i, 'sample_id':sample_id(revision,i), 'label':label,
                'image_sha256':f'{i:064x}'})
    m['content_hash']=canonical_hash(m)
    return m


def resign(m):
    m['content_hash']=canonical_hash({k:v for k,v in m.items() if k!='content_hash'})


class ManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.valid=fixture()

    def test_balanced_manifest(self):
        self.assertEqual(len(validate_manifest(self.valid)['train']),18180)
        self.assertEqual(len(self.valid['val']),2020)

    def test_tampering(self):
        m=deepcopy(self.valid)
        m['train'][0]['label']=1
        with self.assertRaisesRegex(ValueError,'hash mismatch'):
            validate_manifest(m)

    def test_overlap_even_with_valid_hash(self):
        m=deepcopy(self.valid)
        m['val'][0]=deepcopy(m['train'][0])
        resign(m)
        with self.assertRaisesRegex(ValueError,'overlap'):
            validate_manifest(m)

    def test_unbalanced_counts_even_with_valid_hash(self):
        m=deepcopy(self.valid)
        m['train'][0]['label']=1
        resign(m)
        with self.assertRaisesRegex(ValueError,'counts'):
            validate_manifest(m)

    def test_randomness_key(self):
        a=augmentation_seed(42,0,'sample')
        self.assertEqual(a,augmentation_seed(42,0,'sample'))
        self.assertEqual(len({a,augmentation_seed(43,0,'sample'),
                              augmentation_seed(42,1,'sample'),
                              augmentation_seed(42,0,'other')}),4)


@unittest.skipUnless(importlib.util.find_spec('numpy'), 'NumPy not installed')
class SelectionTests(unittest.TestCase):
    def test_pcg64_selection(self):
        from data import select_indices
        labels=[i for i in range(101) for _ in range(750)]
        train,val=select_indices(labels)
        self.assertEqual((len(train),len(val)),(18180,2020))
        self.assertFalse(set(train)&set(val))
        self.assertEqual((train,val),select_indices(labels))
        for label in range(101):
            self.assertEqual(sum(labels[i]==label for i in train),180)
            self.assertEqual(sum(labels[i]==label for i in val),20)


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('numpy'), 'ML stack not installed')
class RNGTests(unittest.TestCase):
    def test_transform_context_restores_all_randomness_without_cuda_seed(self):
        import random
        import numpy as np
        import torch
        from unittest.mock import patch
        from data import image_rng
        random.seed(3)
        np.random.seed(4)
        torch.default_generator.manual_seed(5)
        py_state,np_state,torch_state=random.getstate(),np.random.get_state(),torch.get_rng_state().clone()
        with patch.object(torch.cuda,'manual_seed_all',side_effect=AssertionError('CUDA RNG touched')):
            with image_rng(123):
                values=(random.random(),float(np.random.random()),torch.rand(3))
            with image_rng(123):
                self.assertEqual(values[0],random.random())
                self.assertEqual(values[1],float(np.random.random()))
                self.assertTrue(torch.equal(values[2],torch.rand(3)))
        self.assertEqual(py_state,random.getstate())
        self.assertTrue(np.array_equal(np_state[1],np.random.get_state()[1]))
        self.assertTrue(torch.equal(torch_state,torch.get_rng_state()))


@unittest.skipUnless(all(importlib.util.find_spec(x) for x in ('torch','torchvision','numpy','PIL')), 'ML stack not installed')
class TransformTests(unittest.TestCase):
    def test_images_epoch_sampler_and_collator(self):
        import hashlib
        import io
        from types import SimpleNamespace
        import numpy as np
        from PIL import Image
        import torch
        from data import ManifestDataset, EpochSampler, collate_fn
        image=np.random.default_rng(5).integers(0,256,(96,128,3),dtype=np.uint8)
        buf=io.BytesIO()
        Image.fromarray(image).save(buf,format='PNG')
        raw=buf.getvalue()
        m=fixture()
        for split in ('train','val'):
            for e in m[split]:
                e['image_sha256']=hashlib.sha256(raw).hexdigest()
        resign(m)
        class Source:
            def __getitem__(self,index):
                return {'label':index//750,'image':{'bytes':raw,'path':None}}
        processor=SimpleNamespace(image_mean=[0.5]*3,image_std=[0.5]*3)
        train=ManifestDataset(Source(),m,'train',processor,42)
        val=ManifestDataset(Source(),m,'val',processor,42)
        first=train[0]['pixel_values']
        self.assertTrue(torch.equal(first,train[0]['pixel_values']))
        train.set_epoch(1)
        self.assertFalse(torch.equal(first,train[0]['pixel_values']))
        train.set_epoch(0)
        self.assertTrue(torch.equal(first,train[0]['pixel_values']))
        val_before=val[0]['pixel_values']
        val.set_epoch(7)
        self.assertTrue(torch.equal(val_before,val[0]['pixel_values']))
        sampler=EpochSampler(train,42)
        sampler.set_epoch(3)
        order=list(sampler)
        sampler.set_epoch(4)
        self.assertNotEqual(order,list(sampler))
        sampler.set_epoch(3)
        self.assertEqual(order,list(sampler))
        self.assertEqual(set(order),set(range(len(train))))
        batch=collate_fn([train[0],train[1]])
        self.assertEqual(tuple(batch['pixel_values'].shape),(2,3,224,224))
        self.assertEqual(batch['labels'].dtype,torch.long)
        with self.assertRaisesRegex(ValueError,'test'):
            ManifestDataset(Source(),m,'test',processor,42)


if __name__=='__main__':
    unittest.main()
