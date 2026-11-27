"""Full v2 counts/epochs on CPU fixtures and actual ViT rank-16 assembly/gradients."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock


@unittest.skipUnless(all(importlib.util.find_spec(n) for n in ('torch', 'transformers', 'peft')),
                     'Pinned ML stack required')
class StudyTrainingTests(unittest.TestCase):
    def test_full_lr_and_final_360_updates_older_best_and_numerical_failure(self):
        import torch
        from peft import get_peft_model, LoraConfig
        from safetensors.torch import load_file
        from transformers import set_seed
        from study_protocol import make_final_plan
        from study_results import read_result
        from study_train import execute_training, training_arguments
        from test_study import fixture, lr_results
        torch.set_num_threads(1)

        class Dataset:
            def __init__(self, n): self.n, self.epoch, self.epochs_seen = n, 0, set()
            def __len__(self): return self.n
            def set_epoch(self, epoch): self.epoch = epoch
            def __getitem__(self, i):
                self.epochs_seen.add(self.epoch)
                return {'pixel_values': torch.tensor([i%7/7, i%11/11, self.epoch/10]), 'labels': i%101}

        class Model(torch.nn.Module):
            def __init__(self, invalid=False):
                super().__init__()
                self.query = torch.nn.Linear(3, 3)
                self.classifier = torch.nn.Linear(3, 101)
                self.invalid = invalid
                self.requires_grad_(False)
                self.classifier.requires_grad_(True)
                with torch.no_grad():
                    self.classifier.weight.zero_()
                    self.classifier.bias.zero_()
                    self.classifier.bias[0] = 10
            def forward(self, pixel_values):
                logits = self.classifier(self.query(pixel_values))
                return {'logits': logits*float('nan') if self.invalid else logits}

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _, lr_plan = fixture(root)
            lr_results(root/'lr-fixtures', lr_plan)
            final_plan = make_final_plan(lr_plan, [root/'lr-fixtures'])
            cases = [(lr_plan, 'LP', 142, False), (final_plan, 'B-16', 44, False),
                     (lr_plan, 'A-16', 142, True)]
            for plan, name, seed, invalid in cases:
                with self.subTest(stage=plan['stage'], name=name, invalid=invalid):
                    run = next(r for r in plan['runs'] if r['config']==name and r['seed']==seed)
                    set_seed(seed)
                    model = Model(invalid)
                    if name != 'LP':
                        model = get_peft_model(model, LoraConfig(r=2, lora_alpha=2,
                            lora_dropout=0.1, target_modules=['query'], modules_to_save=['classifier']))
                    output = root/(plan['stage']+'-'+name)
                    args = training_arguments(plan, run, output, use_cpu=True)
                    args.disable_tqdm = True
                    args.logging_strategy = 'no'
                    self.assertEqual((args.seed, args.data_seed, args.num_train_epochs), (seed, seed, 10))
                    train, val = Dataset(18180), Dataset(2020)
                    status = execute_training(model, train, val, args, output, plan, run,
                                              {'fixture': 'CPU integration'})
                    verified = read_result(output, plan, run)
                    expected = 'numerical_invalid' if invalid else 'valid' if plan['stage']=='lr_search' else 'trained'
                    self.assertEqual(status, expected)
                    self.assertIsNone(verified['metrics']['test_accuracy'])
                    self.assertFalse((output/'predictions.npz').exists())
                    if invalid:
                        self.assertTrue((output/'COMPLETE').is_file())
                        self.assertFalse((output/'best').exists())
                        continue
                    self.assertEqual(verified['metrics']['best_epoch'], 1)
                    self.assertEqual(verified['metrics']['global_step'], 360)
                    self.assertEqual(verified['metrics']['applied_steps'], 360)
                    self.assertEqual(train.epochs_seen, set(range(10)))
                    self.assertTrue((output/'checkpoints/checkpoint-36').is_dir())
                    self.assertTrue((output/'checkpoints/checkpoint-360').is_dir())
                    filename = 'model.safetensors' if name=='LP' else 'adapter_model.safetensors'
                    best = load_file(str(output/'checkpoints/checkpoint-36'/filename))
                    export = load_file(str(output/'best'/('head.safetensors' if name=='LP' else filename)))
                    self.assertTrue(all(torch.equal(t, best[k]) for k, t in export.items()))
                    self.assertTrue((output/('COMPLETE' if plan['stage']=='lr_search' else 'TRAINED')).exists())

    def test_new_rank16_real_vit_parameters_gradients_and_freeze(self):
        import torch
        from study_check import parameter_check
        torch.set_num_threads(2)
        a = parameter_check('A-16', 'cpu', image_size=32)
        b = parameter_check('B-16', 'cpu', image_size=32)
        self.assertEqual((a['trainable'], b['trainable']), (1257317, 2731877))
        self.assertEqual(a['head_hash'], b['head_hash'])
        for report in (a, b):
            self.assertEqual(report['applied_steps'], 3)
            self.assertIn('A_and_B_gradients', report['checks'])
            self.assertIn('frozen_unchanged', report['checks'])

    def test_wrong_seed_partial_data_and_interrupt_are_not_scientific_results(self):
        import torch
        from study_train import execute_training, training_arguments
        from study_protocol import make_final_plan
        from test_study import fixture, lr_results
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _, lr = fixture(root)
            lr_results(root/'lr', lr)
            plan = make_final_plan(lr, [root/'lr'])
            run = plan['runs'][0]
            output = root/'interrupted'
            args = training_arguments(plan, run, output, use_cpu=True)
            args.seed = 142
            with self.assertRaisesRegex(ValueError, 'Trainer settings'):
                execute_training(torch.nn.Linear(3, 101), range(18180), range(2020),
                                 args, output, plan, run, {})
            args.seed = run['seed']
            with self.assertRaisesRegex(ValueError, 'full 18180'):
                execute_training(torch.nn.Linear(3, 101), range(772), range(32),
                                 args, output, plan, run, {})
            self.assertFalse(output.exists())
            with patch('train.ImageTrainer.train', side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    execute_training(torch.nn.Linear(3, 101), range(18180), range(2020),
                                     args, output, plan, run, {})
            self.assertEqual(json.loads((output/'status.json').read_text())['status'], 'interrupted')
            self.assertTrue((output/'updates.json').exists())
            self.assertFalse((output/'TRAINED').exists())

    def test_cuda_error_preserves_diagnostics_and_overflow_json_is_standard(self):
        import torch
        from study_train import execute_training, training_arguments, clean_json
        from test_study import fixture
        self.assertEqual(clean_json({'loss': float('nan'), 'gradient': [float('inf')]}),
                         {'loss': None, 'gradient': [None]})
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _, plan = fixture(root)
            run = plan['runs'][0]
            output = root/'error'
            args = training_arguments(plan, run, output, use_cpu=True)
            trainer = SimpleNamespace(state=SimpleNamespace(epoch=None, global_step=0, log_history=[]),
                validation_seconds=0, checkpoint_seconds=0, train=Mock(side_effect=RuntimeError('original failure')))
            with patch('study_train.training_arguments', return_value=args), \
                    patch.object(type(args), 'device', property(lambda _: torch.device('cuda'))), \
                    patch('train.ImageTrainer', return_value=trainer), \
                    patch('torch.cuda.reset_peak_memory_stats'), \
                    patch('torch.cuda.synchronize', side_effect=[None, RuntimeError('context lost')]):
                with self.assertRaisesRegex(RuntimeError, 'original failure'):
                    execute_training(torch.nn.Linear(3, 101), range(18180), range(2020),
                                     args, output, plan, run, {})
            self.assertIn('original failure', json.loads((output/'status.json').read_text())['reason'])
            self.assertTrue((output/'history.json').exists())


if __name__ == '__main__': unittest.main()
