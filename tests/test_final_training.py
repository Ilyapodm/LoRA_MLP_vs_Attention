"""Real pinned Trainer/PEFT on small CPU models, full counts and five final epochs."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


@unittest.skipUnless(all(importlib.util.find_spec(n) for n in ('torch', 'transformers', 'peft')),
                     'ML stack required')
class FinalTrainingTests(unittest.TestCase):
    def test_full_final_lp_peft_older_best_and_numerical_failure(self):
        import torch
        from peft import get_peft_model, LoraConfig
        from safetensors.torch import load_file
        from transformers import set_seed
        from final_train import execute_final_training, training_arguments
        from final_results import read_training_result
        from test_final import fixture
        torch.set_num_threads(1)

        class Dataset:
            def __init__(self, n):
                self.n, self.epoch = n, 0
                self.epochs_seen = set()

            def __len__(self):
                return self.n

            def set_epoch(self, epoch):
                self.epoch = epoch

            def __getitem__(self, i):
                self.epochs_seen.add(self.epoch)
                return {'pixel_values': torch.tensor([i%7/7, i%11/11, self.epoch/10]),
                        'labels': i%101}

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
                    self.classifier.bias[0] = 10  # Stable real ties retain epoch 1.

            def forward(self, pixel_values):
                logits = self.classifier(self.query(pixel_values))
                return {'logits': logits*float('nan') if self.invalid else logits}

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _, _, plan = fixture(root)
            for name, seed, invalid in (('LP', 42, False), ('B-iso', 44, False), ('LP', 43, True)):
                with self.subTest(config=name, seed=seed, invalid=invalid):
                    run = next(r for r in plan['runs'] if r['config'] == name and r['seed'] == seed)
                    set_seed(seed)
                    model = Model(invalid)
                    if name != 'LP':
                        model = get_peft_model(model, LoraConfig(r=2, lora_alpha=2,
                            lora_dropout=0.1, target_modules=['query'], modules_to_save=['classifier']))
                    output = root/f'{name}-{seed}'
                    settings = training_arguments(plan, run, output, use_cpu=True)
                    settings.disable_tqdm = True
                    settings.logging_strategy = 'no'
                    self.assertEqual((settings.seed, settings.data_seed), (seed, seed))
                    train, val = Dataset(18180), Dataset(2020)
                    status = execute_final_training(model, train, val, settings, output, plan, run,
                                                     {'fixture': 'CPU'})
                    result = read_training_result(output, plan, run)
                    self.assertEqual(status, 'numerical_invalid' if invalid else 'trained')
                    self.assertIsNone(result['metrics']['test_accuracy'])
                    self.assertFalse((output/'COMPLETE').exists())
                    if invalid:
                        self.assertFalse((output/'TRAINED').exists())
                        self.assertTrue((output/'TRAINING_FAILED').is_file())
                        continue
                    self.assertEqual(result['metrics']['best_epoch'], 1)
                    self.assertEqual(result['metrics']['applied_steps'], 180)
                    self.assertEqual(train.epochs_seen, {0, 1, 2, 3, 4})
                    self.assertTrue((output/'checkpoints/checkpoint-36').is_dir())
                    filename = 'model.safetensors' if name == 'LP' else 'adapter_model.safetensors'
                    checkpoint = load_file(str(output/'checkpoints/checkpoint-36'/filename))
                    exported = load_file(str(output/'best'/('head.safetensors' if name == 'LP' else filename)))
                    self.assertTrue(all(torch.equal(tensor, checkpoint[key]) for key, tensor in exported.items()))
                    metrics = result['metrics']
                    self.assertAlmostEqual(metrics['wall_train_job_seconds'],
                        metrics['validation_seconds']+metrics['checkpoint_seconds']+
                        metrics['train_loop_residual_seconds'])

    def test_wrong_settings_rejected_and_interrupt_preserves_diagnostics(self):
        import torch
        from final_train import execute_final_training, training_arguments
        from test_final import fixture
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _, _, plan = fixture(root)
            run = plan['runs'][0]
            output = root/'attempt'
            settings = training_arguments(plan, run, output, use_cpu=True)
            settings.seed = 142
            with self.assertRaisesRegex(ValueError, 'settings differ'):
                execute_final_training(torch.nn.Linear(3, 101), range(18180), range(2020),
                                       settings, output, plan, run, {})
            self.assertFalse(output.exists())
            settings.seed = run['seed']
            with patch('train.ImageTrainer.train', side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    execute_final_training(torch.nn.Linear(3, 101), range(18180), range(2020),
                                           settings, output, plan, run, {})
            self.assertEqual(json.loads((output/'status.json').read_text())['status'], 'interrupted')
            self.assertTrue((output/'history.json').exists())
            self.assertFalse((output/'TRAINED').exists())

    def test_cuda_final_amp_peft_and_update_audit(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('CUDA final AMP check requires an admitted GPU')
        from peft import get_peft_model, LoraConfig
        from transformers import set_seed
        from final_train import execute_final_training, training_arguments
        from final_results import read_training_result
        from test_final import fixture

        class Dataset:
            def __init__(self, n):
                self.n = n

            def __len__(self):
                return self.n

            def set_epoch(self, epoch):
                pass

            def __getitem__(self, i):
                return {'pixel_values': torch.tensor([i%7/7, i%11/11, 0.5]), 'labels': i%101}

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.query = torch.nn.Linear(3, 3)
                self.classifier = torch.nn.Linear(3, 101)

            def forward(self, pixel_values):
                return {'logits': self.classifier(self.query(pixel_values))}

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _, _, plan = fixture(root)
            run = next(r for r in plan['runs'] if r['config'] == 'B-iso' and r['seed'] == 44)
            set_seed(run['seed'])
            model = get_peft_model(Model(), LoraConfig(r=2, lora_alpha=2, lora_dropout=0.1,
                target_modules=['query'], modules_to_save=['classifier']))
            output = root/'cuda-final'
            settings = training_arguments(plan, run, output)
            settings.disable_tqdm = True
            settings.logging_strategy = 'no'
            self.assertEqual(execute_final_training(model, Dataset(18180), Dataset(2020),
                settings, output, plan, run, {'fixture': 'CUDA AMP'}), 'trained')
            metrics = read_training_result(output, plan, run)['metrics']
            self.assertEqual(metrics['attempted_steps'], 180)
            self.assertGreater(metrics['applied_steps'], 0)
            self.assertGreater(metrics['peak_allocated_gib'], 0)
            self.assertIsNone(metrics['test_accuracy'])

    def test_failed_cuda_context_still_writes_original_failure(self):
        import torch
        from unittest.mock import Mock
        from final_train import execute_final_training, training_arguments
        from test_final import fixture
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _, _, plan = fixture(root)
            run = plan['runs'][0]
            output = root/'cuda-error'
            settings = training_arguments(plan, run, output, use_cpu=True)
            failure = RuntimeError('original training failure')
            trainer = SimpleNamespace(
                state=SimpleNamespace(epoch=None, global_step=0, log_history=[]),
                validation_seconds=0, checkpoint_seconds=0, train=Mock(side_effect=failure))
            with patch('final_train.training_arguments', return_value=settings), \
                    patch.object(type(settings), 'device', property(lambda _: torch.device('cuda'))), \
                    patch('train.ImageTrainer', return_value=trainer), \
                    patch('torch.cuda.reset_peak_memory_stats'), \
                    patch('torch.cuda.synchronize', side_effect=[None, RuntimeError('context lost')]):
                with self.assertRaisesRegex(RuntimeError, 'original training failure'):
                    execute_final_training(torch.nn.Linear(3, 101), range(18180), range(2020),
                                           settings, output, plan, run, {})
            status = json.loads((output/'status.json').read_text())
            self.assertEqual(status['status'], 'technical_failed')
            self.assertIn('original training failure', status['reason'])
            self.assertTrue((output/'updates.json').is_file())
            self.assertFalse((output/'TRAINED').exists())


if __name__ == '__main__':
    unittest.main()
