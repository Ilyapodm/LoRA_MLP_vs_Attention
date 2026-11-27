"""Exercise the pinned Trainer/Accelerate stack, including an uneven final group."""
import importlib.util
from pathlib import Path
import tempfile
import unittest


@unittest.skipUnless(all(importlib.util.find_spec(n) for n in ('torch', 'transformers', 'accelerate')),
                     'Trainer stack not installed')
class TrainerTests(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(2)

    def make_run(self, output, batch, accumulation):
        import torch
        from transformers import TrainingArguments, set_seed
        from study_trainer import ImageTrainer
        set_seed(73)

        class Dataset:
            epoch = 0

            def set_epoch(self, epoch):
                self.epoch = epoch

            def __len__(self):
                return 14

            def __getitem__(self, i):
                return {'pixel_values': torch.tensor([i/14, (i % 3)/3, self.epoch/10]),
                        'labels': i % 3}

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.classifier = torch.nn.Linear(3, 3)

            def forward(self, pixel_values):
                return {'logits': self.classifier(pixel_values)}

        model = Model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        args = TrainingArguments(
            output_dir=str(output), use_cpu=True, num_train_epochs=1,
            per_device_train_batch_size=batch, gradient_accumulation_steps=accumulation,
            learning_rate=0.1, max_grad_norm=0, weight_decay=0, seed=73, data_seed=73,
            report_to='none', disable_tqdm=True, save_strategy='epoch', logging_strategy='no',
            remove_unused_columns=False, dataloader_num_workers=0, save_safetensors=True,
        )
        trainer = ImageTrainer(model=model, args=args, train_dataset=Dataset(),
                               optimizers=(optimizer, None))
        return trainer

    def test_accumulation_matches_large_batch_including_uneven_group(self):
        import torch
        with tempfile.TemporaryDirectory() as temp:
            large = self.make_run(Path(temp)/'large', 8, 1)
            small = self.make_run(Path(temp)/'small', 4, 2)
            initial = large.model.classifier.weight.detach().clone()
            large.train()
            small.train()
            self.assertEqual(large.state.global_step, 2)
            self.assertEqual(small.state.global_step, 2)
            for a, b in zip(large.model.parameters(), small.model.parameters()):
                torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)
            self.assertGreater(float((large.model.classifier.weight.detach()-initial).abs().max()), 1e-4)
            self.assertEqual(large.lr_scheduler.state_dict()['last_epoch'], 2)
            self.assertEqual(small.lr_scheduler.state_dict()['last_epoch'], 2)


    def test_cuda_amp_skip_preserves_weights_and_does_not_advance_scheduler(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('CUDA AMP check requires an admitted GPU')
        from transformers import TrainingArguments, TrainerCallback, set_seed
        from study_trainer import ImageTrainer, UpdateAudit
        set_seed(91)

        class Dataset:
            def set_epoch(self, epoch):
                pass

            def __len__(self):
                return 4

            def __getitem__(self, i):
                return {'pixel_values': torch.tensor([i/4, (i % 3)/3, 0.5]), 'labels': i % 3}

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.classifier = torch.nn.Linear(3, 3)

            def forward(self, pixel_values):
                return {'logits': self.classifier(pixel_values)}

        model = Model()
        before = model.classifier.weight.detach().clone()
        injected = False

        def first_gradient_overflows(gradient):
            nonlocal injected
            if not injected:
                injected = True
                return torch.full_like(gradient, float('inf'))
            return gradient

        class ObserveFirstStep(TrainerCallback):
            unchanged = False

            def on_step_end(self, args, state, control, model=None, **kwargs):
                if state.global_step == 1:
                    self.unchanged = torch.equal(model.classifier.weight.detach().cpu(), before)

        hook = model.classifier.weight.register_hook(first_gradient_overflows)
        audit, observer = UpdateAudit(), ObserveFirstStep()
        with tempfile.TemporaryDirectory() as temp:
            args = TrainingArguments(
                output_dir=temp, num_train_epochs=1, per_device_train_batch_size=2,
                fp16=True, learning_rate=1e-3, weight_decay=0, max_grad_norm=1,
                optim='adamw_torch',  # Same unfused optimizer as the actual pipeline.
                report_to='none', save_strategy='no', logging_strategy='no',
                disable_tqdm=True, remove_unused_columns=False, seed=91,
            )
            trainer = ImageTrainer(model=model, args=args, train_dataset=Dataset(), callbacks=[audit, observer])
            audit.trainer = trainer
            try:
                trainer.train()
            finally:
                hook.remove()
            self.assertTrue(observer.unchanged)
            self.assertEqual([e['skipped'] for e in audit.events], [True, False])
            self.assertEqual(trainer.state.global_step, 2)
            self.assertEqual(trainer.lr_scheduler.state_dict()['last_epoch'], 1)
            self.assertAlmostEqual(audit.events[0]['lr_after_update'][0], 1e-3)
            self.assertAlmostEqual(audit.events[1]['lr_after_update'][0], 5e-4)
            self.assertFalse(torch.equal(model.classifier.weight.detach().cpu(), before))


if __name__ == '__main__':
    unittest.main()
