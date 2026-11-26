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

    def make_run(self, output, batch, accumulation, epochs=1, callback=None, dropout=0):
        import torch
        from transformers import TrainingArguments, set_seed
        from train import ImageTrainer
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
                self.dropout = torch.nn.Dropout(dropout)
                self.classifier = torch.nn.Linear(3, 3)

            def forward(self, pixel_values):
                return {'logits': self.classifier(self.dropout(pixel_values))}

        model = Model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        args = TrainingArguments(
            output_dir=str(output), use_cpu=True, num_train_epochs=epochs,
            per_device_train_batch_size=batch, gradient_accumulation_steps=accumulation,
            learning_rate=0.1, max_grad_norm=0, weight_decay=0, seed=73, data_seed=73,
            report_to='none', disable_tqdm=True, save_strategy='epoch', logging_strategy='no',
            remove_unused_columns=False, dataloader_num_workers=0, save_safetensors=True,
        )
        trainer = ImageTrainer(model=model, args=args, train_dataset=Dataset(),
                               optimizers=(optimizer, None), callbacks=[callback] if callback else [])
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

    def test_epoch_resume_matches_continuous_with_dropout_and_sampler(self):
        import torch
        from transformers import TrainerCallback

        class StopAfterEpoch(TrainerCallback):
            def on_epoch_end(self, args, state, control, **kwargs):
                control.should_training_stop = True

        with tempfile.TemporaryDirectory() as temp:
            full = self.make_run(Path(temp)/'full', 4, 2, epochs=2, dropout=0.2)
            full.train()
            partial = self.make_run(Path(temp)/'part', 4, 2, epochs=2,
                                    dropout=0.2, callback=StopAfterEpoch())
            partial.train()
            checkpoint = Path(temp)/'part'/'checkpoint-2'
            for name in ('optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'trainer_state.json'):
                self.assertTrue((checkpoint/name).is_file(), name)
            resumed = self.make_run(Path(temp)/'resumed', 4, 2, epochs=2, dropout=0.2)
            resumed.train(resume_from_checkpoint=str(checkpoint))
            self.assertEqual(resumed.state.global_step, 4)
            self.assertEqual(resumed.state.epoch, 2)
            self.assertEqual(full.lr_scheduler.state_dict(), resumed.lr_scheduler.state_dict())
            for a, b in zip(full.model.parameters(), resumed.model.parameters()):
                torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)

    def test_vit_lp_and_peft_bundles_resume_with_older_best(self):
        import torch
        from transformers import ViTConfig, ViTForImageClassification, TrainingArguments, set_seed
        from peft import LoraConfig, get_peft_model
        from train import ImageTrainer, UpdateAudit, StopAfterEpoch
        from checkpoints import restore_bundle, verify_bundle
        from check_resume import compare_weights

        class Dataset:
            epoch = 0

            def set_epoch(self, epoch):
                self.epoch = epoch

            def __len__(self):
                return 4

            def __getitem__(self, i):
                return {'pixel_values': torch.rand(3, 16, 16, generator=torch.Generator().manual_seed(i+self.epoch*100)),
                        'labels': i % 3}

        for config in ('LP', 'B-iso'):
            with self.subTest(config=config), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                identity = {'config': config, 'fixture': 'tiny-ViT-older-best'}

                def run(name, stop=None, bundle=None, storage=False):
                    set_seed(51)
                    model = ViTForImageClassification(ViTConfig(
                        image_size=16, patch_size=8, hidden_size=16, intermediate_size=32,
                        num_hidden_layers=1, num_attention_heads=2, num_labels=3,
                        hidden_dropout_prob=0.1, attn_implementation='eager'))
                    model.requires_grad_(False)
                    model.classifier.requires_grad_(True)
                    if config != 'LP':
                        model = get_peft_model(model, LoraConfig(r=2, lora_alpha=2, lora_dropout=0.1,
                            target_modules=['query', 'value'], modules_to_save=['classifier']))
                    out = root/name
                    out.mkdir()
                    audit = UpdateAudit()
                    checkpoint = None
                    if bundle:
                        checkpoint, metadata = restore_bundle(bundle, out/'restored', identity)
                        audit.events = metadata['audit']['events']
                        audit.consecutive_skips = metadata['audit']['consecutive_skips']
                    settings = TrainingArguments(
                        output_dir=str(out), use_cpu=True, num_train_epochs=3,
                        per_device_train_batch_size=2, per_device_eval_batch_size=2,
                        optim='adamw_torch', learning_rate=1e-3, seed=51, data_seed=51,
                        save_strategy='epoch', eval_strategy='epoch', save_total_limit=2,
                        load_best_model_at_end=True, metric_for_best_model='correct_count', greater_is_better=True,
                        report_to='none', disable_tqdm=True, logging_strategy='no',
                        remove_unused_columns=False, label_names=['labels'])
                    trainer = ImageTrainer(model=model, args=settings, train_dataset=Dataset(), eval_dataset=Dataset(),
                        compute_metrics=lambda p: {'correct_count': 0},
                        callbacks=[audit, StopAfterEpoch(stop)], run_identity=identity, bundle_root=out/'bundles',
                        storage_root=root/'failed-storage' if storage else None)
                    trainer.update_audit = audit
                    audit.trainer = trainer
                    trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
                    return trainer

                full = run('continuous')
                part = run('part', stop=2)
                bundle = part.last_bundle
                verify_bundle(bundle, identity)
                # Relocate the whole original run before resuming.
                import shutil
                shutil.move(str(root/'part'), str(root/'moved'))
                bundle = root/'moved'/'bundles'/bundle.name
                resumed = run('resumed', bundle=bundle)
                self.assertEqual(resumed.state.global_step, 6)
                self.assertEqual(resumed.state.best_global_step, 2)
                filename = 'model.safetensors' if config == 'LP' else 'adapter_model.safetensors'
                for folder in ('resume', 'best_model'):
                    self.assertEqual(compare_weights(full.last_bundle/folder/filename,
                                                     resumed.last_bundle/folder/filename), 0)
                self.assertEqual(full.update_audit.events, resumed.update_audit.events)
                for first, second in zip(full.model.parameters(), resumed.model.parameters()):
                    torch.testing.assert_close(first, second, rtol=0, atol=0)
                from unittest.mock import patch
                from checkpoints import StorageSyncError
                with patch('train.publish_bundle', side_effect=StorageSyncError('simulated exhausted retries')):
                    with self.assertRaises(StorageSyncError):
                        run('sync-failure', storage=True)
                self.assertTrue((root/'sync-failure'/'checkpoint-2').is_dir())
                self.assertFalse((root/'sync-failure'/'checkpoint-4').exists())

    def test_cuda_amp_skip_preserves_weights_and_does_not_advance_scheduler(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('CUDA AMP check requires an admitted GPU')
        from transformers import TrainingArguments, TrainerCallback, set_seed
        from train import ImageTrainer, UpdateAudit
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
