"""Final provenance, seed blocks, immutable artifacts and whole-run failure policy."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from checkpoints import write_json
from data import canonical_hash
from experiment import collect_results, make_plan, read_plan
from final_protocol import (config_record, make_final_plan, make_freeze, read_final_plan,
                            validate_final_plan, write_plan_files, ORDERS)
from final_queue import execute_final_queue
from final_results import seal_training, read_training_result, collect_training_results
from test_experiment import protocol, result as lr_result


def fixture(root):
    proto = protocol(root/'protocol')
    lr_plan = make_plan(proto)
    for i, run in enumerate(lr_plan['runs']):
        lr_result(root/'lr'/str(i), lr_plan, run)
    return proto, lr_plan, make_final_plan(lr_plan, [root/'lr'])


def trained_result(root, plan, run, invalid=False):
    root.mkdir(parents=True)
    config = config_record(plan, run)
    history = [{'epoch': epoch, 'step': epoch*36, 'eval_correct_count': 1000,
                'eval_sample_count': 2020, 'eval_accuracy': 1000/2020}
               for epoch in range(1, 6)]
    updates = [{'step': i, 'skipped': False,
                'lr_after_update': [run['lr']*(1-i/180)]} for i in range(1, 181)]
    metrics = {'test_accuracy': None, 'test_correct_count': None, 'test_sample_count': None,
               'epochs_completed': 0 if invalid else 5, 'global_step': 0 if invalid else 180,
               'attempted_steps': 0 if invalid else 180,
               'applied_steps': 0 if invalid else 180, 'skipped_steps': 0,
               'val_correct_count': None if invalid else 1000,
               'val_sample_count': None if invalid else 2020,
               'val_accuracy': None if invalid else 1000/2020, 'best_epoch': None if invalid else 1}
    status = {'schema_version': 1, 'stage': 'final', 'action': 'train',
              'status': 'numerical_invalid' if invalid else 'trained',
              'reason': 'Nonfinite logits' if invalid else None}
    for name, value in {'config.json': config, 'environment.json': {}, 'status.json': status,
                        'training_status.json': status, 'metrics.json': metrics,
                        'history.json': [] if invalid else history,
                        'updates.json': [] if invalid else updates}.items():
        write_json(root/name, value)
    if not invalid:
        best = root/'best'
        best.mkdir()
        write_json(best/'metadata.json', {**config, 'best_epoch': 1,
                                         'val_correct_count': 1000, 'val_sample_count': 2020})
        (best/('head.safetensors' if run['config'] == 'LP' else 'adapter_model.safetensors')).write_bytes(b'fixture')
        if run['config'] != 'LP':
            write_json(best/'adapter_config.json', {})
    seal_training(root, plan, run)


class FinalPlanTests(unittest.TestCase):
    def test_verified_selection_seed_blocks_portability_and_freeze(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            proto, lr, plan = fixture(root)
            for index, executor in enumerate(('rtx4070ti', 'rtx3090', 'rtx3060')):
                runs = [r for r in plan['runs'] if r['executor'] == executor]
                self.assertEqual([r['config'] for r in runs], list(ORDERS[index]))
                self.assertEqual({r['seed'] for r in runs}, {42+index})
            self.assertEqual(len({r['run_id'] for r in plan['runs']}), 12)
            self.assertNotIn('seed', plan['common'])
            self.assertTrue(all(r['lr'] == 1e-4 for r in plan['runs']))
            selection = collect_results(lr, [root/'lr'])
            for trial in selection['trials']:
                trial['path'] = '/different/computer'
            self.assertEqual(make_final_plan(lr, [root/'lr'], selection), plan)
            shutil.move(root/'lr', root/'moved-lr')
            self.assertEqual(make_final_plan(lr, [root/'moved-lr']), plan)
            write_plan_files(plan, root/'final.json', proto/'freeze.json')
            self.assertEqual(read_final_plan(root/'final.json', proto, proto/'freeze.json'), plan)
            with self.assertRaises(FileExistsError):
                write_plan_files(plan, root/'final.json', proto/'freeze.json')
            with self.assertRaises(FileNotFoundError):
                read_final_plan(root/'final.json', proto, proto/'absent.json')
            freeze = make_freeze(plan)
            freeze['selected_lrs']['LP'] = 0.01
            (proto/'freeze.json').write_text(json.dumps(freeze))
            with self.assertRaisesRegex(ValueError, 'freeze'):
                read_final_plan(root/'final.json', proto, proto/'freeze.json')

    def test_missing_corrupt_or_incorrect_lr_selection_blocks_final(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _, lr, _ = fixture(root)
            selection = collect_results(lr, [root/'lr'])
            selection['selected']['LP']['lr'] = 0.01
            with self.assertRaisesRegex(ValueError, 'selection differs'):
                make_final_plan(lr, [root/'lr'], selection)
            (root/'lr/0/metrics.json').write_text('{}')
            with self.assertRaises(ValueError):
                make_final_plan(lr, [root/'lr'])
            shutil.rmtree(root/'lr/0')
            with self.assertRaisesRegex(ValueError, 'all 12'):
                make_final_plan(lr, [root/'lr'])

    def test_semantic_plan_changes_rejected_even_with_recomputed_hash(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _, _, plan = fixture(root)
            for change in ('seed', 'lr', 'order', 'epochs', 'selection'):
                bad = deepcopy(plan)
                if change in ('seed', 'lr'):
                    bad['runs'][0][change] = 142 if change == 'seed' else 0.01
                elif change == 'order':
                    bad['runs'][0], bad['runs'][1] = bad['runs'][1], bad['runs'][0]
                elif change == 'epochs':
                    bad['common']['epochs'] = 10
                else:
                    bad['selection']['selected']['LP']['lr'] = 0.01
                bad['plan_hash'] = canonical_hash({k: v for k, v in bad.items() if k != 'plan_hash'})
                with self.subTest(change=change), self.assertRaises(ValueError):
                    validate_final_plan(bad)
            proto = root/'protocol'
            write_plan_files(plan, root/'final.json', proto/'freeze.json')
            changed = dict(plan['source_files'], **{'final_train.py': '0'*64})
            with patch('final_protocol.source_manifest', return_value=changed):
                with self.assertRaisesRegex(ValueError, 'code differs'):
                    read_final_plan(root/'final.json', proto, proto/'freeze.json')

    def test_coordinator_cli_and_dry_run_need_no_ml_or_gpu(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            proto, lr, _ = fixture(root)
            write_json(root/'lr-plan.json', lr)
            script = Path(__file__).resolve().parents[1]/'final_queue.py'
            command = [sys.executable, '-B', str(script), '--plan', str(root/'final.json'),
                       '--protocol', str(proto)]
            created = subprocess.run(command+['--make-plan', '--lr-plan', str(root/'lr-plan.json'),
                '--results', str(root/'lr')], capture_output=True, text=True, check=True)
            self.assertIn('Created', created.stdout)
            dry = subprocess.run(command+['--dry-run', '--executor', 'rtx3090'],
                                 capture_output=True, text=True, check=True)
            runs = [json.loads(line) for line in dry.stdout.splitlines()]
            self.assertEqual([r['config'] for r in runs], list(ORDERS[1]))
            self.assertEqual({r['seed'] for r in runs}, {43})


class FinalQueueTests(unittest.TestCase):
    def test_skip_trained_stop_then_retry_same_task_without_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            proto, _, plan = fixture(root)
            calls = []

            def launch(command, log):
                identifier = command[command.index('--run-id')+1]
                run = next(r for r in plan['runs'] if r['run_id'] == identifier)
                target = Path(command[command.index('--output')+1])
                calls.append(run)
                if len(calls) == 2:
                    return 1
                trained_result(target, plan, run)
                return 0

            args = (plan, 'final.json', 'freeze.json', proto, 'rtx4070ti', root/'executor', 'admission')
            with self.assertRaises(RuntimeError):
                execute_final_queue(*args, launch=launch)
            with self.assertRaises(ValueError):
                execute_final_queue(*args, launch=launch)
            self.assertEqual(len(calls), 2)
            resolved = execute_final_queue(*args, retry_failed=True, launch=launch)
            self.assertEqual(len(resolved), 4)
            self.assertEqual(calls[1], calls[2])
            self.assertEqual(len(calls), 5)
            execute_final_queue(*args, launch=launch)
            self.assertEqual(len(calls), 5)
            self.assertFalse((root/'executor/queue.lock').exists())

    def test_numerical_final_blocks_retries_and_collection(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            proto, _, plan = fixture(root)
            calls = []

            def launch(command, log):
                run = next(r for r in plan['runs'] if r['run_id'] == command[command.index('--run-id')+1])
                calls.append(run)
                trained_result(Path(command[command.index('--output')+1]), plan, run, invalid=True)
                return 3

            args = (plan, 'final.json', 'freeze.json', proto, 'rtx4070ti', root/'executor', 'admission')
            for retry in (False, True):
                with self.assertRaisesRegex(RuntimeError, 'Numerically invalid final'):
                    execute_final_queue(*args, retry_failed=retry, launch=launch)
            self.assertEqual(len(calls), 1)
            with self.assertRaisesRegex(ValueError, 'Numerically invalid final'):
                collect_training_results(plan, [root/'executor'])
            # A crash during sealing must not turn a known numerical failure into a retry.
            for marker in (root/'executor').rglob('TRAINING_FAILED'):
                marker.unlink()
            with self.assertRaisesRegex(RuntimeError, 'unsealed evidence'):
                execute_final_queue(*args, retry_failed=True, launch=launch)
            self.assertEqual(len(calls), 1)

    def test_trained_collection_checksums_and_separate_evaluation_status(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _, _, plan = fixture(root)
            for i, run in enumerate(plan['runs']):
                trained_result(root/'finals'/str(i), plan, run)
            summary = collect_training_results(plan, [root/'finals'])
            self.assertEqual(len(summary['runs']), 12)
            self.assertFalse(summary['test_evaluated'])
            # A later failed evaluation cannot invalidate or retrain the sealed training.
            (root/'finals/0/status.json').write_text('{"status":"evaluation_failed"}')
            read_training_result(root/'finals/0', plan, plan['runs'][0])
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                collect_training_results(plan, [root/'finals', root/'finals/0'])
            (root/'finals/0/best/head.safetensors').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                collect_training_results(plan, [root/'finals'])

    def test_training_cannot_publish_test_metrics_or_wrong_best(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _, _, plan = fixture(root)
            run = plan['runs'][0]
            trained_result(root/'trained', plan, run)
            for name in ('TRAINED', 'training_checksums.json'):
                (root/'trained'/name).unlink()
            metrics = json.loads((root/'trained/metrics.json').read_text())
            metrics['test_accuracy'] = 0.9
            (root/'trained/metrics.json').write_text(json.dumps(metrics))
            with self.assertRaisesRegex(ValueError, 'test metrics'):
                seal_training(root/'trained', plan, run)
            metrics['test_accuracy'] = None
            metrics['best_epoch'] = 5
            (root/'trained/metrics.json').write_text(json.dumps(metrics))
            with self.assertRaisesRegex(ValueError, 'best selection'):
                seal_training(root/'trained', plan, run)

    def test_update_audit_accepts_isolated_skip_and_rejects_scheduler_drift(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _, _, plan = fixture(root)
            run = plan['runs'][0]
            trained_result(root/'trained', plan, run)
            for name in ('TRAINED', 'training_checksums.json'):
                (root/'trained'/name).unlink()
            updates = json.loads((root/'trained/updates.json').read_text())
            for event in updates:
                event['skipped'] = event['step'] == 1
                event['lr_after_update'] = [run['lr']*(1-(event['step']-1)/180)]
            metrics = json.loads((root/'trained/metrics.json').read_text())
            metrics.update(applied_steps=179, skipped_steps=1)
            (root/'trained/updates.json').write_text(json.dumps(updates))
            (root/'trained/metrics.json').write_text(json.dumps(metrics))
            seal_training(root/'trained', plan, run)
            read_training_result(root/'trained', plan, run)
            for name in ('TRAINED', 'training_checksums.json'):
                (root/'trained'/name).unlink()
            updates[0]['lr_after_update'] = [0]
            (root/'trained/updates.json').write_text(json.dumps(updates))
            with self.assertRaisesRegex(ValueError, 'Scheduler'):
                seal_training(root/'trained', plan, run)


if __name__ == '__main__':
    unittest.main()
