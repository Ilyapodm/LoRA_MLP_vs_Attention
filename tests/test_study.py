"""V2 design, allocation, shared M-16, artifact policy and executable coordinator CLI."""
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
from experiment import make_plan as make_pilot_plan
from study_configs import (CONFIGS, SERIES, ALIASES, LR_GRID, EPOCHS, FINAL_SEEDS,
                           canonical_config, configuration_specs)
from study_protocol import (make_lr_plan, validate_plan, read_plan, make_final_plan,
                            write_final_files, validate_freeze, config_record,
                            steps_per_epoch, source_manifest, validate_selection, portable_selection)
from study_results import (seal_result, read_result, collect_search_results,
                           collect_final_results)
from study_queue import execute_queue, validate_admission, preflight
from test_experiment import protocol


def fixture(root):
    proto = protocol(root/'protocol')
    return proto, make_lr_plan(proto)


def result(root, plan, run, *, invalid=False, correct=1000):
    root.mkdir(parents=True)
    stage = plan['stage']
    config = config_record(plan, run)
    status = {'schema_version': 2, 'stage': stage, 'action': 'train',
              'status': 'numerical_invalid' if invalid else 'valid' if stage == 'lr_search' else 'trained',
              'reason': 'Nonfinite logits' if invalid else None}
    steps = EPOCHS*steps_per_epoch(plan['common'])
    history = [{'epoch': e, 'step': e*36, 'eval_correct_count': correct,
                'eval_sample_count': 2020, 'eval_accuracy': correct/2020, 'eval_loss': 1.0}
               for e in range(1, EPOCHS+1)]
    updates = [{'step': step, 'epoch': step/36, 'skipped': False, 'scale': None,
                'lr_after_update': [run['lr']*(1-step/steps)]} for step in range(1, steps+1)]
    metrics = {'test_accuracy': None, 'test_correct_count': None, 'test_sample_count': None,
               'epochs_completed': 0 if invalid else EPOCHS, 'global_step': 0 if invalid else steps,
               'attempted_steps': 0 if invalid else steps, 'applied_steps': 0 if invalid else steps,
               'skipped_steps': 0, 'best_epoch': None if invalid else 1,
               'val_correct_count': None if invalid else correct,
               'val_sample_count': None if invalid else 2020,
               'val_accuracy': None if invalid else correct/2020}
    for name, value in {'config.json': config, 'environment.json': {}, 'status.json': status,
                        'history.json': [] if invalid else history,
                        'updates.json': [] if invalid else updates, 'metrics.json': metrics}.items():
        write_json(root/name, value)
    if stage == 'final':
        write_json(root/'training_status.json', status)
    if not invalid:
        best = root/'best'
        best.mkdir()
        write_json(best/'metadata.json', {**config, 'best_epoch': 1,
                                         'val_correct_count': correct, 'val_sample_count': 2020})
        (best/('head.safetensors' if run['config'] == 'LP' else 'adapter_model.safetensors')).write_bytes(b'fixture')
        if run['config'] != 'LP':
            write_json(best/'adapter_config.json', {})
    seal_result(root, plan, run)


def lr_results(root, plan, invalid_method=None):
    for i, run in enumerate(plan['runs']):
        result(root/str(i), plan, run, invalid=run['config'] == invalid_method)


class StudyDesignTests(unittest.TestCase):
    def test_parameter_budgets_groups_shared_model_and_statistics(self):
        specs = configuration_specs()
        self.assertEqual(set(CONFIGS), {'LP', 'A-iso', 'M-iso', 'B-iso', 'A-16', 'B-16'})
        self.assertEqual(canonical_config('M-16'), 'M-iso')
        self.assertEqual([specs[n]['trainable_parameters'] for n in SERIES['iso']], [1552229]*3)
        self.assertEqual([specs[n]['trainable_parameters'] for n in SERIES['rank16']],
                         [1257317, 1552229, 2731877])
        self.assertEqual((specs['A-16']['adapter_count'], specs['B-16']['adapter_count']), (48, 72))
        with tempfile.TemporaryDirectory() as d:
            _, plan = fixture(Path(d))
            stats = plan['common']['statistics']
            self.assertEqual(len(stats['hypotheses']), 18)
            self.assertEqual(len({(h['left'], h['right'], h['seed']) for h in stats['hypotheses']}), 18)
            self.assertEqual(stats['bootstrap']['replicates'], 5000)

    def test_24_trials_eight_per_gpu_all_methods_and_immutable_plan(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            proto, plan = fixture(root)
            self.assertEqual(len(plan['runs']), 24)
            self.assertEqual(plan['common']['epochs'], 10)
            self.assertEqual(steps_per_epoch(plan['common']), 36)
            for gpu in ('rtx4070ti', 'rtx3090', 'rtx3060'):
                tasks = [r for r in plan['runs'] if r['executor'] == gpu]
                self.assertEqual(len(tasks), 8)
                self.assertEqual({r['config'] for r in tasks}, set(CONFIGS))
            for name in CONFIGS:
                self.assertEqual({r['lr'] for r in plan['runs'] if r['config'] == name}, set(LR_GRID))
            write_json(root/'runs.json', plan)
            self.assertEqual(read_plan(root/'runs.json', proto), plan)
            for target in ('epochs', 'recipe', 'run', 'alias'):
                bad = deepcopy(plan)
                if target == 'epochs': bad['common']['epochs'] = 5
                elif target == 'recipe': bad['common']['recipe']['lora_scaling'] = 'alpha=16'
                elif target == 'alias': bad['common']['aliases'] = {}
                else: bad['runs'][0]['lr'] = 0.1
                bad['plan_hash'] = canonical_hash({k: v for k, v in bad.items() if k != 'plan_hash'})
                with self.subTest(target=target), self.assertRaises(ValueError):
                    validate_plan(bad)
            with self.assertRaisesRegex(ValueError, 'pilot v1'):
                validate_plan(make_pilot_plan(proto))
            files = dict(plan['source_files'], **{'study_train.py': '0'*64})
            with patch('study_protocol.source_manifest', return_value=files):
                with self.assertRaisesRegex(ValueError, 'Code changed'):
                    read_plan(root/'runs.json', proto)

    def test_selection_freeze_18_final_tasks_and_portable_delivery(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            proto, lr_plan = fixture(root)
            lr_results(root/'lr', lr_plan)
            selected = collect_search_results(lr_plan, [root/'lr'])
            self.assertTrue(all(s['lr'] == 1e-4 for s in selected['selected'].values()))
            stale = portable_selection(selected)
            stale['trials'][0]['metrics']['global_step'] = 180
            with self.assertRaisesRegex(ValueError, 'Invalid v2 LR metrics'):
                validate_selection(stale, lr_plan)
            final = make_final_plan(lr_plan, [root/'lr'], selected)
            self.assertEqual(len(final['runs']), 18)
            for gpu, seed in zip(('rtx4070ti', 'rtx3090', 'rtx3060'), FINAL_SEEDS):
                tasks = [r for r in final['runs'] if r['executor'] == gpu]
                self.assertEqual(len(tasks), 6)
                self.assertEqual({r['seed'] for r in tasks}, {seed})
                self.assertEqual({r['config'] for r in tasks}, set(CONFIGS))
            self.assertNotIn('M-16', {r['config'] for r in final['runs']})
            shutil.move(root/'lr', root/'moved')
            self.assertEqual(make_final_plan(lr_plan, [root/'moved']), final)
            write_final_files(final, root/'final.json', proto/'freeze.json')
            validate_freeze(proto/'freeze.json', final)
            with self.assertRaises(FileExistsError):
                write_final_files(final, root/'final.json', proto/'freeze.json')
            freeze = json.loads((proto/'freeze.json').read_text())
            self.assertEqual(freeze['test_models'], 18)
            freeze['selected_lrs']['B-16'] = 0.01
            (proto/'freeze.json').write_text(json.dumps(freeze))
            with self.assertRaises(ValueError): validate_freeze(proto/'freeze.json', final)
            selected['selected']['LP']['lr'] = 0.03
            with self.assertRaisesRegex(ValueError, 'Selection differs'):
                make_final_plan(lr_plan, [root/'moved'], selected)

    def test_invalid_missing_duplicate_and_corrupt_lr_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _, plan = fixture(root)
            lr_results(root/'lr', plan, invalid_method='B-16')
            with self.assertRaisesRegex(ValueError, 'All four'):
                make_final_plan(plan, [root/'lr'])
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                collect_search_results(plan, [root/'lr', root/'lr/0'])
            (root/'lr/0/environment.json').write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                make_final_plan(plan, [root/'lr'])
            shutil.rmtree(root/'lr/0')
            with self.assertRaisesRegex(ValueError, 'all 24'):
                make_final_plan(plan, [root/'lr'])

    def test_isolated_numerical_lr_is_not_selected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _, plan = fixture(root)
            for i, run in enumerate(plan['runs']):
                result(root/'lr'/str(i), plan, run, invalid=run['config']=='B-16' and run['lr']==1e-4)
            chosen = collect_search_results(plan, [root/'lr'])
            self.assertEqual(chosen['selected']['B-16']['lr'], 1e-3)


class StudyQueueTests(unittest.TestCase):
    def test_isolated_amp_skip_and_test_metric_gate(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _, plan = fixture(root)
            run = plan['runs'][0]
            target = root/'result'
            result(target, plan, run)
            for name in ('COMPLETE', 'checksums.json'): (target/name).unlink()
            updates = json.loads((target/'updates.json').read_text())
            for event in updates:
                event['skipped'] = event['step'] == 1
                event['lr_after_update'] = [run['lr']*(1-(event['step']-1)/360)]
            metrics = json.loads((target/'metrics.json').read_text())
            metrics.update(applied_steps=359, skipped_steps=1)
            (target/'updates.json').write_text(json.dumps(updates))
            (target/'metrics.json').write_text(json.dumps(metrics))
            seal_result(target, plan, run)
            read_result(target, plan, run)
            for name in ('COMPLETE', 'checksums.json'): (target/name).unlink()
            metrics['global_step'] = 180
            (target/'metrics.json').write_text(json.dumps(metrics))
            with self.assertRaisesRegex(ValueError, 'Incomplete training'):
                seal_result(target, plan, run)
            metrics['global_step'] = 360
            metrics['test_accuracy'] = 0.9
            (target/'metrics.json').write_text(json.dumps(metrics))
            with self.assertRaisesRegex(ValueError, 'test metrics'):
                seal_result(target, plan, run)

    def test_lr_queue_continues_numerical_stops_technical_and_retries_same_task(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            proto, plan = fixture(root)
            calls = []
            def launch(command, log):
                run = next(r for r in plan['runs'] if r['run_id']==command[command.index('--run-id')+1])
                calls.append(run)
                if len(calls)==3: return 1
                result(Path(command[command.index('--output')+1]), plan, run, invalid=len(calls)==1)
                return 3 if len(calls)==1 else 0
            args = (plan, 'plan', None, proto, 'rtx3060', root/'queue', 'admission')
            with self.assertRaises(RuntimeError): execute_queue(*args, launch=launch)
            with self.assertRaises(ValueError): execute_queue(*args, launch=launch)
            self.assertEqual(len(calls), 3)
            self.assertEqual(len(execute_queue(*args, retry_failed=True, launch=launch)), 8)
            self.assertEqual(calls[2], calls[3])
            count = len(calls)
            execute_queue(*args, launch=launch)
            self.assertEqual(len(calls), count)
            self.assertFalse((root/'queue/queue.lock').exists())

    def test_final_failure_is_not_retried_and_trained_results_are_shared(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            proto, lr = fixture(root)
            lr_results(root/'lr', lr)
            plan = make_final_plan(lr, [root/'lr'])
            calls = []
            def launch(command, log):
                run = next(r for r in plan['runs'] if r['run_id']==command[command.index('--run-id')+1])
                calls.append(run)
                self.assertIn('--freeze', command)
                result(Path(command[command.index('--output')+1]), plan, run, invalid=True)
                return 3
            args = (plan, 'plan', 'freeze', proto, 'rtx3090', root/'queue', 'admission')
            for retry in (False, True):
                with self.assertRaisesRegex(RuntimeError, 'Numerically invalid final'):
                    execute_queue(*args, retry_failed=retry, launch=launch)
            self.assertEqual(len(calls), 1)
            for marker in (root/'queue').rglob('TRAINING_FAILED'): marker.unlink()
            with self.assertRaisesRegex(RuntimeError, 'Unsealed numerical'):
                execute_queue(*args, retry_failed=True, launch=launch)
            for i, run in enumerate(plan['runs']): result(root/'final'/str(i), plan, run)
            summary = collect_final_results(plan, [root/'final'])
            self.assertEqual(len(summary['runs']), 18)
            self.assertEqual(summary['series']['rank16'][1], 'M-iso')
            self.assertFalse(summary['test_evaluated'])
            (root/'final/0/status.json').write_text('{"status":"evaluating"}')
            read_result(root/'final/0', plan, plan['runs'][0])

    def test_preflight_covers_all_six_and_real_b16_and_rejects_v1_admission(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            proto, plan = fixture(root)
            plan_path = root/'lr.json'
            write_json(plan_path, plan)
            env = {'gpu': 'NVIDIA GeForce RTX 3060', 'fixture': True}
            commands = []
            with patch('study_queue.environment', return_value=env), \
                    patch('study_queue.subprocess.run', side_effect=lambda cmd, **kw: commands.append(cmd)):
                preflight(plan, plan_path, proto, 'rtx3060', root/'executor')
            receipt = next((root/'executor').rglob('admission.json'))
            validate_admission(receipt, plan, 'rtx3060', env)
            self.assertEqual(len(commands), 4)
            self.assertTrue(any('--parameters' in cmd and any('study_check.py' in s for s in cmd) for cmd in commands))
            self.assertTrue(any('--technical' in cmd for cmd in commands))
            value = json.loads(receipt.read_text())
            value['schema_version'] = 1
            receipt.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, 'new preflight'):
                validate_admission(receipt, plan, 'rtx3060', env)

    def test_coordinator_cli_creates_plan_and_dry_run_without_ml(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            proto, _ = fixture(root)
            script = Path(__file__).resolve().parents[1]/'study_queue.py'
            base = [sys.executable, '-B', str(script), '--plan', str(root/'runs.json'), '--protocol', str(proto)]
            subprocess.run(base+['--make-lr-plan'], capture_output=True, text=True, check=True)
            output = subprocess.run(base+['--dry-run', '--executor', 'rtx4070ti'],
                                    capture_output=True, text=True, check=True)
            tasks = [json.loads(line) for line in output.stdout.splitlines()]
            self.assertEqual(len(tasks), 8)
            self.assertEqual({r['config'] for r in tasks}, set(CONFIGS))


if __name__ == '__main__': unittest.main()
