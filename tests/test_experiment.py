"""Scientific plan coverage, local queue failure policy and LR selection."""
from copy import deepcopy
import json
from pathlib import Path
import runpy
import tempfile
import unittest

from experiment import (make_plan, validate_plan, canonical_hash, finish_result, read_result,
                        collect_results, EXECUTORS)
from run_queue import execute_queue, queue_lock


def protocol(root):
    root.mkdir()
    m=runpy.run_path(str(Path(__file__).with_name('test_data.py')))['fixture']()
    (root/'data_manifest.json').write_text(json.dumps(m))
    (root/'dataset_revision.json').write_text(json.dumps({'dataset':'ethz/food101','revision':'a'*40}))
    (root/'model_revision.json').write_text(json.dumps({'model':'google/vit-base-patch16-224-in21k','revision':'b'*40}))
    return root


def result(root, plan, run, status='valid', correct=1000):
    root.mkdir(parents=True)
    payloads={'config.json':{'common':plan['common'],'run':run}, 'environment.json':{},
        'status.json':{'stage':'lr_search','status':status,'reason':None},'history.json':[], 'updates.json':[],
        'metrics.json':{'epochs_completed':5,'val_correct_count':correct,'val_sample_count':2020,
        'val_accuracy':correct/2020,'best_epoch':1,'test_accuracy':None,'test_correct_count':None,'test_sample_count':None}}
    for n,v in payloads.items():(root/n).write_text(json.dumps(v))
    if status=='valid':
        (root/'best').mkdir();(root/'best'/'metadata.json').write_text('{}')
        (root/'best'/('head.safetensors' if run['config']=='LP' else 'adapter_model.safetensors')).write_bytes(b'fixture')
        if run['config'] != 'LP':(root/'best'/'adapter_config.json').write_text('{}')
    finish_result(root)


class PlanQueueTests(unittest.TestCase):
    def test_coverage_assignments_and_protocol_tamper(self):
        with tempfile.TemporaryDirectory() as d:
            plan=make_plan(protocol(Path(d)/'protocol'))
            self.assertEqual(len(plan['runs']),12)
            for gpu in EXECUTORS:
                self.assertEqual({r['config'] for r in plan['runs'] if r['executor']==gpu},{'LP','A-iso','M-iso','B-iso'})
            for method in ('LP','A-iso','M-iso','B-iso'):
                self.assertEqual({r['executor'] for r in plan['runs'] if r['config']==method},set(EXECUTORS))
            bad=deepcopy(plan);bad['common']['epochs']=6
            bad['plan_hash']=canonical_hash({k:v for k,v in bad.items() if k!='plan_hash'})
            with self.assertRaises(ValueError):validate_plan(bad)

    def test_selection_ties_invalid_candidates_and_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);plan=make_plan(protocol(root/'protocol'))
            for i,run in enumerate(plan['runs']):
                result(root/'results'/str(i),plan,run,'numerical_invalid' if run['config']=='B-iso' and run['lr']==1e-4 else 'valid')
            selected=collect_results(plan,[root/'results'])
            self.assertEqual(selected['selected']['LP']['lr'],1e-4)
            self.assertEqual(selected['selected']['B-iso']['lr'],1e-3)
            (root/'results'/'0'/'metrics.json').write_text('{}')
            with self.assertRaises(ValueError):collect_results(plan,[root/'results'])

    def test_queue_continues_invalid_skips_completed_and_stops_technical_failure(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);proto=protocol(root/'protocol');plan=make_plan(proto);calls=[]
            runs=[r for r in plan['runs'] if r['executor']==EXECUTORS[0]]
            def launch(command,log):
                identifier=command[command.index('--run-id')+1];run=next(r for r in runs if r['run_id']==identifier)
                output=Path(command[command.index('--output')+1]);calls.append(identifier)
                if len(calls)==3:return 1
                result(output,plan,run,'numerical_invalid' if len(calls)==1 else 'valid')
                return 3 if len(calls)==1 else 0
            with self.assertRaises(RuntimeError):
                execute_queue(plan,'plan.json',proto,EXECUTORS[0],root/'executor','admission',launch=launch)
            self.assertEqual(len(calls),3)
            with self.assertRaises(ValueError):
                execute_queue(plan,'plan.json',proto,EXECUTORS[0],root/'executor','admission',launch=launch)
            self.assertEqual(len(calls),3)
            with queue_lock(root/'executor'):
                with self.assertRaises(FileExistsError):
                    with queue_lock(root/'executor'):pass

    def test_all_invalid_blocks_selection(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);plan=make_plan(protocol(root/'protocol'))
            for i,run in enumerate(plan['runs']):result(root/'results'/str(i),plan,run,'numerical_invalid' if run['config']=='LP' else 'valid')
            with self.assertRaisesRegex(ValueError,'All LR candidates'):collect_results(plan,[root/'results'])
