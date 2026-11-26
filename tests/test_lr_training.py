"""Full split counts and five-epoch LR path on a small CPU model, no Hub/GPU."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('transformers'),'ML stack required')
class ScientificTrainingTests(unittest.TestCase):
    def test_full_counts_five_epochs_and_numeric_invalid(self):
        import torch
        from transformers import TrainingArguments
        from train import execute_lr_search
        from experiment import make_plan, read_result
        from test_experiment import protocol
        torch.set_num_threads(2)
        class Dataset:
            def __init__(self,n):self.n=n;self.epoch=0
            def __len__(self):return self.n
            def set_epoch(self,epoch):self.epoch=epoch
            def __getitem__(self,i):
                return {'pixel_values':torch.tensor([i%7/7,i%11/11,self.epoch/10]),'labels':i%101}
        class Model(torch.nn.Module):
            def __init__(self,invalid=False):
                super().__init__();self.classifier=torch.nn.Linear(3,101);self.invalid=invalid
                with torch.no_grad():
                    self.classifier.weight.zero_();self.classifier.bias.zero_();self.classifier.bias[0]=10
            def forward(self,pixel_values):
                logits=self.classifier(pixel_values)
                return {'logits':logits*float('nan') if self.invalid else logits}
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);plan=make_plan(protocol(root/'protocol'));run=next(r for r in plan['runs'] if r['config']=='LP')
            for invalid in (False,True):
                out=root/str(invalid)
                args=TrainingArguments(output_dir=str(out/'checkpoints'),use_cpu=True,num_train_epochs=5,
                    per_device_train_batch_size=16,gradient_accumulation_steps=32,per_device_eval_batch_size=32,
                    learning_rate=run['lr'],weight_decay=0,optim='adamw_torch',seed=142,data_seed=142,
                    eval_strategy='epoch',save_strategy='epoch',logging_strategy='no',save_total_limit=2,
                    load_best_model_at_end=True,metric_for_best_model='correct_count',greater_is_better=True,
                    report_to='none',disable_tqdm=True,remove_unused_columns=False,label_names=['labels'])
                status=execute_lr_search(Model(invalid),Dataset(18180),Dataset(2020),args,out,
                    {'common':plan['common'],'run':run},{'fixture':'CPU integration'},collator=None)
                result=read_result(out,plan,run)
                self.assertEqual(status,'numerical_invalid' if invalid else 'valid')
                self.assertIsNone(result['metrics']['test_accuracy'])
                if not invalid:
                    self.assertEqual(result['metrics']['best_epoch'],1)
                    self.assertEqual(result['metrics']['global_step'],180)
                    self.assertEqual(result['metrics']['applied_steps'],180)
                    self.assertEqual(len([h for h in json.loads((out/'history.json').read_text()) if 'eval_correct_count' in h]),5)
