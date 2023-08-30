import logging

import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchmetrics
from torch.cuda.amp import autocast
from torch.nn.functional import pad


class TestEngine:
    def __init__(self, model, device, num_classes, log_prefix, task='multiclass', verbose=False,
                 logging_interval=50):
        assert task in ['binary', 'multiclass', 'multilabel'], f'{task} must be binary, multiclass, or multilabel'

        self.model = model
        self.device = device
        self.num_classes = num_classes
        self.log_prefix = log_prefix

        self.verbose = verbose
        self.logging_interval = logging_interval

        self.metrics = ['Accuracy', 'F1Score', 'AUROC', 'Specificity', 'Recall', 'Precision']
        self.metric_fn = self.init_metrics(task, 0.5, num_classes, num_classes, 'micro')

    def __call__(self, *args, **kwargs):
        return self.validate(*args, **kwargs)

    def iterate(self, model, data):
        x, y = map(lambda x: x.to(self.device), data)
        x = x.to(memory_format=torch.channels_last)

        with autocast(enabled=True):
            prob = model(x)

        return prob, y

    @torch.no_grad()
    def validate(self, loader):
        self._reset_metric()
        model = self.model
        total = len(loader)

        model.eval()
        for i, data in enumerate(loader):
            prob, target = self.iterate(model, data)
            self._update_metric(prob, target)

            _metrics = self._metrics()
            if i % self.logging_interval == 0 or i == total - 1:
                logging.info(self._print(_metrics, i, total, self.log_prefix))
            break
        return self._metrics()

    def _update_metric(self, prob, target):
        for fn in self.metric_fn.values():
            fn.update(prob, target)

    def _reset_metric(self):
        for fn in self.metric_fn.values():
            fn.reset()

    def _metrics(self):
        result = dict()
        for k, fn in self.metric_fn.items():
            result[k] = fn.compute().tolist()
        return result

    @staticmethod
    def _print(metrics, i, max_iter, mode):
        log = f'{mode}: [{i:>4d}/{max_iter}]  '
        for k, v in metrics.items():
            log += f'{k}:{v:.6f} | '
        return log[:-3]

    def init_metrics(self, task, threshold, num_class, num_label, average, top_k=1):
        metric_fn = dict()

        for metric in self.metrics:
            metric_fn[metric] = torchmetrics.__dict__[metric](task=task, num_classes=num_class,
                                                              average='macro' if metric == 'AUROC' else average,
                                                              num_labels=num_label).to(self.device)
        return metric_fn


class ConsistencyEngine(TestEngine):
    def __init__(self, model, device, num_classes, log_prefix, task='multiclass', verbose=False,
                 logging_interval=50):
        super().__init__(model, device, num_classes, log_prefix, task, verbose, logging_interval)
        self.metrics = ['Accuracy']
        self.metric_fn = self.init_metrics(task, 0.5, num_classes, num_classes, 'micro')

    def __call__(self, *args, **kwargs):
        return self.validate(*args, **kwargs)

    def iterate(self, model, data):
        x, y = map(lambda x: x.to(self.device), data)
        x = x.to(memory_format=torch.channels_last)

        _, _, H, W = x.shape

        off0 = np.random.randint(32, size=2)
        off1 = np.random.randint(32, size=2)

        x0 = x[:, :, off0[0]:off0[0] + 224, off0[1]:off0[1] + 224]
        x1 = x[:, :, off1[0]:off1[0] + 224, off1[1]:off1[1] + 224]

        x0 = pad(x0, [0, W - x0.size(3), 0, H - x0.size(2)])
        x1 = pad(x1, [0, W - x1.size(3), 0, H - x1.size(2)])

        with autocast(enabled=True):
            output0 = model(x0)
            output1 = model(x1)

        return output0, output1.argmax(dim=1, keepdim=False)

    @torch.no_grad()
    def validate(self, loader):
        self._reset_metric()
        model = self.model
        total = len(loader)

        model.eval()
        for i, data in enumerate(loader):
            prob, target = self.iterate(model, data)
            self._update_metric(prob, target)

            _metrics = self._metrics()
            if i % self.logging_interval == 0 or i == total - 1:
                logging.info(self._print(_metrics, i, total, self.log_prefix))
            break
        return self._metrics()
