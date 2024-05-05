# Copyright (c) OpenMMLab. All rights reserved.
import os
from datetime import datetime
from typing import Any, List, Optional, Sequence, Tuple

import torch

from mmengine.evaluator import BaseMetric

from mmaction.evaluation import hhi_eval, results2csv_hhi
from mmaction.registry import METRICS
from mmaction.structures import bbox2result_hhi


@METRICS.register_module()
class HHIMetric(BaseMetric):
    """HHI evaluation metric."""
    default_prefix: Optional[str] = 'mAP'

    def __init__(self,
                 ann_file: str,
                 exclude_file: str,
                 label_file: str,
                 options: Tuple[str] = ('mAP', ),
                 action_thr: float = 0.002,
                 num_classes: int = 34,
                 custom_classes: Optional[List[int]] = None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None):
        super().__init__(collect_device=collect_device, prefix=prefix)
        assert len(options) == 1
        self.ann_file = ann_file
        self.exclude_file = exclude_file
        self.label_file = label_file
        self.num_classes = num_classes
        self.options = options
        self.action_thr = action_thr
        self.custom_classes = custom_classes
        if custom_classes is not None:
            self.custom_classes = list([0] + custom_classes)

    def process(self, data_batch: Sequence[Tuple[Any, dict]],
                data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            result = dict()
            pred = data_sample['pred_instances']
            result['video_id'] = data_sample['video_id']
            result['timestamp'] = data_sample['timestamp']

            num_bboxes = len(pred['bboxes'])
            bbox_inds = torch.tensor([i for i in range(num_bboxes)])
            p1_bbox_inds = torch.cat([bbox_ind.repeat(num_bboxes) for bbox_ind in bbox_inds], 0)
            p2_bbox_inds = bbox_inds.repeat(num_bboxes)

            outputs = bbox2result_hhi(
                pred['bboxes'],
                p1_bbox_inds,
                p2_bbox_inds,
                pred['scores'],
                num_classes=self.num_classes,
                thr=self.action_thr
            )
            result['outputs'] = outputs
            self.results.append(result)
    
    def compute_metrics(self, results: list) -> dict:
        time_now = datetime.now().strftime('%Y%m%d_%H%M%S')
        temp_file = f'HHI_{time_now}_result.csv'
        results2csv_hhi(results, temp_file, self.custom_classes)

        ret = {}
        eval_results_all = hhi_eval(
            temp_file,
            self.options[0],
            self.label_file,
            self.ann_file,
            self.exclude_file,
            verbose=True,
            ignore_empty_frames=True,
            custom_classes=self.custom_classes,
            pred_max=-1
        )
        ret['overall'] = eval_results_all['overall']

        for pred_max in [20, 50, 100, 150]:
            eval_results = hhi_eval(
                temp_file,
                self.options[0],
                self.label_file,
                self.ann_file,
                self.exclude_file,
                verbose=False,
                ignore_empty_frames=True,
                custom_classes=self.custom_classes,
                pred_max=pred_max
            )
            ret[f'recall@{pred_max}'] = eval_results[f'recall@{pred_max}']
        os.remove(temp_file)
        
        return ret