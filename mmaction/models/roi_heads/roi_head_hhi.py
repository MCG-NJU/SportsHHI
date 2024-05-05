# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

import torch
from torch import Tensor

from .hhi_sampler import HHIRandomSampler

from mmaction.utils import ConfigType, InstanceList, SampleList
from mmaction.structures.bbox import bboxes2union

from .hhi_sampler import HHISamplingResult

try:
    from mmdet.models.roi_heads import StandardRoIHead
    # from mmdet.models.task_modules.samplers import SamplingResult
    from mmdet.registry import MODELS as MMDET_MODELS
    from mmdet.registry import TASK_UTILS
    from mmdet.structures.bbox import bbox2roi
    mmdet_imported = True
except (ImportError, ModuleNotFoundError):
    # from mmaction.utils import SamplingResult
    mmdet_imported = False

if mmdet_imported:

    @MMDET_MODELS.register_module()
    class HHIRoIHead(StandardRoIHead):
        
        def init_assigner_sampler(self):
            self.bbox_assigner = None
            self.bbox_sampler = None
            if self.train_cfg:
                self.bbox_assigner = TASK_UTILS.build(self.train_cfg.assigner)
                sampler_cfg = self.train_cfg.sampler
                assert sampler_cfg.type == 'HHIRandomSampler'
                self.bbox_sampler = HHIRandomSampler(
                    num=sampler_cfg.num,
                    pos_fraction=sampler_cfg.pos_fraction,
                    neg_pos_ub=sampler_cfg.neg_pos_ub,
                    add_gt_as_proposals=sampler_cfg.add_gt_as_proposals
                )

        def loss(self, x: Union[Tensor, Tuple[Tensor],],
                 rpn_results_list: InstanceList,
                 data_samples: SampleList, **kwargs) -> dict:
            """Perform forward propagation and loss calculation of the
            detection roi on the features of the upstream network.

            Args:
                x (Tensor or Tuple[Tensor]): The image features extracted by
                    the upstream network.
                rpn_results_list (List[:obj:`InstanceData`]): List of region
                    proposals.
                data_samples (List[:obj:`ActionDataSample`]): The batch
                    data samples.

            Returns:
                Dict[str, Tensor]: A dictionary of loss components.
            """
            assert len(rpn_results_list) == len(data_samples)
            for data_sample in data_samples:
                # ['p1_ids', 'p2_ids', 'img_shape', 'gt_interactions', 'scores', 'proposals', 'gt_instances'] ['bboxes', 'labels']

                # print(data_sample.all_keys(), data_sample.gt_instances.all_keys())
                # print(data_sample.p1_ids)
                # print(data_sample.p2_ids)
                # print(data_sample.gt_interactions.shape)
                # print(len(data_sample.gt_instances.bboxes), len(data_sample.gt_instances.labels), len(data_sample.proposals), len(data_sample.scores))
                # exit()
                assert len(data_sample.p1_ids) == len(data_sample.p2_ids)
                assert len(data_sample.p1_ids) == len(data_sample.gt_interactions)
                assert len(data_sample.gt_instances.bboxes) == len(data_sample.gt_instances.labels)
                assert len(data_sample.proposals) == len(data_sample.scores)

            num_imgs = len(data_samples)
            sampling_results = []
            for i in range(num_imgs):
                data_sample = data_samples[i]
                rpn_results = rpn_results_list[i]
                rpn_results.priors = rpn_results.pop('bboxes')

                assign_result = self.bbox_assigner.assign(
                    rpn_results, data_sample.gt_instances, None
                )
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    data_sample.proposals,
                    data_sample.p1_ids,
                    data_sample.p2_ids,
                    data_sample.gt_instances.bboxes,
                    data_sample.gt_interactions
                )
                sampling_results.append(sampling_result)

            losses = dict()
            # bbox head forward and loss
            bbox_results = self._bbox_forward_train(x, sampling_results)
            losses.update(bbox_results['loss_bbox'])
            
            return losses
        
        def _bbox_forward_train(self, x, sampling_results):
            """Run forward function and calculate loss for box head in
            training."""
            rois = bbox2roi([res.bboxes for res in sampling_results])
            p1_bbox_inds = []
            p2_bbox_inds = []
            prev_bbox_cnt = 0
            union_bbox_list = []
            for batch_id in range(len(sampling_results)):
                p1_bbox_inds.append(sampling_results[batch_id].p1_bbox_inds + prev_bbox_cnt)
                p2_bbox_inds.append(sampling_results[batch_id].p2_bbox_inds + prev_bbox_cnt)
                prev_bbox_cnt += len(sampling_results[batch_id].bboxes)
                p1_bboxes = sampling_results[batch_id].bboxes[sampling_results[batch_id].p1_bbox_inds]
                p2_bboxes = sampling_results[batch_id].bboxes[sampling_results[batch_id].p2_bbox_inds]
                union_bboxes = bboxes2union(p1_bboxes, p2_bboxes)
                union_bbox_list.append(union_bboxes)
            p1_bbox_inds = torch.cat(p1_bbox_inds)
            p2_bbox_inds = torch.cat(p2_bbox_inds)

            union_rois = bbox2roi(union_bbox_list)

            bbox_results = self._bbox_forward(x, rois, p1_bbox_inds, p2_bbox_inds, union_rois)

            bbox_loss_and_target = self.bbox_head.loss_and_target(
                cls_score=bbox_results['cls_score'],
                rois=rois,
                sampling_results=sampling_results,
                rcnn_train_cfg=self.train_cfg)
            
            bbox_results.update(loss_bbox=bbox_loss_and_target['loss_bbox'])
            return bbox_results

        def _bbox_forward(self, x, rois, p1_bbox_inds, p2_bbox_inds, union_rois, test=False) -> dict:
            roi_feat, _ = self.bbox_roi_extractor(x, rois)
            union_roi_feat, _ = self.bbox_roi_extractor(x, union_rois)

            if self.with_shared_head:
                raise NotImplementedError('No shared head allowd for Relation Dataset')

            cls_score = self.bbox_head(rois, roi_feat, p1_bbox_inds, p2_bbox_inds, union_roi_feat)

            bbox_results = dict(
                cls_score=cls_score, bbox_feats=None)
            return bbox_results
        
        def bbox_loss(self, x: Union[Tensor, Tuple[Tensor]],
                      sampling_results: List[HHISamplingResult],
                      batch_img_metas: List[dict]=None, **kwargs) -> dict:
            rois = bbox2roi([res.bboxes for res in sampling_results])
            p1_bbox_inds = []
            p2_bbox_inds = []
            prev_bbox_cnt = 0
            union_bbox_list = []
            for batch_id in range(len(sampling_results)):
                p1_bbox_inds.append(sampling_results[batch_id].p1_bbox_inds + prev_bbox_cnt)
                p2_bbox_inds.append(sampling_results[batch_id].p2_bbox_inds + prev_bbox_cnt)
                prev_bbox_cnt += len(sampling_results[batch_id].bboxes)
                p1_bboxes = sampling_results[batch_id].bboxes[sampling_results[batch_id].p1_bbox_inds]
                p2_bboxes = sampling_results[batch_id].bboxes[sampling_results[batch_id].p2_bbox_inds]
                union_bboxes = bboxes2union(p1_bboxes, p2_bboxes)
                union_bbox_list.append(union_bboxes)
            p1_bbox_inds = torch.cat(p1_bbox_inds)
            p2_bbox_inds = torch.cat(p2_bbox_inds)

            union_rois = bbox2roi(union_bbox_list)

            bbox_results = self._bbox_forward(x, rois, p1_bbox_inds, p2_bbox_inds, union_rois)

            bbox_loss_and_target = self.bbox_head.loss_and_target(
                cls_score=bbox_results['cls_score'],
                rois=rois,
                sampling_results=sampling_results,
                rcnn_train_cfg=self.train_cfg
            )

            bbox_results.update(loss_bbox=bbox_loss_and_target['loss_bbox'])
            return bbox_results
        
        def predict(self, x: Union[Tensor, Tuple[Tensor]],
                    rpn_results_list: InstanceList,
                    data_samples: SampleList, **kwargs) -> InstanceList:
            assert self.with_bbox, 'Bbox head must be implemented'
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]

            if isinstance(x, tuple):
                x_shape = x[0].shape
            else:
                x_shape = x.shape

            assert x_shape[0] == 1, 'only accept 1 sample at test mode'
            assert x_shape[0] == len(batch_img_metas) == len(rpn_results_list)

            results_list = self.predict_bbox(
                x,
                batch_img_metas,
                rpn_results_list,
                rcnn_test_cfg=self.test_cfg)
            
            return results_list
        
        def predict_bbox(self, x: Tuple[Tensor],
                         batch_img_metas: List[dict],
                         rpn_results_list: InstanceList,
                         rcnn_test_cfg: ConfigType) -> InstanceList:
            assert len(rpn_results_list) == 1

            proposals = [res.bboxes for res in rpn_results_list]

            num_bboxes = len(proposals[0])
            bbox_inds = torch.tensor([i for i in range(num_bboxes)])
            p1_bbox_inds = torch.cat([bbox_ind.repeat(num_bboxes) for bbox_ind in bbox_inds], 0)
            p2_bbox_inds = bbox_inds.repeat(num_bboxes)
            bbox_union_list = [bboxes2union(proposals[0][p1_bbox_inds], proposals[0][p2_bbox_inds])]

            rois = bbox2roi(proposals)
            union_rois = bbox2roi(bbox_union_list)

            bbox_results = self._bbox_forward(x, rois, p1_bbox_inds, p2_bbox_inds, union_rois, test=True)

            cls_scores = bbox_results['cls_score']
            num_proposals_per_img = tuple(len(p) for p in proposals)
            rois = rois.split(num_proposals_per_img, 0)
            cls_scores = cls_scores.split(len(cls_scores), 0)

            result_list = self.bbox_head.predict_by_feat(
                rois=rois,
                cls_scores=cls_scores,
                batch_img_metas=batch_img_metas,
                rcnn_test_cfg=rcnn_test_cfg
            )

            return result_list

        

else:

    class HHIRoIHead:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                'Failed to import `bbox2roi` from `mmdet.core.bbox`, '
                'or failed to import `MODELS` from `mmdet.registry`, '
                'or failed to import `StandardRoIHead` from '
                '`mmdet.models.roi_heads`. You will be unable to use '
                '`HHIRoIHead`. ')