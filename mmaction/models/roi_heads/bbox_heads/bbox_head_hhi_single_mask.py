from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from torch import Tensor

from .hhi_utils import transformer, my_minmax
from ..hhi_sampler import HHISamplingResult

from mmaction.structures.bbox import hhi_target
from mmaction.utils import InstanceList

try:
    # from mmdet.models.task_modules.samplers import SamplingResult
    from mmdet.registry import MODELS as MMDET_MODELS
    mmdet_imported = True
except (ImportError, ModuleNotFoundError):
    # from mmaction.utils import SamplingResult
    mmdet_imported = False

from packaging import version as pv

if pv.parse(torch.__version__) < pv.parse('1.10'):

    def cross_entropy_loss(input, target, reduction='None'):
        input = input.log_softmax(dim=-1)  # Compute Log of Softmax
        loss = -(input * target).sum(dim=-1)  # Compute Loss manually
        if reduction.lower() == 'mean':
            return loss.mean()
        elif reduction.lower() == 'sum':
            return loss.sum()
        else:
            return loss
else:
    cross_entropy_loss = F.cross_entropy


def boxes2union_single(box1: torch.Tensor, box2: torch.Tensor, pooling_size: int) -> torch.Tensor:
    """generate spatial masks for bboxes
    modified from https://github.com/rowanz/neural-motifs/blob/master/lib/draw_rectangles/draw_rectangles.pyx

    Args:
        box1 (torch.Tensor): tensor of p1 bboxes
        box2 (torch.Tensor): tensor of p2 bboxes
        pooling_size (int): size of spatial mask

    Returns:
        torch.Tensor: spatial masks
    """
    assert box1.shape == box2.shape
    n, _ = box1.shape

    x1_union = torch.stack((box1[:,0], box2[:,0]), 1).min(1).values
    y1_union = torch.stack((box1[:,1], box2[:,1]), 1).min(1).values
    x2_union = torch.stack((box1[:,2], box2[:,2]), 1).max(1).values
    y2_union = torch.stack((box1[:,3], box2[:,3]), 1).max(1).values

    w = x2_union - x1_union
    h = y2_union - y1_union

    x_mat = torch.arange(pooling_size).repeat(pooling_size, 1)
    x_mat = x_mat.repeat(n, 1, 1).float().to(box1.device)
    y_mat = torch.arange(pooling_size).reshape(-1,1).repeat(1,pooling_size)
    y_mat = y_mat.repeat(n,1,1).float().to(box1.device)
    zero_mat = torch.zeros(x_mat.shape).to(box1.device)
    one_mat = torch.ones(x_mat.shape).to(box1.device)

    x1_box_1 = ((box1[:,0] - x1_union) * pooling_size / w).reshape(-1,1).repeat(1,pooling_size*pooling_size).reshape(-1, pooling_size, pooling_size)
    y1_box_1 = ((box1[:,1] - y1_union) * pooling_size / h).reshape(-1,1).repeat(1,pooling_size*pooling_size).reshape(-1, pooling_size, pooling_size)
    x2_box_1 = ((box1[:,2] - x1_union) * pooling_size / w).reshape(-1,1).repeat(1,pooling_size*pooling_size).reshape(-1, pooling_size, pooling_size)
    y2_box_1 = ((box1[:,3] - y1_union) * pooling_size / h).reshape(-1,1).repeat(1,pooling_size*pooling_size).reshape(-1, pooling_size, pooling_size)
    
    x_contrib_1 = my_minmax(x_mat+one_mat-x1_box_1,zero_mat,one_mat) * my_minmax(x2_box_1-x_mat,zero_mat,one_mat)
    y_contrib_1 = my_minmax(y_mat+one_mat-y1_box_1,zero_mat,one_mat) * my_minmax(y2_box_1-y_mat,zero_mat,one_mat)

    x1_box_2 = ((box2[:,0] - x1_union) * pooling_size / w).reshape(-1,1).repeat(1,pooling_size*pooling_size).reshape(-1, pooling_size, pooling_size)
    y1_box_2 = ((box2[:,1] - y1_union) * pooling_size / h).reshape(-1,1).repeat(1,pooling_size*pooling_size).reshape(-1, pooling_size, pooling_size)
    x2_box_2 = ((box2[:,2] - x1_union) * pooling_size / w).reshape(-1,1).repeat(1,pooling_size*pooling_size).reshape(-1, pooling_size, pooling_size)
    y2_box_2 = ((box2[:,3] - y1_union) * pooling_size / h).reshape(-1,1).repeat(1,pooling_size*pooling_size).reshape(-1, pooling_size, pooling_size)
    x_contrib_2 = my_minmax(x_mat+one_mat-x1_box_2,zero_mat,one_mat) * my_minmax(x2_box_2-x_mat,zero_mat,one_mat)
    y_contrib_2 = my_minmax(y_mat+one_mat-y1_box_2,zero_mat,one_mat) * my_minmax(y2_box_2-y_mat,zero_mat,one_mat)

    return torch.stack([x_contrib_1*y_contrib_1 + x_contrib_2*y_contrib_2, x_contrib_1*y_contrib_1 + x_contrib_2*y_contrib_2]).transpose(1,0)


class BBoxHeadHHI_single_mask(nn.Module):
    def __init__(
            self,
            temporal_pool_type:str='avg',
            spatial_pool_type:str='max',
            in_channels:int=2048,
            focal_gamma:float=0.,
            focal_alpha:float=1.,
            num_classes:int=35,
            dropout_ratio:float=0,
            dropout_before_pool:bool=True,
            topk:Union[int,Tuple[int]]=None,
            multilabel:bool=False,
            use_spatial:bool=True,
            use_attention:bool=True,
            mlp_head:bool=False) -> None:
        super(BBoxHeadHHI_single_mask, self).__init__()
        assert temporal_pool_type in ['max', 'avg']
        assert spatial_pool_type in ['max', 'avg']
        self.temporal_pool_type = temporal_pool_type
        self.spatial_pool_type = spatial_pool_type

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.dropout_ratio = dropout_ratio
        self.dropout_before_pool = dropout_before_pool

        self.multilabel = multilabel

        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha

        if topk is None:
            self.topk = ()
        elif isinstance(topk, int):
            self.topk = (topk,)
        elif isinstance(topk, tuple):
            assert all([isinstance(k, int) for k in topk])
        else:
            raise TypeError('topk shohuld be int or tuple[int], '
                            f'but get {type(topk)}')
        assert all([k < num_classes for k in self.topk])

        self.use_spatial = use_spatial
        self.use_attention = use_attention

        in_channels = self.in_channels
        # Pool by default
        if self.temporal_pool_type == 'avg':
            self.temporal_pool = nn.AdaptiveAvgPool3d((1, None, None))
        else:
            self.temporal_pool = nn.AdaptiveMaxPool3d((1, None, None))
        if self.spatial_pool_type == 'avg':
            self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        else:
            self.spatial_pool = nn.AdaptiveMaxPool3d((None, 1, 1))

        if dropout_ratio > 0:
            self.dropout = nn.Dropout(dropout_ratio)

        if self.use_spatial:
            self.conv = nn.Sequential(
                nn.Conv2d(2, 256 //2, kernel_size=7, stride=2, padding=3, bias=True),
                # nn.Sigmoid(),
                # nn.ReLU(inplace=True),
                # nn.BatchNorm2d(256//2, momentum=0.01),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(256 // 2, 256, kernel_size=3, stride=1, padding=1, bias=True),
                nn.Sigmoid()
                # nn.ReLU(inplace=True),
                # nn.BatchNorm2d(256, momentum=0.01),
            )
            self.sp_fc = nn.Linear(256*7*7, 256)
        self.p1_fc = nn.Linear(in_channels, 512)
        self.p2_fc = nn.Linear(in_channels, 512)
        self.vr_fc = nn.Linear(in_channels, 512)

        if self.use_attention:
            if self.use_spatial:
                self.transformer = transformer(enc_layer_num=1, embed_dim=1792, dropout=dropout_ratio)
            else:
                self.transformer = transformer(enc_layer_num=1, embed_dim=1536, dropout=dropout_ratio)

        if self.use_spatial:
            cls_in_channels = 1792
        else:
            cls_in_channels = 1536
        if mlp_head:
            self.fc_cls = nn.Sequential(
                nn.Linear(cls_in_channels, cls_in_channels),
                nn.ReLU(),
                nn.Linear(cls_in_channels, num_classes)
            )
        else:
            self.fc_cls = nn.Linear(cls_in_channels, num_classes)

    def init_weights(self) -> None:
        """Initialize the classification head."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, rois, x, p1_inds, p2_inds, x_union) -> Tensor:
        if self.dropout_before_pool and self.dropout_ratio > 0:
            x = self.dropout(x)
            x_union = self.dropout(x_union)
        
        x = self.temporal_pool(x)
        x_union = self.temporal_pool(x_union)
        x = self.spatial_pool(x)
        x_union = self.spatial_pool(x_union)

        if not self.dropout_before_pool and self.dropout_ratio > 0:
            x = self.dropout(x)
            x_union = self.dropout(x_union)

        x = x.view(x.size(0), -1)
        x_union = x_union.view(x_union.size(0), -1)

        p1_rep = x[p1_inds]
        p1_rep = self.p1_fc(p1_rep)
        p2_rep = x[p2_inds]
        p2_rep = self.p2_fc(p2_rep)
        vr_rep = self.vr_fc(x_union)

        if self.use_spatial:
            p1_boxes = rois[p1_inds, 1:]
            p2_boxes = rois[p2_inds, 1:]
            unions = boxes2union_single(p1_boxes, p2_boxes, 27)
            spatial_enc = self.conv(unions)
            spatial_enc = self.sp_fc(spatial_enc.view(-1, 256*7*7))
            x_in = torch.cat([p1_rep, p2_rep, vr_rep, spatial_enc], 1)
        else:
            x_in = torch.cat([p1_rep, p2_rep, vr_rep], 1)

        if self.use_attention:
            im_idx = rois[:, 0][p1_inds]
            x_in, _ = self.transformer(x_in, im_idx)
        
        cls_score = self.fc_cls(x_in)
        return cls_score
    
    @staticmethod
    def get_targets(sampling_results: List[HHISamplingResult],
                    rcnn_train_cfg: ConfigDict) -> tuple:
        pos_ind_list = [res.pos_inds for res in sampling_results]
        neg_ind_list = [res.neg_inds for res in sampling_results]
        label_list = [res.pos_pair_labels for res in sampling_results]
        cls_reg_targets = hhi_target(pos_ind_list, neg_ind_list, label_list, rcnn_train_cfg)
        return cls_reg_targets
    
    @staticmethod
    def get_recall_prec(pred_vec: Tensor, target_vec: Tensor) -> tuple:
        correct = pred_vec & target_vec
        recall = correct.sum(1) / target_vec.sum(1).float()  # Enforce Float
        prec = correct.sum(1) / (pred_vec.sum(1) + 1e-6)
        return recall.mean(), prec.mean()
    
    @staticmethod
    def topk_to_matrix(probs: Tensor, k: int) -> Tensor:
        """Converts top-k to binary matrix."""
        topk_labels = probs.topk(k, 1, True, True)[1]
        topk_matrix = probs.new_full(probs.size(), 0, dtype=torch.bool)
        for i in range(probs.shape[0]):
            topk_matrix[i, topk_labels[i]] = 1
        return topk_matrix
    
    def topk_accuracy(self,
                      pred: Tensor,
                      target: Tensor,
                      thr: float = 0.5) -> tuple:
        """Computes the Top-K Accuracies for both single and multi-label
        scenarios."""
        # Define Target vector:
        target_bool = target > 0.5

        # Branch on Multilabel for computing output classification
        if self.multilabel:
            pred = pred.sigmoid()
        else:
            pred = pred.softmax(dim=1)

        # Compute at threshold (K=1 for single)
        if self.multilabel:
            pred_bool = pred > thr
        else:
            pred_bool = self.topk_to_matrix(pred, 1)
        recall_thr, prec_thr = self.get_recall_prec(pred_bool, target_bool)

        # Compute at various K
        recalls_k, precs_k = [], []
        for k in self.topk:
            pred_bool = self.topk_to_matrix(pred, k)
            recall, prec = self.get_recall_prec(pred_bool, target_bool)
            recalls_k.append(recall)
            precs_k.append(prec)

        # Return all
        return recall_thr, prec_thr, recalls_k, precs_k
    
    def loss_and_target(self, cls_score: Tensor, rois: Tensor,
                        sampling_results: List[HHISamplingResult],
                        rcnn_train_cfg: ConfigDict, **kwargs) -> dict:
        cls_targets = self.get_targets(sampling_results, rcnn_train_cfg)
        labels, _ = cls_targets

        losses = dict()
        # Only use the cls_score
        if cls_score is not None:
            labels = labels[:,1:] # Get the valid labels (ignore first one)
            pos_inds = torch.sum(labels, dim=-1) > 0
            cls_score = cls_score[pos_inds, 1:]
            labels = labels[pos_inds]

            # Compute First Recall/Precisions
            #   This has to be done first before normalising the label-space.
            recall_thr, prec_thr, recall_k, prec_k = self.topk_accuracy(
                cls_score, labels, thr=0.5)
            losses['recall@thr=0.5'] = recall_thr
            losses['prec@thr=0.5'] = prec_thr
            for i, k in enumerate(self.topk):
                losses[f'recall@top{k}'] = recall_k[i]
                losses[f'prec@top{k}'] = prec_k[i]

            # If Single-label, need to ensure that target labels sum to 1: ie
            #   that they are valid probabilities.
            if not self.multilabel:
                labels = labels / labels.sum(dim=1, keepdim=True)

            # Select Loss function based on single/multi-label
            #   NB. Both losses auto-compute sigmoid/softmax on prediction
            if self.multilabel:
                loss_func = F.binary_cross_entropy_with_logits
            else:
                loss_func = cross_entropy_loss

            # Compute loss
            loss = loss_func(cls_score, labels, reduction='none')
            pt = torch.exp(-loss)
            F_loss = self.focal_alpha * (1 - pt)**self.focal_gamma * loss
            losses['loss_hhi_cls'] = torch.mean(F_loss)

        return dict(loss_bbox=losses, bbox_targets=cls_targets)
    
    def predict_by_feat(self,
                        rois: Tuple[Tensor],
                        cls_scores: Tuple[Tensor],
                        batch_img_metas: List[dict],
                        rcnn_test_cfg: Optional[ConfigDict] = None,
                        **kwargs) -> InstanceList:
        result_list = []
        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            results = self._predict_by_feat_single(
                roi=rois[img_id],
                cls_score=cls_scores[img_id],
                img_meta=img_meta,
                rcnn_test_cfg=rcnn_test_cfg,
                **kwargs
            )
            result_list.append(results)
       
        return result_list
    
    def _predict_by_feat_single(self,
                                roi: Tensor,
                                cls_score: Tensor,
                                img_meta: dict,
                                rcnn_test_cfg: Optional[ConfigDict] = None,
                                # **kwargs) -> InstanceData:
                                **kwargs) -> dict:
        # results = InstanceData()
        results = {}

        # might be used by testing w. augmentation
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        
        # Handle Multi/Single Label
        if cls_score is not None:
            if self.multilabel:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(dim=-1)
        else:
            scores = None

        bboxes = roi[:, 1:]
        assert bboxes.shape[-1] == 4

        # First reverse the flip
        img_h, img_w = img_meta['img_shape']
        if img_meta.get('flip', False):
            bboxes_ = bboxes.clone()
            bboxes_[:, 0] = img_w - 1 - bboxes[:, 2]
            bboxes_[:, 2] = img_w - 1 - bboxes[:, 0]
            bboxes = bboxes_

        # Then normalize the bbox to [0, 1]
        bboxes[:, 0::2] /= img_w
        bboxes[:, 1::2] /= img_h
        
        def _bbox_crop_undo(bboxes, crop_quadruple):
            decropped = bboxes.clone()

            if crop_quadruple is not None:
                x1, y1, tw, th = crop_quadruple
                decropped[:, 0::2] = bboxes[..., 0::2] * tw + x1
                decropped[:, 1::2] = bboxes[..., 1::2] * th + y1

            return decropped

        crop_quadruple = img_meta.get('crop_quadruple', np.array([0, 0, 1, 1]))
        bboxes = _bbox_crop_undo(bboxes, crop_quadruple)

        # results.bboxes = bboxes
        # results.scores = scores
        results['bboxes'] = bboxes
        results['scores'] = scores

        return results
    

if mmdet_imported:
    MMDET_MODELS.register_module()(BBoxHeadHHI_single_mask)