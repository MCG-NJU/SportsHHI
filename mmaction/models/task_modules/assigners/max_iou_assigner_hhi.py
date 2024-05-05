# from mmdet.models.task_modules.assigners.assign_result import AssignResult
from .assign_result_hhi import AssignResultHHI
import torch
from torch import Tensor

try:
    # from mmdet.models.task_modules import AssignResult, MaxIoUAssigner
    from mmdet.models.task_modules import MaxIoUAssigner
    from mmdet.registry import TASK_UTILS as MMDET_TASK_UTILS
    mmdet_imported = True
except (ImportError, ModuleNotFoundError):
    mmdet_imported = False

if mmdet_imported:
    @MMDET_TASK_UTILS.register_module()
    class MaxIoUAssignerHHI(MaxIoUAssigner):
        def assign_wrt_overlaps(self, overlaps: Tensor,
                                gt_labels: Tensor) -> AssignResultHHI:
            num_gts, num_bboxes = overlaps.size(0), overlaps.size(1)
            num_classes = gt_labels.size(1) - 2

            gt_bbox_labels = []
            for gt_bbox_ind in range(1, num_gts+1):
                selected_inds = [i for i in range(len(gt_labels))
                if gt_labels[i][0] == gt_bbox_ind or gt_labels[i][1] == gt_bbox_ind]
                labels = torch.zeros(num_classes, device=gt_labels.device)
                for ind in selected_inds:
                    labels = torch.logical_or(labels, gt_labels[ind,2:])
                gt_bbox_labels.append(labels)
            gt_bbox_labels = torch.stack(gt_bbox_labels).float()

            # 1. assign -1 by default
            assigned_gt_inds = overlaps.new_full((num_bboxes, ),
                                                 -1,
                                                 dtype=torch.long)

            if num_gts == 0 or num_bboxes == 0:
                # No ground truth or boxes, return empty assignment
                max_overlaps = overlaps.new_zeros((num_bboxes, ))
                if num_gts == 0:
                    # No truth, assign everything to background
                    assigned_gt_inds[:] = 0
                if gt_labels is None:
                    assigned_labels = None
                else:
                    assigned_labels = overlaps.new_full((num_bboxes, ),
                                                        -1,
                                                        dtype=torch.long)
                return AssignResultHHI(
                    num_gts,
                    assigned_gt_inds,
                    max_overlaps,
                    labels=assigned_labels)
            
            # for each anchor, which gt best overlaps with it
            # for each anchor, the max iou of all gts
            max_overlaps, argmax_overlaps = overlaps.max(dim=0)
            # for each gt, which anchor best overlaps with it
            # for each gt, the max iou of all proposals
            gt_max_overlaps, gt_argmax_overlaps = overlaps.max(dim=1)

            assigned_bbox_gt_inds = overlaps.new_full((num_bboxes, ),
                                                 -1,
                                                 dtype=torch.long)
            # 2. assign negative: below
            # the negative inds are set to be 0
            if isinstance(self.neg_iou_thr, float):
                assigned_bbox_gt_inds[(max_overlaps >= 0)
                                 & (max_overlaps < self.neg_iou_thr)] = 0
            elif isinstance(self.neg_iou_thr, tuple):
                assert len(self.neg_iou_thr) == 2
                assigned_bbox_gt_inds[(max_overlaps >= self.neg_iou_thr[0])
                                 & (max_overlaps < self.neg_iou_thr[1])] = 0

            # 3. assign positive: above positive IoU threshold
            pos_inds = max_overlaps >= self.pos_iou_thr
            assigned_bbox_gt_inds[pos_inds] = argmax_overlaps[pos_inds] + 1

            if self.match_low_quality:
                # Low-quality matching will overwrite the assigned_gt_inds
                # assigned in Step 3. Thus, the assigned gt might not be the
                # best one for prediction.
                # For example, if bbox A has 0.9 and 0.8 iou with GT bbox
                # 1 & 2, bbox 1 will be assigned as the best target for bbox A
                # in step 3. However, if GT bbox 2's gt_argmax_overlaps = A,
                # bbox A's assigned_gt_inds will be overwritten to be bbox B.
                # This might be the reason that it is not used in ROI Heads.
                for i in range(num_gts):
                    if gt_max_overlaps[i] >= self.min_pos_iou:
                        if self.gt_max_assign_all:
                            max_iou_inds = overlaps[i, :] == gt_max_overlaps[i]
                            assigned_bbox_gt_inds[max_iou_inds] = i + 1
                        else:
                            assigned_bbox_gt_inds[gt_argmax_overlaps[i]] = i + 1
            if gt_labels is not None:
                assert len(gt_labels[0]) > 2
                assigned_labels = assigned_gt_inds.new_zeros(
                    (num_bboxes, len(gt_labels[0])-2), dtype=torch.float32)
                pos_inds = torch.nonzero(
                    assigned_bbox_gt_inds > 0, as_tuple=False).squeeze()
                if pos_inds.numel() > 0:
                    assigned_labels[pos_inds] = gt_bbox_labels[
                        assigned_bbox_gt_inds[pos_inds]-1]
            else:
                assigned_labels = None

            return AssignResultHHI(
                num_gts,
                assigned_bbox_gt_inds,
                max_overlaps,
                labels=assigned_labels
            )
        

else:
    # define an empty class, so that can be imported
    class MaxIoUAssignerAVA:

        def __init__(self, *args, **kwargs):
            raise ImportError(
                'Failed to import `AssignResult`, `MaxIoUAssigner` from '
                '`mmdet.core.bbox` or failed to import `TASK_UTILS` from '
                '`mmdet.registry`. The class `MaxIoUAssignerHHI` is '
                'invalid. ')