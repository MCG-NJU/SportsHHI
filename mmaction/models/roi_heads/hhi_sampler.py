from abc import ABCMeta, abstractmethod
import torch
from mmdet.utils import util_mixins
import numpy as np

class HHISamplingResult(util_mixins.NiceRepr):
    def __init__(self, pos_inds, neg_inds, p1_bbox_inds, p2_bbox_inds, pair_labels, bboxes):
        self.pos_inds = pos_inds
        self.neg_inds = neg_inds
        self.pos_p1_bbox_inds = p1_bbox_inds[pos_inds]
        self.pos_p2_bbox_inds = p2_bbox_inds[pos_inds]
        self.neg_p1_bbox_inds = p1_bbox_inds[neg_inds]
        self.neg_p2_bbox_inds = p2_bbox_inds[neg_inds]
        self.pos_pair_labels = pair_labels[pos_inds]
        self.bboxes = bboxes

    @property
    def p1_bbox_inds(self):
        return torch.cat([self.pos_p1_bbox_inds, self.neg_p1_bbox_inds])
    
    @property
    def p2_bbox_inds(self):
        return torch.cat([self.pos_p2_bbox_inds, self.neg_p2_bbox_inds])
    
    def to(self, device):
        _dict = self.__dict__
        for key, value in _dict.items():
            if isinstance(value, torch.Tensor):
                _dict[key] = value.to(device)
        return self
    
    def __nice__(self):
        data = self.info.copy()
        data['pos_pair_labels'] = data.pop('pos_pair_labels').shape
        data['bboxes'] = data.pop('bboxes').shape
        parts = [f"'{k}': {v!r}" for k, v in sorted(data.items())]
        body = '    ' + ',\n    '.join(parts)
        return '{\n' + body + '\n}'
    
    @property
    def info(self):
        return {
            'pos_inds': self.pos_inds,
            'neg_inds': self.neg_inds,
            'pos_p1_bbox_inds': self.pos_p1_bbox_inds,
            'pos_p2_bbox_inds': self.pos_p2_bbox_inds,
            'neg_p1_bbox_inds': self.neg_p1_bbox_inds,
            'neg_p2_bbox_inds': self.neg_p2_bbox_inds,
            'pos_pair_labels': self.pos_pair_labels,
            'bboxes': self.bboxes,
        }
    
class HHIBaseSampler(metaclass=ABCMeta):
    def __init__(self,
                 num,
                 pos_fraction,
                 neg_pos_ub=-1,
                 add_gt_as_proposals=True):
        self.num = num
        self.pos_fraction = pos_fraction
        self.neg_pos_ub = neg_pos_ub
        self.add_gt_as_proposals = add_gt_as_proposals
        self.pos_sampler = self
        self.neg_sampler = self

    @abstractmethod
    def _sample_pos(self, assign_result, num_expected, **kwargs):
        """Sample positive samples."""
        pass

    @abstractmethod
    def _sample_neg(self, assign_result, num_expected, **kwargs):
        """Sample negative samples"""
        pass

    def sample(self,
               assign_result,
               bboxes,
               gt_p1_ids,
               gt_p2_ids,
               gt_bboxes,
               gt_labels,
               **kwargs):
        """Sample positive records
        """
        gt_labels = np.concatenate((gt_p1_ids.reshape(-1,1), gt_p2_ids.reshape(-1,1), gt_labels), axis=1)
        gt_labels = torch.from_numpy(gt_labels).to(dtype=gt_bboxes.dtype, device=gt_bboxes.device)

        bboxes = bboxes.priors
        if len(bboxes.shape) < 2:
            bboxes = bboxes[None, :]
        
        bboxes = bboxes[:, :4]

        gt_bbox_labels = torch.zeros((gt_bboxes.shape[0], gt_labels.shape[1]-2), device=gt_bboxes.device)
        for row in gt_labels:
            p1, p2 = int(row[0]), int(row[1])
            gt_bbox_labels[p1] = torch.logical_or(gt_bbox_labels[p1], row[2:])
            gt_bbox_labels[p2] = torch.logical_or(gt_bbox_labels[p2], row[2:])

        gt_flags = bboxes.new_zeros((bboxes.shape[0],), dtype=torch.uint8)
        if self.add_gt_as_proposals and len(gt_bboxes) > 0:
            if gt_labels is None:
                raise ValueError('gt_labels must be given when add_gt_as_proposals is True')
            bboxes = torch.cat([gt_bboxes, bboxes], dim=0)
            assign_result.add_gt_(gt_bbox_labels)
            gt_ones = bboxes.new_ones(gt_bboxes.shape[0], dtype=torch.uint8)
            gt_flags = torch.cat([gt_ones, gt_flags])
        
        num_bboxes = len(bboxes)
        bbox_inds = torch.tensor([i for i in range(num_bboxes)])
        p1_bbox_inds = torch.cat([bbox_ind.repeat(num_bboxes) for bbox_ind in bbox_inds], 0)
        p2_bbox_inds = bbox_inds.repeat(num_bboxes)

        p1_inds = torch.cat([i.repeat(num_bboxes) for i in assign_result.gt_inds])
        p2_inds = assign_result.gt_inds.repeat(num_bboxes)

        assert p1_bbox_inds.shape[0] == p2_bbox_inds.shape[0]
        assert p1_bbox_inds.shape[0] == num_bboxes * num_bboxes
        assert p1_bbox_inds.shape[0] == p1_inds.shape[0]
        assert p2_bbox_inds.shape[0] == p2_inds.shape[0]

        assign_gt_record_inds = torch.zeros(num_bboxes * num_bboxes, device=gt_labels.device)
        pair_labels = torch.zeros((num_bboxes*num_bboxes, gt_labels.shape[1]-2), device=gt_labels.device)
        for i in range(len(gt_labels)):
            row = gt_labels[i]
            p1, p2 = row[0]+1, row[1]+1
            inds = torch.nonzero(torch.logical_and(p1_inds==p1, p2_inds==p2))
            assign_gt_record_inds[inds] = i + 1
            pair_labels[inds] = row[2:]
        
        # num_pairs = self.num * self.num
        num_expected_pos = int(self.num * self.num * self.pos_fraction)
        pos_inds = self.pos_sampler._sample_pos(
            assign_gt_record_inds, num_expected_pos, bboxes=bboxes, **kwargs
        )
        pos_inds = pos_inds.unique()
        num_sampled_pos = pos_inds.numel()
        num_expected_neg = self.num * self.num - num_sampled_pos
        if self.neg_pos_ub >= 0:
            _pos = max(1, num_sampled_pos)
            neg_upper_bound = int(self.neg_pos_ub * _pos)
            if num_expected_neg > neg_upper_bound:
                num_expected_neg = neg_upper_bound
        neg_inds = self.neg_sampler._sample_neg(
            assign_gt_record_inds, num_expected_neg, bboxes=bboxes, **kwargs
        )
        neg_inds = neg_inds.unique()
        
        return HHISamplingResult(pos_inds, neg_inds, p1_bbox_inds, p2_bbox_inds, pair_labels, bboxes)
    

class HHIRandomSampler(HHIBaseSampler):
    def __init__(self,
                 num,
                 pos_fraction,
                 neg_pos_ub=-1,
                 add_gt_as_proposals=True):
        super(HHIRandomSampler, self).__init__(num, pos_fraction, neg_pos_ub, add_gt_as_proposals)

    def random_choice(self, gallery, num):
        assert len(gallery) >= num
        is_tensor = isinstance(gallery, torch.Tensor)
        if not is_tensor:
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
            else:
                device = 'cpu'
            gallery = torch.tensor(gallery, dtype=torch.long, device=device)
        perm = torch.randperm(gallery.numel())[:num].to(device=gallery.device)
        rand_inds = gallery[perm]
        if not is_tensor:
            rand_inds = rand_inds.cpu().numpy()
        return rand_inds

    def _sample_pos(self, assign_gt_record_inds, num_expected, **kwargs):
        pos_inds = torch.nonzero(assign_gt_record_inds>0, as_tuple=False)
        if pos_inds.numel() != 0:
            pos_inds = pos_inds.squeeze(1)
        if pos_inds.numel() <= num_expected:
            return pos_inds
        else:
            return self.random_choice(pos_inds, num_expected)
    
    def _sample_neg(self, assign_gt_record_inds, num_expected, **kwargs):
        neg_inds = torch.nonzero(assign_gt_record_inds==0, as_tuple=False)
        if neg_inds.numel() != 0:
            neg_inds = neg_inds.squeeze(1)
        if len(neg_inds) <= num_expected:
            return neg_inds
        else:
            return self.random_choice(neg_inds, num_expected)