import torch
from torch import Tensor

from mmdet.utils import util_mixins


class AssignResultHHI(util_mixins.NiceRepr):
    def __init__(self, num_gts: int, gt_inds: Tensor,
                 max_overlaps: Tensor, labels: Tensor) -> None:
        self.num_gts = num_gts
        self.gt_inds = gt_inds
        self.max_overlaps = max_overlaps
        self.labels = labels
        # Interface for possible user-defined properties
        self._extra_properties = {}

    @property
    def num_preds(self):
        return len(self.gt_inds)
    
    def set_extra_property(self, key, value):
        assert key not in self.info
        self._extra_properties[key] = value

    def get_extra_property(self, key):
        return self._extra_properties.get(key, None)
    
    @property
    def info(self):
        basic_info = {
            'num_gts': self.num_gts,
            'num_preds': self.num_preds,
            'gt_inds': self.gt_inds,
            'max_overlaps': self.max_overlaps,
            'labels': self.labels
        }
        basic_info.update(self._extra_properties)
        return basic_info
    
    def __nice__(self):
        parts = []
        parts.append(f'num_gts={self.num_gts!r}')
        if self.gt_inds is None:
            parts.append(f'gt_inds={self.gt_inds!r}')
        else:
            parts.append(f'gt_inds.shape={tuple(self.gt_inds.shape)!r}')
        if self.max_overlaps is None:
            parts.append(f'max_overlaps={self.max_overlaps!r}')
        else:
            parts.append('max_overlaps.shape='
                         f'{tuple(self.max_overlaps.shape)!r}')
        if self.labels is None:
            parts.append(f'labels={self.labels!r}')
        else:
            parts.append(f'labels.shape={tuple(self.labels.shape)!r}')
        return ', '.join(parts)
    
    @classmethod
    def random(cls, **kwargs):
        raise NotImplementedError("Should not use random for HHI Dataset.")
    
    def add_gt_(self, gt_labels):
        self_inds = torch.arange(
            1, len(gt_labels) + 1, dtype=torch.long,
            device=gt_labels.device
        )

        self.gt_inds = torch.cat([self_inds, self.gt_inds])

        self.max_overlaps = torch.cat(
            [self.max_overlaps.new_ones(len(gt_labels)), self.max_overlaps])

        if self.labels.size(1) != gt_labels.size(1):
            assert gt_labels.size(1) == self.labels.size(1) + 2
            self.labels = torch.cat([gt_labels[:,2:], self.labels])

        elif self.labels is not None:
            self.labels = torch.cat([gt_labels, self.labels])