import os.path as osp
from collections import defaultdict
from typing import Callable, List, Optional, Union

import numpy as np
from mmengine.fileio import exists, list_from_file, load
from mmengine.logging import MMLogger

from mmaction.evaluation import read_labelmap
from mmaction.registry import DATASETS
from mmaction.utils import ConfigType
from .base import BaseActionDataset


@DATASETS.register_module()
class SportsHHIDataset(BaseActionDataset):
    def __init__(self,
                 ann_file: str,
                 exclude_file: str,
                 pipeline: List[Union[ConfigType, Callable]],
                 label_file: str,
                 filename_tmpl: str = '{:06}.jpg',
                 start_index: int = 0,
                 proposal_file: str = None,
                 person_det_score_thr: float = 0.9,
                 num_classes: int = 34,
                 custom_classes: Optional[List[int]] = None,
                 data_prefix: ConfigType = dict(img=''),
                 modality: str = 'RGB',
                 test_mode: bool = False,
                 num_max_proposals: int = 10000,
                 timestamp_start: int = 0,
                 timestamp_end: int = 900,
                 fps: int = 5,
                 **kwargs) -> None:
        self._FPS = fps # number of frames between keyframes, not extracting fps
        self.custom_classes = custom_classes
        if custom_classes is not None:
            assert num_classes == len(custom_classes) + 1
            assert 0 not in custom_classes
            _, class_white_list = read_labelmap(open(label_file))
            assert set(custom_classes).issubset(class_white_list)
            self.custom_classes = list([0] + custom_classes)
        self.exclude_file = exclude_file
        self.label_file = label_file
        self.proposal_file = proposal_file
        assert 0 <= person_det_score_thr <= 1, (
            'The value of '
            'person_det_score_thr should in [0,1]. '
        )
        self.person_det_score_thr = person_det_score_thr
        self.timestamp_start = timestamp_start
        self.timestamp_end = timestamp_end
        self.num_max_proposals = num_max_proposals
        self.filename_tmpl = filename_tmpl

        super().__init__(
            ann_file,
            pipeline=pipeline,
            data_prefix=data_prefix,
            test_mode=test_mode,
            num_classes=num_classes,
            start_index=start_index,
            modality=modality,
            **kwargs
        )

        if self.proposal_file is not None:
            self.proposals = load(self.proposal_file)
        else:
            self.proposals = None

    def parse_img_bboxes(self, bbox_list):
        bboxes, labels = [], []
        while len(bbox_list) > 0:
            bbox = bbox_list[0]
            num_bboxes = len(bbox_list)

            selected_bboxes = [
                x for x in bbox_list
                if np.array_equal(x['bbox'], bbox['bbox'])
            ]

            num_selected_bboxes = len(selected_bboxes)
            bbox_list = [
                x for x in bbox_list if 
                not np.array_equal(x['bbox'], bbox['bbox'])
            ]

            assert len(bbox_list) + num_selected_bboxes == num_bboxes

            bboxes.append(bbox['bbox'])
            valid_labels = np.array([
                selected_bbox['label']
                for selected_bbox in selected_bboxes
            ])

            label = np.zeros(self.num_classes, dtype=np.float32)
            label[valid_labels] = 1

            labels.append(label)

        bboxes = np.stack(bboxes)
        labels = np.stack(labels)
        return bboxes, labels

    def parse_img_record(self, img_records: List[dict], bboxes:List[dict]=None) -> tuple:
        p1_boxes, p2_boxes, labels, p1_ids, p2_ids = [], [], [], [], []
        while len(img_records) > 0:
            img_record = img_records[0]
            num_img_records = len(img_records)

            selected_records = [
                x for x in img_records
                if np.array_equal(x['p1_box'], img_record['p1_box'])
                and np.array_equal(x['p2_box'], img_record['p2_box'])
            ]

            num_selected_records = len(selected_records)
            if num_selected_records != 1:
                raise ValueError('Duplicate records in SportsHHI Dataset.')
            img_records = [
                x for x in img_records if 
                not (np.array_equal(x['p1_box'], img_record['p1_box'])
                     and np.array_equal(x['p2_box'], img_record['p2_box']))
            ]

            assert len(img_records) + num_selected_records == num_img_records

            p1_boxes.append(img_record['p1_box'])
            p2_boxes.append(img_record['p2_box'])
            valid_labels = np.array([
                selected_record['label'] for selected_record in selected_records
            ])
            
            label = np.zeros(self.num_classes, dtype=np.float32)
            label[valid_labels] = 1
            labels.append(label)

            p1_ind, p2_ind = -1, -1

            for ind in range(len(bboxes)):
                if np.array_equal(bboxes[ind], img_record['p1_box']):
                    p1_ind = ind
                    break
            assert p1_ind > -1

            for ind in range(len(bboxes)):
                if np.array_equal(bboxes[ind], img_record['p2_box']):
                    p2_ind = ind
                    break
            assert p2_ind > -1

            p1_ids.append(p1_ind)
            p2_ids.append(p2_ind)

        p1_boxes = np.stack(p1_boxes)
        p2_boxes = np.stack(p2_boxes)
        labels = np.stack(labels)
        p1_ids = np.stack(p1_ids)
        p2_ids = np.stack(p2_ids)
        return p1_boxes, p2_boxes, labels, p1_ids, p2_ids
    
    def filter_data(self) -> List[dict]:
        """Filter out records in the excluded_file."""
        valid_indexes = []
        if self.exclude_file is None:
            valid_indexes = list(range(len(self.data_list)))
        else:
            excluded_video_infos = [
                x.strip().split(',') for x in open(self.exclude_file)
            ]
            for i, data_info in enumerate(self.data_list):
                valid_indexes.append(i)
                for video_id, timestamp in excluded_video_infos:
                    if (data_info['video_id'] == video_id
                            and data_info['timestamp'] == int(timestamp)):
                        valid_indexes.pop()
                        break
        
        logger = MMLogger.get_current_instance()
        logger.info(f'{len(valid_indexes)} out of {len(self.data_list)}'
                    f' frames are valid.')
        data_list = [self.data_list[i] for i in valid_indexes]

        return data_list
    
    def load_data_list(self) -> List[dict]:
        """Load HHI annotations"""
        exists(self.ann_file)
        data_list = []
        records_dict_by_img = defaultdict(list)
        bboxes_dict_by_img = defaultdict(list)
        fin = list_from_file(self.ann_file)
        for line in fin:
            line_split = line.strip().split(',')
            # [vid, fid, x11, y11, x12, y12, x21, y21, x22, y22, label, p1, p2]
            assert len(line_split) == 13
            label = int(line_split[10])
            if self.custom_classes is not None:
                if label not in self.custom_classes:
                    continue
                label = self.custom_classes.index(label)

            video_id = line_split[0]
            # line_split[1]: fid (NOT TIMESTAMP)
            timestamp = int(line_split[1]) // self._FPS
            img_key = f'{video_id},{timestamp:04d}'

            p1_box = np.array(list(map(float, line_split[2:6])))
            p1_id = int(line_split[-2])
            p2_box = np.array(list(map(float, line_split[6:10])))
            p2_id = int(line_split[-1])
            shot_info = (1, (self.timestamp_end-self.timestamp_start)*self._FPS + 1) # starts from 1 in newer version of mmaction, 0 in older version

            video_info = dict(
                video_id=video_id,
                timestamp = timestamp,
                p1_box=p1_box,
                p2_box=p2_box,
                p1_id=p1_id,
                label=label,
                shot_info=shot_info
            )
            video_box_1 = dict(
                bbox=p1_box,
                label=label
            )
            video_box_2 = dict(
                bbox=p2_box,
                label=label
            )
            records_dict_by_img[img_key].append(video_info)
            bboxes_dict_by_img[img_key].append(video_box_1)
            bboxes_dict_by_img[img_key].append(video_box_2)
        
        for img_key in records_dict_by_img:
            video_id, timestamp = img_key.split(',')
            start, end = self.timestamp_start, self.timestamp_end
            bboxes, labels = self.parse_img_bboxes(
                bboxes_dict_by_img[img_key]
            )
            p1_boxes, p2_boxes, interactions, p1_ids, p2_ids = self.parse_img_record(
                records_dict_by_img[img_key],
                bboxes
            )
            ann = dict(
                gt_bboxes=bboxes,
                gt_labels=labels,
                gt_interactions=interactions,
                p1_ids=p1_ids,
                p2_ids=p2_ids
            )
            frame_dir = video_id
            if self.data_prefix['img'] is not None:
                frame_dir = osp.join(self.data_prefix['img'], frame_dir)
            video_info = dict(
                frame_dir=frame_dir,
                video_id=video_id,
                timestamp=int(timestamp),
                timestamp_start=start,
                timestamp_end=end,
                img_key=img_key,
                shot_info=shot_info,
                fps=self._FPS,
                ann=ann
            )
            data_list.append(video_info)

        return data_list
    
    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index"""
        data_info = super().get_data_info(idx)
        img_key = data_info['img_key']
        data_info['filename_tmpl'] = self.filename_tmpl
        if 'timestamp_start' not in data_info:
            data_info['timestamp_start'] = self.timestamp_start
            data_info['timestamp_end'] = self.timestamp_end

        if self.proposals is not None:
            proposal_key = img_key
            if proposal_key not in self.proposals:
                print(f'WARNING: {img_key} not found in proposals')
                data_info['proposals'] = np.array([[0, 0, 1, 1]])
                data_info['scores'] = np.array([1])
            else:
                proposals = self.proposals[proposal_key]
                assert proposals.shape[-1] in [4, 5]
                if proposals.shape[-1] == 5:
                    thr = min(self.person_det_score_thr, max(proposals[:,4]))
                    positive_inds = (proposals[:,4] >= thr)
                    proposals = proposals[positive_inds]
                    proposals = proposals[:self.num_max_proposals]
                    data_info['proposals'] = proposals[:,:4]
                    data_info['scores'] = proposals[:,4]
                else:
                    proposals = proposals[:self.num_max_proposals]
                    data_info['proposals'] = proposals
        
        ann = data_info.pop('ann')
        data_info['gt_bboxes'] = ann['gt_bboxes']
        data_info['gt_labels'] = ann['gt_labels']
        data_info['gt_interactions'] = ann['gt_interactions']
        data_info['p1_ids'] = ann['p1_ids']
        data_info['p2_ids'] = ann['p2_ids']
        
        return data_info