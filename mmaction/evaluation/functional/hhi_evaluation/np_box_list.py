import numpy as np


class BoxList:
    """Box collection.

    BoxList represents a list of bounding boxes as numpy array, where each
    bounding box is represented as a row of 4 numbers,
    [y_min, x_min, y_max, x_max].  It is assumed that all bounding boxes within
    a given list correspond to a single image.

    Optionally, users can add additional related fields (such as
    objectness/classification scores).
    """

    def __init__(self, data):
        """Constructs box collection.

        Args:
            data: a numpy array of shape [N, 4] representing box coordinates

        Raises:
            ValueError: if bbox data is not a numpy array
            ValueError: if invalid dimensions for bbox data
        """
        if not isinstance(data, np.ndarray):
            raise ValueError('data must be a numpy array.')
        if len(data.shape) != 2 or data.shape[1] != 4:
            raise ValueError('Invalid dimensions for box data.')
        if data.dtype != np.float32 and data.dtype != np.float64:
            raise ValueError(
                'Invalid data type for box data: float is required.')
        if not self._is_valid_boxes(data):
            raise ValueError('Invalid box data. data must be a numpy array of '
                             'N*[y_min, x_min, y_max, x_max]')
        self.data = {'boxes': data}

    def num_boxes(self):
        return self.data['boxes'].shape[0]
    
    def get_extra_fields(self):
        return [k for k in self.data if k != 'boxes']

    def has_field(self, field):
        return field in self.data
    
    def add_field(self, field, field_data):
        if self.has_field(field):
            raise ValueError('Field ' + field + 'already exists')
        if len(field_data.shape) < 1 or field_data.shape[0] != self.num_boxes():
            raise ValueError('Invalid dimensions for field data')
        self.data[field] = field_data
    
    def get_field(self, field):
        if not self.has_field(field):
            raise ValueError(f'field {field} does not exist')
        return self.data[field]

    def get(self):
        return self.get_field('boxes')

    def get_coordinates(self):
        """Get corner coordinates of boxes.

        Returns:
            a list of 4 1-d numpy arrays [y_min, x_min, y_max, x_max]
        """
        box_coordinates = self.get()
        y_min = box_coordinates[:, 0]
        x_min = box_coordinates[:, 1]
        y_max = box_coordinates[:, 2]
        x_max = box_coordinates[:, 3]
        return [y_min, x_min, y_max, x_max]

    @staticmethod
    def _is_valid_boxes(data):
        """Check whether data fulfills the format of N*[ymin, xmin, ymax,
        xmin].

        Args:
            data: a numpy array of shape [N, 4] representing box coordinates

        Returns:
            a boolean indicating whether all ymax of boxes are equal or greater
            than ymin, and all xmax of boxes are equal or greater than xmin.
        """
        if len(data) != 0:
            for v in data:
                if v[0] > v[2] or v[1] > v[3]:
                    return False
        return True