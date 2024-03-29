import tensorflow as tf
import numpy as np


class AnchorGenerator(object):
    """Standard anchor generator for 2D anchor-based detectors.
    Args:
        strides (list[int] | list[tuple[int, int]]): Strides of anchors
            in multiple feature levels in order (w, h).
        ratios (list[float]): The list of ratios between the height and width
            of anchors in a single level.
        scales (list[int] | None): Anchor scales for anchors in a single level.
            It cannot be set at the same time if `octave_base_scale` and
            `scales_per_octave` are set.
        base_sizes (list[int] | None): The basic sizes
            of anchors in multiple levels.
            If None is given, strides will be used as base_sizes.
            (If strides are non square, the shortest stride is taken.)
        scale_major (bool): Whether to multiply scales first when generating
            base anchors. If true, the anchors in the same row will have the
            same scales. By default it is True in V2.0
        octave_base_scale (int): The base scale of octave.
        scales_per_octave (int): Number of scales for each octave.
            `octave_base_scale` and `scales_per_octave` are usually used in
            retinanet and the `scales` should be None when they are set.
        centers (list[tuple[float, float]] | None): The centers of the anchor
            relative to the feature grid center in multiple feature levels.
            By default it is set to be None and not used. If a list of tuple of
            float is given, they will be used to shift the centers of anchors.
        center_offset (float): The offset of center in proportion to anchors'
            width and height. By default it is 0 in V2.0.

    Examples:
        >>> from mmdet.core import AnchorGenerator
        >>> self = AnchorGenerator([16], [1.], [1.], [9])
        >>> all_anchors = self.grid_anchors([(2, 2)], device='cpu')
        >>> print(all_anchors)
        [tensor([[-4.5000, -4.5000,  4.5000,  4.5000],
                [11.5000, -4.5000, 20.5000,  4.5000],
                [-4.5000, 11.5000,  4.5000, 20.5000],
                [11.5000, 11.5000, 20.5000, 20.5000]])]
        >>> self = AnchorGenerator([16, 32], [1.], [1.], [9, 18])
        >>> all_anchors = self.grid_anchors([(2, 2), (1, 1)], device='cpu')
        >>> print(all_anchors)
        [tensor([[-4.5000, -4.5000,  4.5000,  4.5000],
                [11.5000, -4.5000, 20.5000,  4.5000],
                [-4.5000, 11.5000,  4.5000, 20.5000],
                [11.5000, 11.5000, 20.5000, 20.5000]]), \
        tensor([[-9., -9., 9., 9.]])]
    """

    def __init__(self,
                 strides,
                 ratios,
                 scales=None,
                 base_sizes=None,
                 scale_major=True,
                 octave_base_scale=None,
                 scales_per_octave=None,
                 centers=None,
                 center_offset=0.):

        # check center and center_offset

        if center_offset != 0:
            assert centers is None, 'center cannot be set when center_offset' \
                f'!=0, {centers} is given.'
        if not (0 <= center_offset <= 1):
            raise ValueError('center_offset should be in range [0, 1], '
                             f'{center_offset} is given.')
        if centers is not None:
            assert len(centers) == len(strides), \
                'The number of strides should be the same as centers, got ' \
                f'{strides} and {centers}'
        # calculate base sizes of anchors
        self.strides = [tuple([stride, stride]) for stride in strides]

        self.base_sizes = [min(stride) for stride in self.strides
                           ] if base_sizes is None else base_sizes

        # base size and strides
        assert len(self.base_sizes) == len(self.strides), \
            'The number of strides should be the same as base sizes, got ' \
            f'{self.strides} and {self.base_sizes}'

        # calculate scales of anchors
        assert ((octave_base_scale is not None
                and scales_per_octave is not None) ^ (scales is not None)), \
            'scales and octave_base_scale with scales_per_octave cannot' \
            ' be set at the same time'

        if scales is not None:
            self.scales = tf.cast(scales, dtype=tf.dtypes.float32)
        elif octave_base_scale is not None and scales_per_octave is not None:
            octave_scales = np.array(
                [2**(i / scales_per_octave) for i in range(scales_per_octave)])
            scales = octave_scales * octave_base_scale
            self.scales = tf.cast(scales, dtype=tf.dtypes.float32)
        else:
            raise ValueError('Either scales or octave_base_scale with '
                             'scales_per_octave should be set')

        self.octave_base_scale = octave_base_scale
        self.scales_per_octave = scales_per_octave
        self.ratios = tf.cast(ratios, dtype=tf.dtypes.float32)
        self.scale_major = scale_major
        self.centers = centers
        self.center_offset = center_offset
        self.base_anchors = self.gen_base_anchors()

    def gen_base_anchors(self):
        """Generate base anchors.
        Returns:
            list(torch.Tensor): Base anchors of a feature grid in multiple \
                feature levels.
        """
        multi_level_base_anchors = []

        for i, base_size in enumerate(self.base_sizes):
            center = None
            if self.centers is not None:
                center = self.centers[i]
            multi_level_base_anchors.append(
                self.gen_single_level_base_anchors(base_size,
                                                   scales=self.scales,
                                                   ratios=self.ratios,
                                                   center=center))

        return multi_level_base_anchors

    def gen_single_level_base_anchors(self,
                                      base_size,
                                      scales,
                                      ratios,
                                      center=None):
        """Generate base anchors of a single level.

        Args:
            base_size (int | float): Basic size of an anchor.
            scales (torch.Tensor): Scales of the anchor.
            ratios (torch.Tensor): The ratio between between the height
                and width of anchors in a single level.
            center (tuple[float], optional): The center of the base anchor
                related to a single feature grid. Defaults to None.
        Returns:
            torch.Tensor: Anchors in a single-level feature maps.
        """
        w = base_size
        h = base_size
        if center is None:
            x_center = self.center_offset * w
            y_center = self.center_offset * h
        else:
            x_center, y_center = center
        h_ratios = tf.math.sqrt(ratios)
        w_ratios = 1 / h_ratios
        if self.scale_major:
            ws = tf.reshape((w * w_ratios[:, None] * scales[None, :]), [-1])
            hs = tf.reshape((h * h_ratios[:, None] * scales[None, :]), [-1])
        else:
            ws = tf.reshape((w * scales[:, None] * w_ratios[None, :]), [-1])
            hs = tf.reshape((h * scales[:, None] * h_ratios[None, :]), [-1])

        # use float anchor and the anchor's center is aligned with the
        # pixel center
        base_anchors = [
            x_center - 0.5 * ws, y_center - 0.5 * hs, x_center + 0.5 * ws,
            y_center + 0.5 * hs
        ]
        base_anchors = tf.stack(base_anchors, axis=-1)
        return base_anchors

    def grid_anchors(self, batch, featmap_sizes):
        """Generate grid anchors in multiple feature levels.

        Args:
            featmap_sizes (list[tuple]): List of feature map sizes in
                multiple feature levels.
            device (str): Device where the anchors will be put on.

        Return:
            list[torch.Tensor]: Anchors in multiple feature levels. \
                The sizes of each tensor should be [N, 4], where \
                N = width * height * num_base_anchors, width and height \
                are the sizes of the corresponding feature level, \
                num_base_anchors is the number of anchors for that level.
        """

        assert self.num_levels == len(featmap_sizes)
        multi_level_anchors, num_level_anchors = [], []

        for i in range(self.num_levels):
            anchors = self.single_level_grid_anchors(self.base_anchors[i],
                                                     featmap_sizes[i],
                                                     self.strides[i])
            num_level_anchors.append(tf.shape(anchors)[0])
            anchors = tf.tile(anchors[None, :, :], [batch, 1, 1])
            multi_level_anchors.append(anchors)
        # xxxx
        num_level_anchors = tf.tile(
            tf.cast(num_level_anchors, tf.int32)[None, :], [batch, 1])
        return multi_level_anchors, num_level_anchors

    def single_level_grid_anchors(self,
                                  base_anchors,
                                  featmap_size,
                                  stride=(16, 16)):
        """Generate grid anchors of a single level.

        Note:
            This function is usually called by method ``self.grid_anchors``.

        Args:
            base_anchors (torch.Tensor): The base anchors of a feature grid.
            featmap_size (tuple[int]): Size of the feature maps.
            stride (tuple[int], optional): Stride of the feature map in order
                (w, h). Defaults to (16, 16).
            device (str, optional): Device the tensor will be put on.
                Defaults to 'cuda'.

        Returns:
            torch.Tensor: Anchors in the overall feature maps.
        """

        feat_h = featmap_size[0]
        feat_w = featmap_size[1]
        shift_x = tf.range(0, feat_w) * stride[0]
        shift_y = tf.range(0, feat_h) * stride[1]
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        shifts = tf.stack([shift_xx, shift_yy, shift_xx, shift_yy], axis=-1)
        # first feat_w elements correspond to the first row of shifts
        # add A anchors (1, A, 4) to K shifts (K, 1, 4) to get
        # shifted anchors (K, A, 4), reshape to (K*A, 4)
        all_anchors = base_anchors[None, :, :] + tf.cast(
            shifts[:, None, :], tf.float32)
        all_anchors = tf.reshape(all_anchors, [-1, 4])
        # first A rows correspond to A anchors of (0, 0) in feature map,
        # then (0, 1), (0, 2), ...
        return all_anchors

    def _meshgrid(self, x, y, row_major=True):
        """Generate mesh grid of x and y.

        Args:
            x (tf.Tensor): Grids of x dimension.
            y (tf.Tensor): Grids of y dimension.
            row_major (bool, optional): Whether to return y grids first.
                Defaults to True.

        Returns:
            tuple[tf.Tensor]: The mesh grids of x and y.
        """

        xx = tf.tile(x, [tf.shape(y)[0]])
        yy = tf.repeat(y, tf.shape(x)[0])
        if row_major:
            return xx, yy
        else:
            return yy, xx

    def valid_flags(self, batch, featmap_sizes, pad_shape):
        """Generate valid flags of anchors in multiple feature levels.

        Args:
            featmap_sizes (list(tuple)): List of feature map sizes in
                multiple feature levels.
            pad_shape (tuple): The padded shape of the image.
            device (str): Device where the anchors will be put on.

        Return:
            list(torch.Tensor): Valid flags of anchors in multiple levels.
        """
        assert self.num_levels == len(featmap_sizes)
        multi_level_flags = []
        for i in range(self.num_levels):
            anchor_stride = self.strides[i]
            feat_h = featmap_sizes[i][0]
            feat_w = featmap_sizes[i][1]
            # feat_h, feat_w = featmap_sizes[i]
            h = pad_shape[0]
            w = pad_shape[1]
            valid_feat_h = tf.math.minimum(
                tf.cast(tf.math.ceil(h / anchor_stride[1]), tf.int32), feat_h)
            valid_feat_w = tf.math.minimum(
                tf.cast(tf.math.ceil(w / anchor_stride[0]), tf.int32), feat_w)
            # valid_feat_h = min(int(np.ceil(h / anchor_stride[1])), feat_h)
            # valid_feat_w = min(int(np.ceil(w / anchor_stride[0])), feat_w)
            flags = self.single_level_valid_flags((feat_h, feat_w),
                                                  (valid_feat_h, valid_feat_w),
                                                  self.num_base_anchors[i])

            flags = tf.tile(flags[None, :], [batch, 1])

            multi_level_flags.append(flags)
        return multi_level_flags

    def single_level_valid_flags(self, featmap_size, valid_size,
                                 num_base_anchors):
        """Generate the valid flags of anchor in a single feature map.

        Args:
            featmap_size (tuple[int]): The size of feature maps.
            valid_size (tuple[int]): The valid size of the feature maps.
            num_base_anchors (int): The number of base anchors.
            device (str, optional): Device where the flags will be put on.
                Defaults to 'cuda'.

        Returns:
            torch.Tensor: The valid flags of each anchor in a single level \
                feature map.
        """
        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size
        # assert valid_h <= feat_h and valid_w <= feat_w

        valid_x = tf.zeros(feat_w, dtype=tf.float32)
        valid_y = tf.zeros(feat_h, dtype=tf.float32)

        valid_x = tf.tensor_scatter_nd_update(valid_x,
                                              tf.range(valid_w)[:, None],
                                              tf.ones(shape=(valid_w, )))
        valid_y = tf.tensor_scatter_nd_update(valid_y,
                                              tf.range(valid_h)[:, None],
                                              tf.ones(shape=(valid_h, )))
        valid_xx, valid_yy = self._meshgrid(valid_x, valid_y)
        valid_xx = tf.cast(valid_xx, tf.bool)
        valid_yy = tf.cast(valid_yy, tf.bool)
        valid = tf.math.logical_and(valid_xx, valid_yy)
        # valid = valid_xx & valid_yy
        valid = tf.tile(valid[:, None], [1, num_base_anchors])
        valid = tf.reshape(valid, [-1])
        return valid

    @property
    def num_base_anchors(self):
        """list[int]: total number of base anchors in a feature grid"""
        return [base_anchors.shape[0] for base_anchors in self.base_anchors]

    @property
    def num_levels(self):
        """int: number of feature levels that the generator will be applied"""
        return len(self.strides)
