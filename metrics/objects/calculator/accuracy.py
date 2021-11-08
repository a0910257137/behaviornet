import dataclasses
from typing import ClassVar, Dict, List, Optional, Union

from metrics.objects.metric.iou.iou import IoU
from metrics.objects.metric.shift.point import PointDistance


@dataclasses.dataclass
class AccCalculator:
    """Accuracy calculator calculates attributes accuracy or regression with threshold.

    Usage:
        # define input args
        metric = PointDistance()
        transform = Compose([Box2DToKeyPoints(), MinimalBox2D(), GetShape()])
        calculator = AccCalculator(metric, transform, parser)
        calculator(matched_objects={'gt':gt_lbs, 'eval':eval_lbs})
        print(calculator.report)


    """
    def __call__(self,
                 matched_objects: Dict[str, List],
                 transform: object,
                 metric: object,
                 reporter: object,
                 threshold: Optional[Union[int, float, str]],
                 threshold_amplifier: Optional[Union[dict, int, float]] = None,
                 discarded_keys: Optional[Union[list, tuple]] = None,
                 **kwargs):
        # Note: matched_objects['gt'], matched_objects['eval'] (list). It's frame level matched objects
        #       The original input should be `matched_objects`, `unmatched_objects`, `raw_objects`, but
        #       `AccCalculator` only cares about `matched_objects`, so using `**kwargs` to hide other two inputs.
        assert isinstance(threshold_amplifier,
                          (dict, int, float)) or threshold_amplifier is None
        assert isinstance(discarded_keys,
                          (list, tuple)) or discarded_keys is None
        assert isinstance(transform, object)
        self.transform = transform
        self.metric = metric
        self.reporter = reporter
        self.threshold = threshold
        self.__default_frame_id__: ClassVar[str] = '__all_frame__'
        # discarded_keys defined in functoin has higher priority than class pre-defined
        discarded_keys = discarded_keys

        gt_lbs, eval_lbs = matched_objects['gt'], matched_objects['eval']

        if not isinstance(gt_lbs, (list, tuple)):
            gt_lbs = [gt_lbs]
        if not isinstance(eval_lbs, (list, tuple)):
            eval_lbs = [eval_lbs]

        for gt_lb, eval_lb in zip(gt_lbs, eval_lbs):
            user_lb_dict = dict(gt=gt_lb, eval=eval_lb)
            result = self.calculate(user_lb_dict, discarded_keys)
            # heere the reporter should get methods
            # only support two metric types
            if isinstance(self.metric, IoU):
                self.reporter.update(result,
                                     self.threshold,
                                     threshold_amplifier,
                                     reversed_op=True)
            elif isinstance(self.metric, PointDistance):
                self.reporter.update(result,
                                     self.threshold,
                                     threshold_amplifier,
                                     reversed_op=False)

    @property
    def report(self):
        return self.reporter.report

    def reset(self):
        self.reporter.reset()

    def calculate(self,
                  user_lb_dict: Dict[str, List[Union[dict]]],
                  discarded_keys: Optional[Union[list, tuple]] = None,
                  **kwargs):
        """The calculate function for 1 v.s 1 calculation

        Args:
            user_lb_dict (Dict[str, List[Union[BaseLabelObject, dict]]]): user label
            threshold_amplifier (Optional[Union[dict, int, float]], optional): threshold amplifier, specify the amplifying factor with keys or with scalar. Defaults to None.
            discarded_keys (Optional[list], optional): specify in list which key is not needed. Defaults to None.
        """

        # parse target key, attributes or shape
        # Note: gt_lb, eval_lb: {'frame_id': Any, 'object': Union[BDDObject, dict]}
        gt_lb = user_lb_dict['gt'][0]
        eval_lb = user_lb_dict['eval'][0]
        # gt_data, eval_data = self.parser(gt_lb), self.parser(eval_lb)
        # data transform
        if self.transform is not None:
            gt_data = self.transform(gt_lb)
            eval_data = self.transform(eval_lb)

        # discard unnecessary key
        if discarded_keys is not None:
            for key in discarded_keys:
                gt_data.pop(key, None)
                eval_data.pop(key, None)

        assert gt_data and eval_data, f'One of gt_data and eval_data is empty. gt_data:{gt_data}, eval_data:{eval_data}'
        # calculate in metric, only calculate numbers not for thresholding

        metric_ret = self.metric(gt_data, eval_data)
        if isinstance(self.metric, IoU):
            metric_ret = dict(IoU=metric_ret)
        try:
            frame_id = gt_lb['frame_id']
        except:
            frame_id = self.__default_frame_id__
        result = dict(frame_id=frame_id, metric=metric_ret)
        return result
