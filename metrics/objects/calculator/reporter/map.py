import dataclasses
from collections import Hashable
from typing import List, Union

from bddhelper.objects.base import BaseLabelObject
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from .base import Reporter


@dataclasses.dataclass
class COCOMAPBoxReporter(Reporter):
    def __post_init__(self):
        self._gt_annotations, self._pred_annotations = [], []
        self._imgs, self._cats = {}, {}
        self._id_count = 0

    @property
    def report(self):
        coco_gt = COCO()
        coco_gt.dataset = dict(annotations=self._gt_annotations,
                               images=[{
                                   'file_name': img,
                                   'id': id
                               } for img, id in self._imgs.items()],
                               categories=[{
                                   'name': cat,
                                   'id': id
                               } for cat, id in self._cats.items()])
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(self._pred_annotations)
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.params.imgIds = list(self._imgs.values())
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        coco_report_title = [
            'AP | IoU=0.50:0.95 | area=all | maxDets=100',
            'AP | IoU=0.50 | area=all | maxDets=100',
            'AP | IoU=0.75 | area=all | maxDets=100',
            'AP | IoU=0.50:0.95 | area=small | maxDets=100',
            'AP | IoU=0.50:0.95 | area=medium | maxDets=100',
            'AP | IoU=0.50:0.95 | area=large | maxDets=100',
            'AR | IoU=0.50:0.95 | area=all | maxDets=1',
            'AR | IoU=0.50:0.95 | area=all | maxDets=10',
            'AR | IoU=0.50:0.95 | area=all | maxDets=100',
            'AR | IoU=0.50:0.95 | area=small | maxDets=100',
            'AR | IoU=0.50:0.95 | area=medium | maxDets=100',
            'AR | IoU=0.50:0.95 | area=large | maxDets=100'
        ]
        return {
            title: round(metric, 3)
            for title, metric in zip(coco_report_title, coco_eval.stats)
        }

    def bdd_to_coco(self, lb: BaseLabelObject, image_name: Hashable):
        """transform bdd object to coco object

        Args:
            lb (BaseLabelObject): BDD object
            image_name (Hashable): image name

        Returns:
            dict: coco object
        """
        # In COCOAPI, we need to provide:
        # 1. categories: [{'name':'cat1', 'id':1}, {'name':'cat2', 'id':2}}]
        # 2. images: [{'file_name': 'xxx', 'id':1}, {'file_name': 'yyy', id:2}]
        # 3. annotations: [{'bbox':[x1, y1, w, h], 'category_id': int, 'image_id': int, 'id': int, iscrowd: int, area: float}]
        assert lb.shape_type == 'box2d', (
            f'expected box2d, got {lb.shape_type}')
        bbox = [
            lb.shape['x1'], lb.shape['y1'], lb.shape['x2'] - lb.shape['x1'],
            lb.shape['y2'] - lb.shape['y1']
        ]
        category_id = self._cats[lb.category]
        score = lb.meta_ds.get('score', 1.0)
        image_id = self._imgs[image_name]
        area = bbox[2] * bbox[3]
        self._id_count += 1
        return dict(bbox=bbox,
                    category_id=category_id,
                    score=score,
                    image_id=image_id,
                    area=area,
                    id=self._id_count,
                    iscrowd=0)

    def update_imgs(self, imgs: Union[Hashable, List[Hashable]]):
        """ update imgs and make id mapping

        Args:
            cats (Union[Hashable, List[Hashable]]): image list
        """
        if not isinstance(imgs, (list, tuple)):
            imgs = [imgs]
        for img in imgs:
            if img not in self._imgs:
                self._imgs[img] = len(self._imgs) + 1

    def update_categories(self, cats: Union[Hashable, List[Hashable]]):
        """ update category and make id mapping

        Args:
            cats (Union[Hashable, List[Hashable]]): category list
        """
        if not isinstance(cats, (list, tuple)):
            cats = [cats]
        for cat in cats:
            if cat not in self._cats:
                self._cats[cat] = len(self._cats) + 1

    def update(self, annotations: dict):
        assert isinstance(annotations['gt'], list), (
            f'expected ann["gt"] in type list, got {type(annotations.get("gt"))}'
        )
        assert isinstance(annotations['eval'], list), (
            f'expected ann["eval"] in type list, got {type(annotations.get("eval"))}'
        )

        self.update_imgs(imgs=[
            ann['frame_id'] for anns in annotations.values() for ann in anns
        ])
        self.update_categories(cats=[
            ann['object'].category for anns in annotations.values()
            for ann in anns
        ])

        self._gt_annotations += [
            self.bdd_to_coco(ann['object'], ann['frame_id'])
            for ann in annotations['gt']
        ]
        self._pred_annotations += [
            self.bdd_to_coco(ann['object'], ann['frame_id'])
            for ann in annotations['eval']
        ]
