
import os
from collections import Counter, defaultdict


class BDD:
    def __init__(self, json_file):
        self.pathDict = {}
        self.pathSortedList = []
        self.category_statistic = []
        for frame in json_file['frame_list']:
            imgPath = os.path.join(frame['name'])
            labelsList = []
            for label in frame['labels']:
                self.category_statistic.append(label['category'].upper())
                labelsList.append(
                    dict(category=label['category'].upper(),
                         box2d=label['box2d']))
            self.pathDict[imgPath] = labelsList
        self.pathSortedList = sorted(self.pathDict)
        self.category_counts = Counter(self.category_statistic)
        self.category_iou = defaultdict(list)

    def __len__(self):
        return len(self.pathDict)

    def get_item(self, n):
        return self.pathSortedList[n]

    def get_labels(self, n):
        return self.pathDict[self.pathSortedList[n]]

    def get_cates(self):
        # just check the first frame
        return tuple(sorted(self.category_counts.keys()))

    def get_cates_counts(self):
        # just check the first frame
        return self.category_counts
