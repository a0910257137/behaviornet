from pprint import pprint
import numpy as np


class Base:

    def parse_lnmk(self, item):
        for key in list(item.keys()):
            item[key] = np.asarray(item[key]).astype(np.float32)
        return item
