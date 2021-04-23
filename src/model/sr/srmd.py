from model import common
from model import edsr
import misc

from torch import nn

def model_class(*args, **kwargs):
    return common.model_class(SRMD, *args, **kwargs)


class SRMD(edsr.EDSR):
    '''
    SRMD model (baseline: EDSR)

    Note:
    '''

    def __init__(self, *args, **kwargs):
        super(SRMD, self).__init__(*args, **kwargs)
        self.conv = kwargs['conv'](kwargs['n_colors'] + 1, kwargs['n_feats'], 3)
