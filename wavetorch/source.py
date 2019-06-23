import torch
import skimage

class Source(object):
    def __init__(self, geometry, mask):
        super(Source, self).__init__()
        self.mask = mask

    def __call__(self, X):
        # (batch_num, ) ---> (bath_num, dim1, dim2, ...)
        return self.mask * X.unsqueeze(-1).unsqueeze(-1)

class LineSource(Source):
    def __init__(self, geometry, r0, c0, r1, c1):
        x, y = skimage.draw.line(r0, c0, r1, c1)
        self.mask = torch.zeros(geometry.rho.size())
        self.mask[x, y] = 1

class PointSource(Source):
    def __init__(self, geometry, x, y):
        self.x = x;
        self.y = y;
        self.mask = torch.zeros(geometry.rho.size())
        self.mask[x, y] = 1
