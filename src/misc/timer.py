import time

class Timer():
    '''
    A simple timer class.
    '''

    def __init__(self):
        self.acc = 0
        self.tic()

    def __str__(self):
        return str(self.acc)

    def __format__(self, fmt):
        return self.acc.__format__(fmt)

    def __enter__(self):
        self.tic()

    def __exit__(self, *args, **kwargs):
        self.toc()

    def tic(self):
        self.t0 = time.time()

    def toc(self):
        now = time.time()
        self.acc += (now - self.t0)
        self.t0 = now
        return self.acc
