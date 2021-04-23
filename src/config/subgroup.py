class SubGroup(object):

    def __init__(self, group, name):
        self.subgroup = group.add_argument_group(name)

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        pass

    def add(self, *args, **kwargs):
        self.subgroup.add_argument(*args, **kwargs)
