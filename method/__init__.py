# Only ReSA is implemented; get_method keeps the upstream interface.
from .resa import ReSA


def get_method(args):
    return ReSA(args)
