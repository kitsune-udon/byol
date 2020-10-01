import inspect
from argparse import Namespace


def extract_kwargs_from_argparse_args(cls, args, **kwargs):
    assert(isinstance(args, Namespace))
    params = vars(args)

    valid_kwargs = inspect.signature(cls.__init__).parameters
    tmp_kwargs = dict((name, params[name]) for name in valid_kwargs if name in params)
    tmp_kwargs.update(**kwargs)

    return tmp_kwargs

def from_argparse_args(cls, args, **kwargs):
    return cls(**extract_kwargs_from_argparse_args(cls, args, **kwargs))
