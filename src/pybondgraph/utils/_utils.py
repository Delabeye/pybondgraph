"""

General purpose classes, functions, global variables and imports

"""

### Local

from pybondgraph.utils._type_utils import *

###

from abc import ABC, ABCMeta, abstractmethod, abstractproperty
from dataclasses import dataclass, field

import os, sys
import inspect
from pathlib import Path
import uuid, shortuuid

import cloudpickle as pickle
import copy

from pytictoc import TicToc
import time
import datetime
import joblib

from operator import itemgetter, attrgetter
import itertools
import functools
from collections import OrderedDict
from ordered_set import OrderedSet


import matplotlib

matplotlib.use("Qt5Agg")

import matplotlib.pyplot as plt
from colorama import Fore, Back, Style

import pandas as pd
import numpy as np
import scipy
import scipy.linalg
import scipy.optimize
import scipy.stats
import scipy.signal
import scipy.spatial
import scipy.integrate
import scipy.special

# Dev only
from icecream import ic, install

install()  # icecream


"""

Toolbox

"""


class tmp_:
    """generate & manipulate temporary id in variable name"""

    def id():
        return f"{shortuuid.random(8)}__tmp"

    def contains_tmp_id(name):
        return str(name).endswith("__tmp")

    def stem(name):
        return str(name)[: -len(tmp_.id())]


#


def new(cls, n=1, *args, **kwargs):
    """Create and initialise `n` instances of a class"""
    return [cls(*args, **kwargs) for _ in range(n)]


#


class SortedDict(OrderedDict):
    """Dictionary sorted by key upon query ( `items()`, `keys()`, `values()` )"""

    def items(self):
        return OrderedDict(sorted(super().items())).items()

    def values(self):
        return OrderedDict(sorted(super().items())).values()

    def keys(self):
        return OrderedDict(sorted(super().items())).keys()


#


tictoc = TicToc()
tic = lambda *args, **kwargs: tictoc.tic(*args, **kwargs)
toc = lambda *args, **kwargs: tictoc.toc(*args, **kwargs)


def all_attrs(object, exclude=[]) -> dict:
    """Builds a dictionary out of a class' attributes (excluding protected & private members and methods)"""
    return {
        attr: val
        for attr, val in inspect.getmembers(object)
        if not inspect.ismethod(val)
        and not attr.startswith("_")
        and not attr in exclude
    }


class Summable(object):
    def __add__(self, obj):
        return obj


class MultiKeyDict:
    """Multi Key Dict
    adapted from https://stackoverflow.com/a/11105962
    """

    def __init__(self, **kwargs):
        self._keys = {}
        self._data = {}
        for k, v in iter(kwargs.items()):
            self[k] = v

    def __getitem__(self, key):
        try:
            return self._data[key]
        except KeyError:
            return self._data[self._keys[key]]

    def __setitem__(self, key, val):
        try:
            self._data[self._keys[key]] = val
        except KeyError:
            if isinstance(key, tuple):
                if not key:
                    raise ValueError("Empty tuple cannot be used as a key")
                key = list(set(key))  # avoid duplicate keys
                key, other_keys = key[0], key[1:]
            else:
                other_keys = []
            self._data[key] = val
            for k in other_keys:
                self._keys[k] = key

    def add_keys(self, to_key, new_keys):
        if to_key not in self._data:
            to_key = self._keys[to_key]
        for key in new_keys:
            self._keys[key] = to_key

    @classmethod
    def from_dict(cls, dic):
        result = cls()
        for key, val in dic.items():
            result[key] = val
        return result

    def keys(self) -> Sequence[Sequence[str]]:
        return [
            (k, *[alias for alias in self._keys.keys() if self._keys[alias] == k])
            for k in self._data.keys()
        ]

    def values(self) -> Sequence[Any]:
        return list(self._data.values())

    def items(self):
        return {K: self._data[K[0]] for K in self.keys()}

    def __repr__(self) -> str:
        return str(self.items())


"""

DEBUG & LOGGING

"""

# from tqdm import tqdm
# from rich.progress import track as tqdm
from tqdm.rich import tqdm
from tqdm.auto import tqdm as tqdm_

import traceback
import logging, colorlog


class ProgressParallel(joblib.Parallel):
    """Same as joblib.Parallel, with progress bar.
    https://stackoverflow.com/questions/37804279/how-can-we-use-tqdm-in-a-parallel-execution-with-joblib
    """

    def __call__(self, *args, **kwargs):
        with tqdm_() as self._pbar:
            return joblib.Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


def generate_method(obj: object, name: str, function: Callable, set_kwargs={}):
    """generate a method and set some of its default kwargs (/!\ lambda function)"""
    setattr(
        obj, name, lambda *args, **kwargs: function(*args, **{**kwargs, **set_kwargs})
    )


class Logger:
    """Custom logger"""

    def __init__(self) -> None:
        self.logs = []
        Logger.setup_logging()
        self.logger = logging.getLogger(__name__)
        for logtype in [
            "debug",
            "info",
            "warning",
            "error",
            "critical",
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
            "CRITICAL",
        ]:
            generate_method(self, logtype, self.log, set_kwargs=dict(logtype=logtype))

    @staticmethod
    def setup_logging():
        """customise logger display"""
        root = logging.getLogger(__name__)
        root.setLevel(logging.DEBUG)
        format = "%(asctime)s - %(levelname)-8s - %(message)s"
        date_format = "%H:%M:%S"  # '%Y-%m-%d %H:%M:%S'
        if "colorlog" in sys.modules and os.isatty(2):
            cformat = "%(log_color)s" + format
            f = colorlog.ColoredFormatter(
                cformat,
                date_format,
                log_colors={
                    "DEBUG": "blue",
                    "WARNING": "bold_yellow",
                    "INFO": "green",
                    "ERROR": "bold_red",
                    "CRITICAL": "bold_red",
                },
            )
            # Available colors:
            #   black, red, green, yellow, blue, purple, cyan and white
            # Options:
            #   {color}, fg_{color}, bg_{color}, reset                  : foreground/background color; reset
            #   bold, bold_{color}, fg_bold_{color}, bg_bold_{color}    : bold/bright
            #   thin, thin_{color}, fg_thin_{color}                     : thin
        else:
            f = logging.Formatter(format, date_format)
        ch = logging.StreamHandler()
        ch.setFormatter(f)
        root.addHandler(ch)

    def log(
        self, msg, check=None, show_only_once=False, logtype="info", *args, **kwargs
    ):
        """conditional logging"""
        if check is None or check is False:
            if logtype in ["critical", "CRITICAL", "error", "ERROR"]:
                if check is False:
                    try:
                        raise AssertionError
                    except:
                        # caller = getframeinfo(stack()[2][0])
                        # print('\nFile "%s":%d:\n\n\t%s\n' % (caller.filename, caller.lineno, caller.code_context[0][:-1]))
                        getattr(self.logger, logtype)(msg, *args, **kwargs)
                        print("\n".join(traceback.format_stack()[:-2]))
                        exit()
                else:
                    getattr(self.logger, logtype)(msg, *args, **kwargs)
                    exit()
            else:
                if not show_only_once or (show_only_once and not msg in self.logs):
                    getattr(self.logger, logtype)(msg, *args, **kwargs)
                    self.logs.append(msg)
        return check

    def debug(self, msg, check=None):
        """Overwritten at initialisation"""

    def info(self, msg, check=None):
        """Overwritten at initialisation"""

    def warning(self, msg, check=None):
        """Overwritten at initialisation"""

    def error(self, msg, check=None):
        """Overwritten at initialisation"""

    def critical(self, msg, check=None):
        """Overwritten at initialisation"""


log = Logger()

### mute TqdmExperimentalWarning (rich's tqdm equivalent being in beta version ATTOW)
import warnings
from tqdm import TqdmExperimentalWarning

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
