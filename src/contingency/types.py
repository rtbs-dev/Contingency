from typing import Literal, Any, ParamSpec, TypeVar, Callable, Type, Self
from jaxtyping import Bool, Num, jaxtyped
from numpy import ndarray as nda

Params = ParamSpec("Params")
Returns = TypeVar("Returns")

type IsolineMetric = Literal['F','G', 'mcc']

type ScoreOptions = Literal[
    'F',
    'F2',
    'G',
    'recall',
    'precision',
    'mcc',
    'aps',
    'auc',
]

type PredProb = Num[nda, 'features']
type ProbThres = Num[nda, '*#batch']
type PredThres = Bool[nda, '*#batch features']
