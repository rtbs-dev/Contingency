from typing import Literal, Any, ParamSpec, TypeVar, Callable, Type, Self
from jaxtyping import Bool, Num, jaxtyped
from numpy import ndarray as nda

Params = ParamSpec("Params")
Returns = TypeVar("Returns")

type IsolineMetric = Literal['F','G', 'mcc', "J"]

type ScoreOptions = Literal[
    'F',
    'F2',
    'G',
    'fowlkes_mallows',
    'J',
    'recall',
    'precision',
    'informedness',
    'markedness',
    'mcc',
    'aps',
    'auc',
]

type PredProb = Num[nda, 'features']
type ProbThres = Num[nda, '*#batch']
type PredThres = Bool[nda, '*#batch features']
