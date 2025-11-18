import numpy as np
from hypothesis import given, settings
import hypothesis.strategies as st
import hypothesis.extra.numpy as hnp
from jaxtyping import install_import_hook

# with install_import_hook("contingency", "beartype.beartype"):
#     from contingency import Contingent

from contingency import Contingent

# import warnings
# warnings.filterwarnings("error")


@st.composite  
def make_shapes(draw):
    return draw(hnp.array_shapes(max_dims=2, min_dims=1, min_side=5))


@st.composite
def make_bools(draw, shape=(1,5)):
    arr = draw(hnp.arrays(
        bool,
        shape,
        elements=st.just(True), fill=st.just(False),
    ))
    return arr


@st.composite
def make_true_pred(draw):
    shape = draw(make_shapes())
    y_true = draw(make_bools(shape=shape[-1]))
    y_pred = draw(make_bools(shape=shape))
    return (y_true, y_pred)

@given(make_true_pred())
def test_scoring(y_Y):
    y_true, y_pred = y_Y
    M = Contingent(y_true, y_pred)
    assert M.mcc.dtype == 'float'
    assert  np.all(M.F <= M.G)
    # try:
    #     assert M.mcc.dtype == 'float'
    # except RuntimeWarning:
    #     print(M)
