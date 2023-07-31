"""Tests comparing analytical results with results obtained using relations
known from theory of thin isotropic plates and automatic differentiation"""

from jax.numpy import isclose

from const import a, b
from core import w, M_x, M_y, Q_x, Q_y
from book import w_max, M_x_max, M_y_max, _Q_x, _Q_y


x = a / 2
y = b / 2


def test_w():
    assert isclose(w(x, y), w_max)


def test_M_x():
    assert isclose(M_x(x, y), M_x_max)


def test_M_y():
    assert isclose(M_y(x, y), M_y_max)


def test_Q_x():
    assert isclose(Q_x(x, y), _Q_x(x, y))


def test_Q_y():
    assert isclose(Q_y(x, y), _Q_y(x, y))
