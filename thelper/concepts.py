"""Framework concepts module for high-level interface documenting.

The decorators and type checkers defined in this module help highlight the
purpose of classes and functions with respect to high-level ML tasks.
"""
import functools
import inspect
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from typing import Any, AnyStr, Callable, Optional, Union  # noqa: F401
    from types import FunctionType  # noqa: F401

SUPPORT_PREFIX = "supports_"
"""Prefix that is applied before any 'concept' decorator."""


def apply_support(func_or_cls=None, concept=None):
    # type: (Optional[Union[FunctionType, Callable]], Optional[AnyStr]) -> Callable
    """
    Utility decorator that allows marking a function or a class as *supporting* a certain ``concept``.

    Notes:
        ``concept`` support by the function or class is marked only as documentation reference, no strict
        validation is accomplished to ensure that further underlying requirements are met for the *concept*.

    .. seealso::
        | :func:`thelper.concepts.classification`
        | :func:`thelper.concepts.detection`
        | :func:`thelper.concepts.segmentation`
        | :func:`thelper.concepts.regression`
    """

    # actual function that applies the concept to the decorated 'thing'
    def apply_concept(_thing, _concept, *_args, **_kwargs):
        assert isinstance(_concept, str) and len(_concept)
        _concept = f"{SUPPORT_PREFIX}{_concept}" if not _concept.startswith(SUPPORT_PREFIX) else _concept
        setattr(_thing, _concept, True)
        return _thing

    # wrapper generator for class types or class instances
    class ApplyDecorator(object):
        def __new__(cls, wrapped=None, _concept=None, *args, **kwargs):
            cls.__wrapped__ = wrapped
            return apply_concept(wrapped, _concept, *args, **kwargs)

    # wrapper generator for functions or class methods
    def apply_decorator(f_or_c):
        if inspect.isclass(f_or_c):
            return ApplyDecorator(f_or_c, concept)

        @functools.wraps(f_or_c)  # lift wrapped object definitions, so that it still looks like the original
        def decorate(*args, **kwargs):
            return apply_concept(f_or_c, concept)(*args, **kwargs)
        return decorate

    if inspect.isclass(func_or_cls):
        # the '_concept' input passed down here ensures that a class decorated with a predefined
        # shortcut decorator function called without parenthesis will still receive the wanted 'concept'
        # ex:
        #   @support_classification     # <== no () here
        #   def ClassifObj(): pass
        return ApplyDecorator(func_or_cls, _concept=concept)
    elif func_or_cls:
        # this is in case the decorator is applied to a function instead of a class
        return apply_decorator(func_or_cls)
    return apply_decorator


def supports(thing, concept):
    # type: (Any, AnyStr) -> bool
    """Utility method to evaluate if ``thing`` *supports* a given ``concept`` as defined by decorators.

    Arguments:
        thing: any type, function, method, class or object instance to evaluate if it is marked by the concept
        concept: concept to check

    .. seealso::
        | :func:`thelper.concepts.classification`
        | :func:`thelper.concepts.detection`
        | :func:`thelper.concepts.segmentation`
        | :func:`thelper.concepts.regression`
    """
    if not isinstance(concept, str):
        return False
    concept = concept.lower()  # in case it was capitalized
    return getattr(thing, f"{SUPPORT_PREFIX}{concept}", False)


def classification(func_or_cls=None):
    # type: (Optional[Union[FunctionType, Callable]]) -> Callable
    """Decorator that allows marking a function or class as *supporting* the image classification task.

    Example::

        @thelper.concepts.classification
        class ClassifObject():
            pass

        c = ClassifObject()
        c.supports_classification
        > True

        thelper.concepts.supports(c, "classification")
        > True
    """
    return apply_support(func_or_cls, "classification")


def detection(func_or_cls=None):
    # type: (Optional[Union[FunctionType, Callable]]) -> Callable
    """Decorator that allows marking a function or class as *supporting* the object detection task.

    Example::

        @thelper.concepts.detection
        class DetectObject():
            pass

        d = DetectObject()
        d.supports_detection
        > True

        thelper.concepts.supports(d, "detection")
        > True
    """
    return apply_support(func_or_cls, "detection")


def segmentation(func_or_cls=None):
    # type: (Optional[Union[FunctionType, Callable]]) -> Callable
    """Decorator that allows marking a function or class as *supporting* the image segmentation task.

    Example::

        @thelper.concepts.segmentation
        class SegmentObject():
            pass

        s = SegmentObject()
        s.supports_segmentation
        > True

        thelper.concepts.supports(s, "segmentation")
        > True
    """
    return apply_support(func_or_cls, "segmentation")


def regression(func_or_cls=None):
    # type: (Optional[Union[FunctionType, Callable]]) -> Callable
    """Decorator that allows marking a function or class as *supporting* the generic regression task.

    Example::

        @thelper.concepts.regression
        class RegrObject():
            pass

        r = RegrObject()
        r.supports_regression
        > True

        thelper.concepts.supports(r, "regression")
        > True
    """
    return apply_support(func_or_cls, "regression")
