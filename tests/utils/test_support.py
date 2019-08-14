"""
Tests support concept decorators
"""
import thelper.utils


def test_class_decorated_with_parenthesis_and_concept():

    @thelper.utils.apply_support(concept="test1")
    class Test1(object):
        pass

    @thelper.utils.apply_support(concept="test2")
    class Test2(Test1):
        pass

    class Test3(Test1):
        pass

    assert thelper.utils.supports(Test1, "test1")
    assert thelper.utils.supports(Test2, "test1")
    assert thelper.utils.supports(Test2, "test2")
    assert thelper.utils.supports(Test3, "test1")
    assert not thelper.utils.supports(Test3, "test2")

    t1 = Test1()
    t2 = Test2()
    t3 = Test3()
    assert thelper.utils.supports(t1, "test1")
    assert thelper.utils.supports(t2, "test1")
    assert thelper.utils.supports(t2, "test2")
    assert thelper.utils.supports(t3, "test1")
    assert not thelper.utils.supports(t3, "test2")


def test_class_decorated_with_helper_and_parenthesis():
    def helper1(func=None):
        return thelper.utils.apply_support(func, "test1")

    def helper2(func=None):
        return thelper.utils.apply_support(func, "test2")

    @helper1()
    class Test1(object):
        pass

    @helper2()
    class Test2(Test1):
        pass

    class Test3(Test1):
        pass

    assert thelper.utils.supports(Test1, "test1")
    assert thelper.utils.supports(Test2, "test1")
    assert thelper.utils.supports(Test2, "test2")
    assert thelper.utils.supports(Test3, "test1")
    assert not thelper.utils.supports(Test3, "test2")

    t1 = Test1()
    t2 = Test2()
    t3 = Test3()
    assert thelper.utils.supports(t1, "test1")
    assert thelper.utils.supports(t2, "test1")
    assert thelper.utils.supports(t2, "test2")
    assert thelper.utils.supports(t3, "test1")
    assert not thelper.utils.supports(t3, "test2")


def test_class_decorated_with_helper_without_parenthesis():
    def helper1(func=None):
        return thelper.utils.apply_support(func, "test1")

    def helper2(func=None):
        return thelper.utils.apply_support(func, "test2")

    @helper1    # no parenthesis here
    class Test1(object):
        pass

    @helper2    # no parenthesis here
    class Test2(Test1):
        pass

    class Test3(Test1):
        pass

    assert thelper.utils.supports(Test1, "test1")
    assert thelper.utils.supports(Test2, "test1")
    assert thelper.utils.supports(Test2, "test2")
    assert thelper.utils.supports(Test3, "test1")
    assert not thelper.utils.supports(Test3, "test2")

    t1 = Test1()
    t2 = Test2()
    t3 = Test3()
    assert thelper.utils.supports(t1, "test1")
    assert thelper.utils.supports(t2, "test1")
    assert thelper.utils.supports(t2, "test2")
    assert thelper.utils.supports(t3, "test1")
    assert not thelper.utils.supports(t3, "test2")
