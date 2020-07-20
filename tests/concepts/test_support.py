"""
Tests support concept decorators
"""
import thelper.concepts


def test_class_decorated_with_parenthesis_and_concept():

    @thelper.concepts.apply_support(concept="test1")
    class Test1(object):
        pass

    @thelper.concepts.apply_support(concept="test2")
    class Test2(Test1):
        pass

    class Test3(Test1):
        pass

    assert thelper.concepts.supports(Test1, "test1")
    assert thelper.concepts.supports(Test2, "test1")
    assert thelper.concepts.supports(Test2, "test2")
    assert thelper.concepts.supports(Test3, "test1")
    assert not thelper.concepts.supports(Test3, "test2")

    t1 = Test1()
    t2 = Test2()
    t3 = Test3()
    assert thelper.concepts.supports(t1, "test1")
    assert thelper.concepts.supports(t2, "test1")
    assert thelper.concepts.supports(t2, "test2")
    assert thelper.concepts.supports(t3, "test1")
    assert not thelper.concepts.supports(t3, "test2")


def test_class_decorated_with_helper_and_parenthesis():
    def helper1(func=None):
        return thelper.concepts.apply_support(func, "test1")

    def helper2(func=None):
        return thelper.concepts.apply_support(func, "test2")

    @helper1()
    class Test1(object):
        pass

    @helper2()
    class Test2(Test1):
        pass

    class Test3(Test1):
        pass

    assert thelper.concepts.supports(Test1, "test1")
    assert thelper.concepts.supports(Test2, "test1")
    assert thelper.concepts.supports(Test2, "test2")
    assert thelper.concepts.supports(Test3, "test1")
    assert not thelper.concepts.supports(Test3, "test2")

    t1 = Test1()
    t2 = Test2()
    t3 = Test3()
    assert thelper.concepts.supports(t1, "test1")
    assert thelper.concepts.supports(t2, "test1")
    assert thelper.concepts.supports(t2, "test2")
    assert thelper.concepts.supports(t3, "test1")
    assert not thelper.concepts.supports(t3, "test2")


def test_class_decorated_with_helper_without_parenthesis():
    def helper1(func=None):
        return thelper.concepts.apply_support(func, "test1")

    def helper2(func=None):
        return thelper.concepts.apply_support(func, "test2")

    @helper1    # no parenthesis here
    class Test1(object):
        pass

    @helper2    # no parenthesis here
    class Test2(Test1):
        pass

    class Test3(Test1):
        pass

    assert thelper.concepts.supports(Test1, "test1")
    assert thelper.concepts.supports(Test2, "test1")
    assert thelper.concepts.supports(Test2, "test2")
    assert thelper.concepts.supports(Test3, "test1")
    assert not thelper.concepts.supports(Test3, "test2")

    t1 = Test1()
    t2 = Test2()
    t3 = Test3()
    assert thelper.concepts.supports(t1, "test1")
    assert thelper.concepts.supports(t2, "test1")
    assert thelper.concepts.supports(t2, "test2")
    assert thelper.concepts.supports(t3, "test1")
    assert not thelper.concepts.supports(t3, "test2")


def test_concept_any_capitalization():

    @thelper.concepts.apply_support(concept="test1")
    class Test1(object):
        pass

    t1 = Test1()

    for concept in ["test1", "TEST1", "Test1"]:
        assert thelper.concepts.supports(Test1, concept)
        assert thelper.concepts.supports(t1, concept)
    for concept in ["test2", "TEST2", "Test2"]:
        assert not thelper.concepts.supports(Test1, concept)
        assert not thelper.concepts.supports(t1, concept)
