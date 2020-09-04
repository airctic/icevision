from icevision.all import *


def test_capture_stdout_simple():
    with CaptureStdout() as out:
        print("mantis")
        print("shrimp")

    assert out == ["mantis", "shrimp"]


def test_capture_stdout_propagate():
    with CaptureStdout() as out1:
        print("mantis")
        with CaptureStdout(propagate_stdout=True) as out2:
            print("shrimp")

    assert out1 == ["mantis", "shrimp"]


def test_capture_stdout_block_propagate():
    with CaptureStdout() as out1:
        print("mantis")
        with CaptureStdout() as out2:
            print("shrimp")

    assert out1 == ["mantis"]
    assert out2 == ["shrimp"]
