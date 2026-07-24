import unittest

from pipewire.capture_probe import wait_for_byte


class Selector:
    def __init__(self, events):
        self.events = iter(events)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def register(self, *_):
        return None

    def select(self, _timeout):
        return next(self.events)


class CaptureProbeTests(unittest.TestCase):
    def test_frame_byte_proves_progress(self):
        selector = lambda: Selector([[(1, 1)]])
        self.assertTrue(
            wait_for_byte(3, 0.1, selector_factory=selector, read=lambda *_: b"x")
        )

    def test_readable_eof_is_failure(self):
        selector = lambda: Selector([[(1, 1)]])
        self.assertFalse(
            wait_for_byte(3, 0.1, selector_factory=selector, read=lambda *_: b"")
        )

    def test_timeout_is_failure(self):
        selector = lambda: Selector([[]])
        self.assertFalse(
            wait_for_byte(3, 0.001, selector_factory=selector, read=lambda *_: b"x")
        )


if __name__ == "__main__":
    unittest.main()
