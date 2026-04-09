import unittest

from gr00t.utils.history import (
    format_observe_frame_history_tag,
    resolve_observe_frame_history,
)


class ObserveFrameHistoryConfigTest(unittest.TestCase):
    def test_defaults_to_contiguous_history(self):
        history = resolve_observe_frame_history(5, None)
        self.assertEqual(history.offsets, (5, 4, 3, 2, 1))
        self.assertEqual(history.frame_count, 5)
        self.assertEqual(history.start_index, 5)

    def test_accepts_explicit_offsets(self):
        history = resolve_observe_frame_history(5, [10, 5, 3, 2, 1])
        self.assertEqual(history.offsets, (10, 5, 3, 2, 1))
        self.assertEqual(history.frame_count, 5)
        self.assertEqual(history.start_index, 10)

    def test_rejects_invalid_offsets(self):
        with self.assertRaisesRegex(ValueError, "same length as window_length"):
            resolve_observe_frame_history(5, [10, 5, 3])
        with self.assertRaisesRegex(ValueError, "positive integers"):
            resolve_observe_frame_history(3, [3, 1, 0])
        with self.assertRaisesRegex(ValueError, "must not contain duplicates"):
            resolve_observe_frame_history(3, [5, 5, 1])
        with self.assertRaisesRegex(ValueError, "strictly decreasing"):
            resolve_observe_frame_history(3, [3, 4, 1])

    def test_result_tag_uses_offsets_when_explicit(self):
        self.assertEqual(format_observe_frame_history_tag(5, None), "window_5")
        self.assertEqual(
            format_observe_frame_history_tag(5, [10, 5, 3, 2, 1]),
            "offsets_10-5-3-2-1",
        )


if __name__ == "__main__":
    unittest.main()
