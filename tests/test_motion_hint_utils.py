import unittest

import numpy as np

from gr00t.data.dataset import (
    _aggregate_weighted_farneback_flows,
    _compute_motion_hint_frame_count,
    _encode_motion_hint,
    compute_motion_hint_from_frames,
    get_motion_hint_cache_dir,
    get_motion_hint_image_path,
)
from gr00t.utils.video import get_uniform_sample_indices


class MotionHintUtilsTest(unittest.TestCase):
    def test_compute_motion_hint_frame_count(self):
        self.assertEqual(_compute_motion_hint_frame_count(20, 0.2), 4)
        self.assertEqual(_compute_motion_hint_frame_count(10, 0.2), 2)
        self.assertEqual(_compute_motion_hint_frame_count(9, 0.2), 2)

    def test_uniform_sample_indices(self):
        indices = get_uniform_sample_indices(total_frames=10, num_sampled_frames=6)
        self.assertTrue(np.array_equal(indices, np.array([0, 2, 4, 5, 7, 9], dtype=np.int64)))
        self.assertIsNone(get_uniform_sample_indices(total_frames=5, num_sampled_frames=6))

    def test_aggregate_weighted_farneback_flows(self):
        flow_1 = np.zeros((2, 2, 2), dtype=np.float32)
        flow_2 = np.zeros((2, 2, 2), dtype=np.float32)
        flow_1[..., 0] = 1.0
        flow_1[..., 1] = 2.0
        flow_2[..., 0] = 3.0
        flow_2[..., 1] = 4.0

        u_bar, v_bar, magnitude = _aggregate_weighted_farneback_flows([flow_1, flow_2])
        self.assertTrue(np.allclose(u_bar, 3.5))
        self.assertTrue(np.allclose(v_bar, 5.0))
        self.assertTrue(np.allclose(magnitude, np.sqrt(3.5**2 + 5.0**2)))

    def test_encode_motion_hint(self):
        u_bar = np.array([[-1.0, 0.0, 1.0]], dtype=np.float32)
        v_bar = np.array([[-2.0, 0.0, 2.0]], dtype=np.float32)
        magnitude = np.array([[0.0, 1.0, 2.0]], dtype=np.float32)

        encoded = _encode_motion_hint(u_bar, v_bar, magnitude)
        self.assertEqual(encoded.dtype, np.uint8)
        self.assertEqual(encoded.shape, (1, 3, 3))
        self.assertTrue(np.array_equal(encoded[0, :, 0], np.array([0, 128, 255], dtype=np.uint8)))
        self.assertTrue(np.array_equal(encoded[0, :, 1], np.array([0, 128, 255], dtype=np.uint8)))
        self.assertTrue(np.array_equal(encoded[0, :, 2], np.array([0, 128, 255], dtype=np.uint8)))

    def test_cache_path_helpers(self):
        cache_dir = get_motion_hint_cache_dir("/tmp/example_dataset", 0.2)
        self.assertEqual(str(cache_dir), "/tmp/example_dataset/meta/motion_hint_farneback/ratio_0p200000")
        image_path = get_motion_hint_image_path(cache_dir, 12)
        self.assertEqual(str(image_path), "/tmp/example_dataset/meta/motion_hint_farneback/ratio_0p200000/episode_000012.png")

    def test_compute_motion_hint_from_frames(self):
        frames = np.zeros((3, 8, 8, 3), dtype=np.uint8)
        frames[0, 2:4, 2:4] = 255
        frames[1, 3:5, 3:5] = 255
        frames[2, 4:6, 4:6] = 255
        motion_hint = compute_motion_hint_from_frames(frames)
        self.assertEqual(motion_hint.dtype, np.uint8)
        self.assertEqual(motion_hint.shape, (8, 8, 3))


if __name__ == "__main__":
    unittest.main()
