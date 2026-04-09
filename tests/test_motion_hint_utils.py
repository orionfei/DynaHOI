import unittest

import numpy as np

from gr00t.utils.motion_hint import (
    MOTION_HINT_ALGORITHM,
    MOTION_HINT_CACHE_SUBDIR,
    compute_motion_hint_from_frames,
    format_motion_hint_ratio_tag,
    get_motion_hint_cache_dir,
    get_motion_hint_image_path,
    get_motion_hint_manifest_path,
    get_motion_hint_sample_indices,
    resolve_motion_hint_worker_count,
)


class MotionHintUtilsTest(unittest.TestCase):
    def test_sample_indices_use_fixed_stride(self):
        indices = get_motion_hint_sample_indices(prefix_length=16)
        self.assertTrue(np.array_equal(indices, np.array([0, 5, 10, 15], dtype=np.int64)))
        self.assertIsNone(get_motion_hint_sample_indices(prefix_length=5))

    def test_compute_motion_hint_from_frames(self):
        frames = np.zeros((4, 8, 8, 3), dtype=np.uint8)
        frames[0, 2:4, 2:4, 0] = 255
        frames[1, 2:4, 3:5, 0] = 255
        frames[2, 2:4, 4:6, 1] = 255
        frames[3, 2:4, 5:7, 2] = 255

        motion_hint = compute_motion_hint_from_frames(frames)
        self.assertEqual(motion_hint.dtype, np.uint8)
        self.assertEqual(motion_hint.shape, (8, 8, 3))
        self.assertTrue(np.any(motion_hint[..., 0] > 0))
        self.assertTrue(np.any(motion_hint[..., 1] > 0))
        self.assertTrue(np.any(motion_hint[..., 2] > 0))

    def test_zero_motion_raises(self):
        frames = np.zeros((2, 4, 4, 3), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "identically zero"):
            compute_motion_hint_from_frames(frames)

    def test_cache_path_helpers(self):
        self.assertEqual(format_motion_hint_ratio_tag(0.2), "ratio_0p200000")
        cache_dir = get_motion_hint_cache_dir("/tmp/example_dataset", 0.2)
        self.assertEqual(
            str(cache_dir),
            f"/tmp/example_dataset/{MOTION_HINT_CACHE_SUBDIR}/ratio_0p200000",
        )
        image_path = get_motion_hint_image_path(cache_dir, 12)
        self.assertEqual(
            str(image_path),
            f"/tmp/example_dataset/{MOTION_HINT_CACHE_SUBDIR}/ratio_0p200000/episode_000012.png",
        )
        manifest_path = get_motion_hint_manifest_path(cache_dir)
        self.assertEqual(
            str(manifest_path),
            f"/tmp/example_dataset/{MOTION_HINT_CACHE_SUBDIR}/ratio_0p200000/manifest.json",
        )

    def test_worker_count_auto_mode(self):
        self.assertEqual(resolve_motion_hint_worker_count(0, logical_cpu_count=16), 8)
        self.assertEqual(resolve_motion_hint_worker_count(0, logical_cpu_count=6), 3)
        self.assertEqual(resolve_motion_hint_worker_count(3, logical_cpu_count=16), 3)

    def test_algorithm_name_is_stable(self):
        self.assertEqual(MOTION_HINT_ALGORITHM, "rgb_absdiff_prefix_stride5_sum_v1")


if __name__ == "__main__":
    unittest.main()
