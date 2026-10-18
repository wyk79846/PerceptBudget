from pathlib import Path
import tempfile
import unittest
import numpy as np
from perceptbudget.trace_frames import DiskFrames


class DiskFramesTest(unittest.TestCase):
    def test_record_pop_and_npz_keep_exact_raw_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            frames = DiskFrames(Path(directory)/'frames.npy', 10, np)
            expected = [np.full((3, 4, 3), i, dtype=np.uint8) for i in range(9)]
            for frame in expected:
                frames.append(frame)
            frames.pop()
            self.assertEqual(len(frames), 8)
            np.testing.assert_array_equal(np.asarray(frames), np.asarray(expected[:8]))
            output = Path(directory)/'frames.npz'
            np.savez_compressed(output, frames=np.asarray(frames))
            with np.load(output) as archive:
                np.testing.assert_array_equal(archive['frames'], np.asarray(expected[:8]))
            np.testing.assert_array_equal(frames[0], expected[0])

    def test_capacity_and_existing_files_are_not_silently_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'frames.npy'
            frames = DiskFrames(path, 1, np)
            frames.append(np.zeros((2, 2, 3), dtype=np.uint8))
            with self.assertRaises(ValueError):
                frames.append(np.zeros((2, 2, 3), dtype=np.uint8))
            with self.assertRaises(FileExistsError):
                DiskFrames(path, 1, np)
