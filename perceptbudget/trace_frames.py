"""Bounded-RAM raw frame recording for diagnostic rollouts."""
import mmap


class DiskFrames:
    def __init__(self, path, capacity, np):
        self.path, self.capacity, self.np = path, capacity, np
        self.data = None
        self.count = 0
        if path.exists():
            raise FileExistsError(path)

    def __len__(self):
        return self.count

    def append(self, frame):
        if self.count >= self.capacity:
            raise ValueError('frame recording capacity exceeded')
        if self.data is None:
            self.data = self.np.lib.format.open_memmap(
                self.path, mode='w+', dtype=frame.dtype,
                shape=(self.capacity,)+frame.shape)
        if frame.shape != self.data.shape[1:] or frame.dtype != self.data.dtype:
            raise ValueError('frame shape/dtype changed')
        self.data[self.count] = frame
        self.count += 1
        # Clean, file-backed pages can be reclaimed instead of accumulating
        # alongside the Jetson GPU's shared-memory allocations.
        if self.count % 8 == 0:
            self.data.flush()
            if hasattr(self.data._mmap, 'madvise'):
                self.data._mmap.madvise(mmap.MADV_DONTNEED)

    def pop(self):
        if not self.count:
            raise IndexError('empty frames')
        self.count -= 1

    def __getitem__(self, index):
        return self.data[:self.count][index]

    def __array__(self, dtype=None, copy=None):
        if self.data is None:
            return self.np.empty((0,), dtype=self.np.uint8 if dtype is None else dtype)
        self.data.flush()
        result = self.data[:self.count]
        if dtype is not None:
            result = result.astype(dtype, copy=False)
        return result.copy() if copy else result
