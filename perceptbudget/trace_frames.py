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


def read_recorded_frame(spool, index):
    """Read one frame from the mmap spool or its lossless NPZ replacement.

    Seeking inside the compressed NPY avoids expanding an entire episode in RAM.
    """
    from pathlib import Path
    from zipfile import ZipFile
    import numpy as np
    spool=Path(spool)
    if type(index) is not int or index<0:
        raise IndexError('Frame index must be nonnegative')
    if spool.exists():
        frames=np.load(spool,mmap_mode='r',allow_pickle=False)
        return frames[index].copy()
    archive=spool.with_name(spool.name.replace('_frames_spool.npy','_frames.npz'))
    with ZipFile(archive) as saved, saved.open('frames.npy') as stream:
        version=np.lib.format.read_magic(stream)
        if version==(1,0):shape,fortran,dtype=np.lib.format.read_array_header_1_0(stream)
        elif version==(2,0):shape,fortran,dtype=np.lib.format.read_array_header_2_0(stream)
        else:raise ValueError('Unsupported recorded frame format')
        if fortran or dtype.hasobject or len(shape)<2:
            raise ValueError('Recorded frames must be C-order numeric arrays')
        if index>=shape[0]:raise IndexError('Frame index beyond recorded episode')
        size=int(np.prod(shape[1:]))*dtype.itemsize
        stream.seek(stream.tell()+index*size)
        data=stream.read(size)
        if len(data)!=size:raise ValueError('Truncated recorded frame')
        return np.frombuffer(data,dtype=dtype).reshape(shape[1:]).copy()
