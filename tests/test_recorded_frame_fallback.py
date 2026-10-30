import numpy as np
import pytest
from perceptbudget.trace_frames import read_recorded_frame


def test_removed_spool_reads_identical_frames_from_lossless_archive(tmp_path):
    spool=tmp_path/'q5_frames_spool.npy'
    frames=np.arange(5*3*4*3,dtype=np.uint8).reshape(5,3,4,3)
    np.save(spool,frames)
    np.savez_compressed(tmp_path/'q5_frames.npz',frames=frames)
    before=read_recorded_frame(spool,3)
    spool.unlink()
    for i in (0,3,4):assert np.array_equal(read_recorded_frame(spool,i),frames[i])
    assert np.array_equal(before,read_recorded_frame(spool,3))
    with pytest.raises(IndexError):read_recorded_frame(spool,5)
    with pytest.raises(IndexError):read_recorded_frame(spool,-1)
