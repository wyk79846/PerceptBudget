from scripts.eval_lerobot_rtc import prediction_ready


class Future:
    def __init__(self,done):
        self.completed=done
        self.waits=0
    def done(self):return self.completed
    def result(self):
        self.waits+=1
        self.completed=True


def test_controlled_waits_at_five_steps_and_never_commits_early():
    early=Future(True)
    assert not prediction_ready(early,9,5,True)
    assert early.waits==0
    late=Future(False)
    assert prediction_ready(late,10,5,True)
    assert late.waits==1


def test_actual_mode_never_waits_for_future():
    future=Future(False)
    assert not prediction_ready(future,20,5,False)
    assert future.waits==0
    future.completed=True
    assert prediction_ready(future,20,5,False)
    assert future.waits==0
    assert not prediction_ready(None,0,0,True)
