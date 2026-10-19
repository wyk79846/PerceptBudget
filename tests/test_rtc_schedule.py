from perceptbudget.rtc_schedule import DelayQueue


def test_history_max_and_eviction():
    history = DelayQueue(8, size=3)
    assert history.estimate == 8
    history.observe(6)
    history.observe(7)
    assert history.estimate == 8
    history.observe(5)
    assert history.estimate == 7
    assert list(history.values) == [6, 7, 5]
