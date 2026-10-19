"""Independent paper counter vs production absolute-step queue handoff."""
from scripts.probe_delay_buffer import commit, refresh_due
from perceptbudget.rtc_schedule import DelayQueue


def test_paper_relative_counter_matches_absolute_queue_across_variable_delays():
    horizon, minimum = 20, 5
    queue = {}
    current_chunk = list(range(horizon))
    commit(queue, current_chunk, 0, 0, 0)
    origin = previous_request = 0
    delays = iter((8, 6, 7, 5, 8, 7))
    ready = None
    request_origin = None
    next_chunk = None
    history = DelayQueue(8)
    paper_delays = [8]
    prediction = 0
    for step in range(46):
        if ready == step:
            s = request_origin-origin
            # Algorithm 1: after A_cur=A_new, t <- t-s.
            old_t = step-origin
            paper_t = old_t-s
            assert paper_t == step-request_origin
            current_chunk, origin = next_chunk, request_origin
            commit(queue, next_chunk, origin, step, prediction)
            history.observe(paper_t)
            paper_delays = (paper_delays+[paper_t])[-10:]
            ready = None
        paper_t = step-origin
        due = refresh_due(step, previous_request, minimum, 'latest_due')
        if ready is None:
            assert due == (paper_t >= minimum)
            if due:
                assert history.estimate == max(paper_delays)
                old_tail = [queue[i][0] for i in range(step,origin+horizon)]
                assert old_tail == current_chunk[paper_t:]
                request_origin = step
                previous_request = step
                prediction += 1
                next_chunk = [prediction*100+i for i in range(horizon)]
                ready = step+next(delays)
        # GETACTION consumes current chunk's *actual* relative index,
        # never the estimated inference delay.
        assert queue.pop(step)[0] == current_chunk[step-origin]
