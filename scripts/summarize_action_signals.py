#!/usr/bin/env python3
"""Pool existing action records only. No model, environment, CUDA or rollout."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from perceptbudget.action_events import EventConfig
from perceptbudget.action_signal_experiment import (
    METRICS, VERSION, ALIGNED_VERSION, extract, prepare, overlap_summary,
)


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    i = (len(values)-1)*q
    low = int(i)
    return values[low]+(values[min(low+1, len(values)-1)]-values[low])*(i-low)


def packets(index):
    for row in index['rows']:
        path = Path(row['run'])/'rtc/q5_requests.json'
        data = json.loads(path.read_text())
        requests = data if isinstance(data, list) else data['requests']
        by_id = {r['prediction_id']: r for r in requests}
        seen = set()
        for request in requests:
            timing = request.get('m3_timing', {})
            proof = timing.get('critical_forecast')
            if not proof or proof['prediction_id'] in seen:
                continue
            pid = proof['prediction_id']; seen.add(pid)
            current = by_id[pid]
            prior_ids = [i for i in by_id if i < pid]
            prior = by_id[max(prior_ids)] if prior_ids else None
            reference = dict(origin=prior['observation_step'], actions=prior['actions']) if prior else None
            yield dict(actions=current['actions'], origin=current['observation_step'],
                source_step=proof['source_step'], history={}, reference=reference,
                config=EventConfig(), period=proof['source_period_seconds'],
                response_seconds=timing['feedback_review']['response_seconds'],
                margin_seconds=timing['margin_seconds'], prediction_count=pid+1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--version', choices=(VERSION, ALIGNED_VERSION), default=VERSION)
    parser.add_argument('--no-refresh-budget', action='store_true')
    args = parser.parse_args()
    index = json.loads(args.index.read_text())
    distributions = defaultdict(list)
    chunks = 0; overlap = 0
    aligned = args.version == ALIGNED_VERSION
    for p in packets(index):
        rows = extract(**{k: p[k] for k in ('actions','origin','source_step','history','reference','config')},
                       within_plan_only=aligned)
        chunks += 1
        for f in rows:
            overlap += f['reference_overlap']
            for k in METRICS[:2]:
                if f[k] is not None:
                    distributions[k].append(f[k])
        evidence_rows = rows if aligned else [overlap_summary(rows)]
        for evidence in evidence_rows:
            for k in METRICS[2:]:
                if evidence[k] is not None:
                    distributions[k].append(evidence[k])
    if any(not distributions[k] for k in METRICS):
        raise ValueError('Need existing action changes and overlapping predictions for all four metrics')
    thresholds = {k: max(1e-6, percentile(distributions[k], .9)) for k in METRICS}
    settings = dict(version=args.version, thresholds=thresholds, minimum_request_steps=20,
        minimum_mean_steps=22, threshold_quantile=.9, source_index=str(args.index.resolve()),
        calibration='All available matching M4 episodes, pooled without success/task/seed labels; no parameter search',
        aggregation='Motion changes: per two-step window; plan disagreements: maximum sustained overlap per chunk',
        consumed_prefix='excluded; first two rows without executed history not used for change quantiles')
    if aligned:
        settings.update(aggregation='Both changes and disagreements: per two-step window, same absolute steps',
            consumed_prefix='excluded offline and online; handoff history never supplies new-cue changes')
    if args.no_refresh_budget:
        settings['refresh_budget_enabled'] = False
    counts = Counter()
    for p in packets(index):
        r = prepare(**p, settings=settings)['review']
        counts['packets_with_supported_candidate'] += r['candidate'] is not None
        counts['two_family_steps'] += r['opportunity_count']
        counts['steps_rejected_by_refresh_budget'] += r['budget_rejected']
    episodes = index['rows']
    intervals = sum(max(0, r['requests']-1) for r in episodes)
    mean_interval = sum(r['mean_refresh_interval_steps']*max(0, r['requests']-1)
                        for r in episodes)/intervals
    report = dict(episodes=len(episodes), baseline_successes=sum(r['outcome']=='success' for r in episodes),
        baseline_mean_refresh_steps=mean_interval, packet_count=chunks, overlap_steps=overlap,
        feature_distributions={k: dict(count=len(v), median=percentile(v,.5), p90=percentile(v,.9),
                                      p95=percentile(v,.95)) for k,v in distributions.items()},
        candidate_counts=dict(counts),
        scope='Offline feature coverage only; not new success, energy or refresh-interval measurements')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'settings.json').write_text(json.dumps(settings, indent=2)+'\n')
    (args.output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
