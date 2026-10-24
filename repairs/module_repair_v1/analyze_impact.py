"""Summarize every changed interval direction and diagnose exact-tie sensitivity."""
from pathlib import Path
import collections,csv,json
import numpy as np
from fdh_recompute import H,clean,dump
out=H/'reports/combined'
def direction(r):
    c=r.get('CI95');return 'unavailable' if c is None else 'lower' if c[1]<0 else 'higher' if c[0]>0 else 'includes_zero'
def compact(r):
    return dict(cell=r.get('cell'),contrast_id=r.get('contrast',{}).get('contrast_id'),models=r.get('models',[r.get('A'),r.get('B')]),window=r['window'],metric=r['metric_id'],before_mean=r['before']['mean'],after_mean=r['mean'],before_CI95=r['before']['CI95'],after_CI95=r['CI95'],before_direction=direction(r['before']),after_direction=direction(r))
counts={};changed={};primary=[]
for kind in ['METHOD_CONTRASTS','FACTOR_CONTRASTS']:
    rows=[json.loads(x) for x in (out/(kind+'.jsonl')).open()]
    diff=[r for r in rows if direction(r)!=direction(r['before'])]
    changed[kind]=[compact(r) for r in diff]
    counts[kind]=dict(rows=len(rows),direction_changes=len(diff),by_metric=dict(collections.Counter(r['metric_id'] for r in diff)),primary_frontier_NMSE_direction_changes=sum(r['metric_id']=='frontier|FDH_box|NMSE' for r in diff),frontier_MAE_direction_changes=sum(r['metric_id']=='frontier|FDH_box|MAE' for r in diff))
    if kind=='METHOD_CONTRASTS':primary=[r for r in rows if r['window']=='base_200' and r['metric_id']=='frontier|FDH_box|NMSE']
dump(out/'CHANGED_DIRECTIONS.json',changed)
dump(out/'IMPACT_SUMMARY.json',dict(interval_changes=counts,core_counts_unchanged=json.loads((out/'MULTIPLICITY.json').read_text())['core_counts']==json.loads((H.parent/'reviewer_audit_20261025/MULTIPLICITY_SUMMARY.json').read_text())['core_counts'],primary_comparisons_updated=len(primary),largest_primary_mean_change=max((dict(cell=r['cell'],A=r['A'],B=r['B'],absolute_change=abs(r['mean']-r['before']['mean']),before=r['before']['mean'],after=r['mean']) for r in primary),key=lambda r:r['absolute_change']),note='Direction classifications use the original pointwise intervals. Ancillary rows include repeated windows, correlated metrics, and native/common aliases; change counts are not independent events or multiplicity-adjusted evidence.'))
rank_cells={r['cell'] for r in changed['METHOD_CONTRASTS'] if r['metric'].endswith(('Spearman','Kendall'))}
manifest={r['task']:r for r in json.loads((H/'INPUT_MANIFEST.json').read_text())}
fdh=np.load(H/'fdh/full/PREDICTIONS.npz');diag=collections.defaultdict(list)
for line in (H/'fdh/full/METRICS.jsonl').open():
    r=json.loads(line)
    if r['cell'] not in rank_cells or not r['old_available'] or r['rep']>200:continue
    p=Path(manifest[r['task']]['directory']);n=manifest[r['task']]['n']
    with np.load(p/'analysis_truth.npz') as z:y=z['y']
    with np.load(p/'ALL_PREDICTIONS.npz') as z:old=z['FDH_common_raw']
    new=y/fdh[r['task']][:n];delta=abs(old-new)
    # A separate diagnostic: rounding is NOT adopted for the study's rank metrics.
    rec=dict(task=r['task'],maximum_score_change=float(delta.max()),scores_changed_over_1e8=int((delta>1e-8).sum()),rounded8_scores_identical=bool(np.array_equal(np.round(old,8),np.round(new,8))),before_Spearman=r['FDH_before']['TE_common_raw|training|Spearman'],after_Spearman=r['FDH_after']['TE_common_raw|training|Spearman'],before_Kendall=r['FDH_before']['TE_common_raw|training|Kendall'],after_Kendall=r['FDH_after']['TE_common_raw|training|Kendall'])
    diag[r['cell']].append(rec)
fdh.close()
summary=[]
for cell,rs in sorted(diag.items()):
    row=dict(cell=cell,originally_available_datasets=len(rs),max_score_change=max(r['maximum_score_change'] for r in rs),scores_changed_over_1e8=sum(r['scores_changed_over_1e8'] for r in rs),datasets_with_identical_rounded8_scores=sum(r['rounded8_scores_identical'] for r in rs))
    for metric in ['Spearman','Kendall']:
        valid=[r for r in rs if r['before_'+metric] is not None and r['after_'+metric] is not None]
        row[metric+'_mean_change_on_original_available_cases']=float(np.mean([r['after_'+metric]-r['before_'+metric] for r in valid])) if valid else None
    summary.append(row)
dump(out/'RANK_SENSITIVITY.json',dict(scope='Original-available cases in cells whose base method-rank interval direction changed; excludes newly recovered observations to isolate route sensitivity',diagnostic_rounding_decimals=8,rounding_adopted_for_metrics=False,cells=summary,interpretation='Exact rank and equality statistics can move under very small score changes at ties. Keep this limitation explicit; do not describe every auxiliary conclusion as unchanged.'))
print(json.dumps(json.loads((out/'IMPACT_SUMMARY.json').read_text()),indent=2));print(json.dumps(summary,indent=2))
