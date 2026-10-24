"""Merge preserved repair summaries and recompute the frozen Holm families.

Requires saved aggregate inputs; performs no estimator fits. See README.md.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', required=True, type=Path,
                        help='Directory containing original_data/ and repair/')
    parser.add_argument('--output', required=True, type=Path,
                        help='New directory outside the inputs and checkout')
    args = parser.parse_args()
    if sys.flags.optimize:
        parser.error('Run without -O: scientific assertions must remain enabled')
    ROOT = args.inputs.resolve()
    H = args.output.resolve()
    checkout = Path(__file__).resolve().parents[1]
    if H.exists() or H == ROOT or ROOT in H.parents or H == checkout or checkout in H.parents:
        parser.error('Output must be a new directory outside the inputs and checkout')
    BASE = ROOT / 'original_data'
    REPAIR = ROOT / 'repair'
    OVER = REPAIR / 'reports/combined'
    required = [BASE / n for n in ['MODEL_METRICS.csv', 'METHOD_CONTRASTS.csv',
                'FACTOR_CONTRASTS.csv', 'AVAILABILITY_ALL_PLANNED.csv']]
    required += [OVER / n for n in ['MODEL_SUMMARIES.jsonl', 'METHOD_CONTRASTS.jsonl',
                 'FACTOR_CONTRASTS.jsonl', 'FRONTIER_PRIMARY_TESTS.json', 'MULTIPLICITY.json']]
    required += [REPAIR / 'ORIGINAL_UNAVAILABLE.json']
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        parser.error('Missing saved inputs: ' + ', '.join(missing))
    from scipy.stats import t
    H.mkdir()  # Existing destinations are never reused.
    (H / 'data').mkdir()
    F = 'frontier|FDH_box|NMSE'
    sources, checks = {}, {}

    def sha(p):
        h = hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda: f.read(1048576), b''):
                h.update(block)
        return h.hexdigest()

    def source(p):
        sources[str(p.relative_to(ROOT))] = sha(p)
        return p

    def readcsv(p):
        with source(p).open() as f:
            return list(csv.DictReader(f))

    def readjson(p):
        return json.loads(source(p).read_text())

    def csvout(name, rows):
        with (H / 'data' / name).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def cp(p, q):
        source(p)
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, q)

    def close(a, b):
        a = None if a in ['', None] else float(a)
        b = None if b in ['', None] else float(b)
        return (a is None and b is None or a is not None and b is not None
                and math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12))

    # Merge only by complete scientific keys; verify each old field against before.
    keyfunc={
     'MODEL_SUMMARIES':lambda r:(r['cell'],r['window'],r['model'],r['metric_id']),
     'METHOD_CONTRASTS':lambda r:(r['cell'],r['window'],r['A'],r['B'],r['metric_id']),
     'FACTOR_CONTRASTS':lambda r:(r.get('contrast_id',r.get('contrast',{}).get('contrast_id')),r['window'],tuple(r['models'].split(' - ')) if isinstance(r['models'],str) else tuple(r['models']),r['metric_id'])}
    data={}
    for kind,csvname in [('MODEL_SUMMARIES','MODEL_METRICS.csv'),('METHOD_CONTRASTS','METHOD_CONTRASTS.csv'),('FACTOR_CONTRASTS','FACTOR_CONTRASTS.csv')]:
     q=source(OVER/(kind+'.jsonl')); overlay={}
     for line in q.open():
      r=json.loads(line);k=keyfunc[kind](r);assert k not in overlay;overlay[k]=r
     cp(q,H/'data'/('REPAIR_'+kind+'.jsonl'))
     rows=readcsv(BASE/csvname); count=0
     for r in rows:
      s=overlay.get(keyfunc[kind](r))
      if s is None:continue
      before=s['before'];mk='mean' if kind=='MODEL_SUMMARIES' else 'mean_difference'
      for old,new in [(mk,'mean'),('MCSE','MCSE')]:assert close(r[old],before.get(new)),(kind,keyfunc[kind](r),old,r[old],before.get(new))
      ci=before.get('CI95') or [None,None]
      assert close(r['CI95_low'],ci[0]) and close(r['CI95_high'],ci[1])
      ci=s.get('CI95') or [None,None]
      r.update({mk:s.get('mean'),'MCSE':s.get('MCSE'),'CI95_low':ci[0],'CI95_high':ci[1]})
      if kind=='MODEL_SUMMARIES':
       for f in ['attempted','available','median','q95']:r[f]=s.get(f)
      elif kind=='METHOD_CONTRASTS':
       r.update(s['availability_2x2']);r['attempted']=s['attempted'];r['negative_fraction']=s.get('negative_fraction')
       r['direction']='unavailable' if s['mean'] is None else 'A_lower' if s['mean']<0 else 'B_lower' if s['mean']>0 else 'exact_equal'
       r['pointwise_interval']='unavailable' if ci[0] is None else 'A_lower' if ci[1]<0 else 'B_lower' if ci[0]>0 else 'interval_contains_zero'
      else:
       if s['contrast']['paired']:
        av=s['availability_2x2'];r.update({'attempted_A':s['attempted'],'attempted_B':s['attempted'],'available_A':av['both'],'available_B':av['both'],'A_only':av['A_only'],'B_only':av['B_only'],'neither':av['neither']})
       else:
        for f in ['attempted_A','attempted_B','available_A','available_B']:r[f]=s[f]
       # Masks and domain provenance are inherited, as they were not changed.
       assert close(r['domain_points_mean'],s['domain_points_mean'])
      count+=1
     csvout(csvname,rows);data[kind]=rows;checks[kind]={'published_extract_rows':len(rows),'replaced_rows':count,'before_fields_checked':True}
    mods=data['MODEL_SUMMARIES'];meth=data['METHOD_CONTRASTS'];fac=data['FACTOR_CONTRASTS']
    assert all(int(r['both'])+int(r['A_only'])+int(r['B_only'])+int(r['neither'])==int(r['attempted']) for r in meth)

    # Retain the original execution availability separately; derived layer is explicit.
    av=readcsv(BASE/'AVAILABILITY_ALL_PLANNED.csv');csvout('ORIGINAL_AVAILABILITY_ALL_PLANNED.csv',av)
    fail=readjson(REPAIR/'ORIGINAL_UNAVAILABLE.json')
    checks['original_unavailable_structure']=type(fail).__name__
    for r in av:
     if r['model']=='FDH':r['numerical_qualified']=r['attempted'];r['states']=json.dumps({'DIRECT_ENUMERATION_QUALIFIED':int(r['attempted'])})
     if r['model']=='StoNED_QLE' and r['cell'] in ['U5_m5_n100_D04','U5_m5_n100_D06']:
      states=json.loads(r['states']);assert states.pop('TIMEOUT')==1;states['QUALIFIED_INTERIOR']=states.get('QUALIFIED_INTERIOR',0)+1;r['states']=json.dumps(states);r['numerical_qualified']=r['attempted']
    assert sum(int(r['numerical_qualified']) for r in av)==270000
    csvout('DERIVED_AVAILABILITY_ALL_PLANNED.csv',av)

    # Recalculate full multiplicity families from current means and denominators.
    # The repair diagnostic primary-test file retained some old ancillary columns;
    # it is used only to cross-check its recalculated p-values, never as row data.
    primary=[dict(r) for r in meth if r['window']=='base_200' and r['metric_id']==F];assert len(primary)==2457
    for r in primary:
     assert int(r['both'])==200 and float(r['MCSE'])>0
     r['t_pvalue']=float(2*t.sf(abs(float(r['mean_difference'])/float(r['MCSE'])),int(r['both'])-1))
     r['core_family']=r['panel']=='P1_core' and r['regime']=='mixed'
     r['holm_core_756']=''
    def holm(rows,field):
     ordered=sorted(rows,key=lambda r:r['t_pvalue']);prev=0
     for i,r in enumerate(ordered):prev=max(prev,min(1.,(len(rows)-i)*r['t_pvalue']));r[field]=prev
    holm(primary,'holm_all_2457');core=[r for r in primary if r['core_family']];assert len(core)==756;holm(core,'holm_core_756')
    diagnostic=readjson(OVER/'FRONTIER_PRIMARY_TESTS.json');lookup={keyfunc['METHOD_CONTRASTS'](r):r for r in diagnostic}
    for r in primary:
     s=lookup[keyfunc['METHOD_CONTRASTS'](r)]
     assert close(r['t_pvalue'],s['p']) and close(r['holm_all_2457'],s['holm_all'])
    csvout('MULTIPLICITY_ALL.csv',primary)
    counts=readjson(OVER/'MULTIPLICITY.json')['core_counts']
    for c in counts:
     rr=[r for r in core if r['A']==c['A'] and r['B']==c['B']];assert len(rr)==36
     for lab,field in [('pointwise','t_pvalue'),('holm_core','holm_core_756'),('holm_all','holm_all_2457')]:
      ns=[sum(r[field]<.05 and float(r['mean_difference'])<0 for r in rr),sum(r[field]<.05 and float(r['mean_difference'])>0 for r in rr),sum(r[field]>=.05 for r in rr)]
      assert ns==c[lab],(c,lab,ns)
    checks['core_counts_verified']=21;checks['all_primary_pairs_complete']=2457


    (H/'REPAIR_MERGE_CHECK.json').write_text(json.dumps(checks,indent=2)+'\n')

    assert all(sha(ROOT / name) == digest for name, digest in sources.items())
    (H / 'INPUT_HASHES.json').write_text(json.dumps(sources, indent=2) + '\n')
    print(json.dumps({'status': 'PASS', 'output': str(H), 'inputs_unchanged': True,
                      'estimator_fits': 0, 'checks': checks}, indent=2))


if __name__ == '__main__':
    main()
