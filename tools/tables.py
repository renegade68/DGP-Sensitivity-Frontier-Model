"""A small raw-to-reader table; production claims require the full run audit."""
from context import *
LABELS={'EAT':'Efficiency analysis trees','CEAT':'Convexified efficiency analysis trees','SVF_PAPER':'Support vector frontier','FDH':'Free disposal hull','DEA_VRS':'Data envelopment analysis (VRS)','SFA_CD_HN':'Stochastic frontier: Cobb–Douglas, half-normal','SFA_TL_HN':'Stochastic frontier: translog, half-normal','SFA_CD_EXP':'Stochastic frontier: Cobb–Douglas, exponential','SFA_TL_EXP':'Stochastic frontier: translog, exponential','StoNED_QLE':'StoNED (quasi-likelihood)'}
def build(task,out):
    task,out=Path(task),Path(out);out.mkdir(parents=True,exist_ok=False);r=read(task/'METRICS.json');key='frontier|FDH_box|NMSE';rows=[]
    for m in MODELS:
        v=r['values'][m].get(key);rows.append(dict(label=LABELS[m],model_id=m,metric_id=key,source_value=v,display='NA' if v is None else format(v,'.8f')))
    write(out/'TABLE_SOURCE.json',dict(task_id=r['task_id'],parent_metrics_sha256=sha(task/'METRICS.json'),rows=rows,engineering_example=True))
    lines=['Reproducibility example: one generated dataset; not Monte Carlo estimates.','Normalized frontier MSE on the common input domain.','']+[x['label']+'\t'+x['display'] for x in rows]
    (out/'example_table.txt').write_text('\n'.join(lines)+'\n')
    # Parse displayed values against the locked full-precision source.
    parsed=(out/'example_table.txt').read_text().splitlines()[3:]
    assert all(p==x['label']+'\t'+('NA' if x['source_value'] is None else format(x['source_value'],'.8f')) for p,x in zip(parsed,rows))
    write(out/'TABLE_AUDIT.json',dict(status='PASS',rows=len(rows),source=sha(out/'TABLE_SOURCE.json'),display=sha(out/'example_table.txt'),rounding='display only; eight decimal places',hand_transcription=False))
if __name__=='__main__':build(sys.argv[1],sys.argv[2])
