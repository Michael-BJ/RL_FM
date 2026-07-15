import csv, glob, os, re, datetime

TZ = datetime.timezone(datetime.timedelta(hours=8))

def epoch_local(e):
    return datetime.datetime.fromtimestamp(float(e), TZ)

def id_time(s):  # tYYYYMMDDHHMMSSmmm
    return datetime.datetime.strptime(s[1:], '%Y%m%d%H%M%S%f').replace(tzinfo=TZ)

logs = []
for lf in glob.glob(os.path.expanduser('~/.ros/log/python3_*.log')):
    txt = open(lf, errors='replace').read()
    if 'Lap-alternation' not in txt:
        continue
    m0 = re.search(r'\[(\d+\.\d+)\]', txt)
    legs = {1: epoch_local(m0.group(1))}
    for m in re.finditer(r'\[(\d+\.\d+)\].*Lap-alternation leg (\d)/3', txt):
        legs[int(m.group(2))] = epoch_local(m.group(1))
    md = re.search(r'\[(\d+\.\d+)\].*PILOT complete .*\((\d+) samples', txt)
    done = epoch_local(md.group(1)) if md else None
    n = int(md.group(2)) if md else None
    logs.append((legs[1], legs, done, n, lf))
logs.sort()

runs = sorted(glob.glob(os.path.expanduser('~/training_data/run_20260709_*')))

def sigs(s):
    return [] if s in ('free', '') else [int(x) for x in re.split(r'[,_]', s)]

tot = {1: [0, 0], 2: [0, 0]}  # leg -> [compliant, violated]
free_split = {1: 0, -1: 0}
for t0, legs, done, n, lf in logs:
    run = None
    for r in runs:
        rt = datetime.datetime.strptime(os.path.basename(r)[4:], '%Y%m%d_%H%M%S').replace(tzinfo=TZ)
        if abs((rt - t0).total_seconds()) < 90:
            run = r
    tag = os.path.basename(run) if run else '???'
    rows = list(csv.DictReader(open(os.path.join(run, 'train.csv')))) if run else []
    out = [f"{tag} (log {t0:%H:%M:%S}, {n} samples, done={'yes' if done else 'NO'})"]
    for leg, forced in ((1, 1), (2, -1), (3, None)):
        if leg not in legs:
            out.append(f"   leg{leg}: NOT REACHED"); continue
        t_lo = legs[leg]
        t_hi = legs.get(leg + 1) or done or datetime.datetime.max.replace(tzinfo=TZ)
        lr = [r for r in rows if t_lo <= id_time(r['id']) < t_hi]
        nf = [r for r in lr if r['mode_sig'] not in ('free', '')]
        if forced is None:
            pos = sum(1 for r in nf for s in sigs(r['mode_sig']) if s > 0)
            neg = sum(1 for r in nf for s in sigs(r['mode_sig']) if s < 0)
            free_split[1] += pos; free_split[-1] += neg
            out.append(f"   leg3 (free): rows={len(lr)} nonfree={len(nf)} side +1:{pos} -1:{neg}")
        else:
            bad = [r for r in nf if any(s != forced for s in sigs(r['mode_sig']))]
            ok = len(nf) - len(bad)
            tot[leg][0] += ok; tot[leg][1] += len(bad)
            out.append(f"   leg{leg} (forced {forced:+d}): rows={len(lr)} nonfree={len(nf)} compliant={ok} VIOLATED={len(bad)}"
                       + (f"  ids={[r['id'][-9:] for r in bad]}" if bad else ''))
    print('\n'.join(out))
print('=' * 60)
print(f"TOTAL leg1 (left/+1): compliant={tot[1][0]} violated={tot[1][1]}")
print(f"TOTAL leg2 (right/-1): compliant={tot[2][0]} violated={tot[2][1]}")
print(f"TOTAL leg3 free: +1={free_split[1]} vs -1={free_split[-1]}")
