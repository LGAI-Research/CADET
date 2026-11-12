"""Density calibration statistic: target (fit_params boards) vs realized (generated dir) pins/cm^2. The reported axis is the
median, and the gate truncates the generated distribution (not lognormal), so the default ratio is target_p50 / realized_p50
(with a 2-4k pilot the p50 standard error is ~2%); --estimator gm uses exp(mean(log x)) on both sides instead.

    python cadet/generation/calib_stats.py <fit_params.json> <generated dir> [--json]
prints: ratio=<target/realized> target_gm=.. realized_gm=.. n=.. realized_p50=.. target_p50=..
"""
import argparse, json, math, statistics, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from difficulty_report import gen, real_rows  # noqa: E402


def gmean(v): return math.exp(statistics.fmean(math.log(x) for x in v if x > 0))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("fit"); ap.add_argument("gen_dir"); ap.add_argument("--json", action="store_true")
    ap.add_argument("--estimator", choices=["p50", "gm"], default="p50")
    a = ap.parse_args()
    t = [r["pins_cm2"] for r in real_rows(a.fit)]; g = [r["pins_cm2"] for r in gen(a.gen_dir)]
    if not g: sys.exit(f"no generated boards under {a.gen_dir}")
    q = lambda v, p: sorted(v)[int(p * (len(v) - 1))]
    ratio = q(t, .5) / q(g, .5) if a.estimator == "p50" else gmean(t) / gmean(g)
    out = dict(ratio=round(ratio, 4), estimator=a.estimator, target_gm=round(gmean(t), 3), realized_gm=round(gmean(g), 3), n=len(g),
               target_p50=round(q(t, .5), 3), realized_p10=round(q(g, .1), 3), realized_p50=round(q(g, .5), 3), realized_p90=round(q(g, .9), 3),
               se_pct=round(100 * statistics.pstdev([math.log(x) for x in g if x > 0]) / math.sqrt(len(g)), 2))
    print(json.dumps(out) if a.json else " ".join(f"{k}={v}" for k, v in out.items()))


if __name__ == "__main__":
    main()
