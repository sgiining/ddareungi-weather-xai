"""[실험 4] 예측 근거 분석 — SHAP, 기온 반응 곡선, 오차 큰 날 (계획서 3.4절 → 4.5.3절)

실행: 프로젝트 폴더에서  py experiments/04_shap.py
점검: py experiments/04_shap.py --quick   (시드 2개, 트리 50개, 결과 파일에 _quick)
      --quick 은 02_main.py --quick 결과(02_predictions_quick.csv)가 있어야 오차 큰 날까지 확인된다.

대상 모델: 기상 추가 RF(P), 01_tune_selected.json 설정, 시드 5개, 두 테스트 기간 (2025, 2026H1)

설정 (eda/DECISIONS.md 8번)
- SHAP: shap.TreeExplainer(model, feature_perturbation="tree_path_dependent"), 기준 데이터 없음
- 시드 5개 모델의 SHAP을 평균해 대표 설명으로 쓴다 (= 시드 평균 예측의 SHAP).
  시드별 중요도는 6단계(설명 안정성, Jaccard)를 위해 따로 저장한다.
- 원-핫 열(월 12, 요일 7)은 더해서 월·요일 단위로 본다. 날씨 묶음은 부호 그대로 더한다.
- 정답이 log(1+y)면 SHAP은 로그 단위다. exp(φ) − 1 을 "예측을 약 몇 % 바꿨는지"로 읽는다.
- 기온 반응 곡선: 일교차 유지, 학습 관측 범위 안, 0.5°C 간격, centered ICE (analysis_rules.py)
- 오차 큰 날: 02_predictions.csv 의 시드 평균 예측, 강건 z > 3, 과대·과소예측 각 상위 3일

결과
- results/tables/04_importance.csv            : 변수·묶음 중요도 (평균 |SHAP|, 평균 SHAP, %)
- results/tables/04_importance_by_seed.csv    : 시드별 중요도·순위 (6단계 Jaccard 입력)
- results/tables/04_by_segment.csv            : 평일/쉬는 날, 계절별 변수·묶음 기여도
- results/tables/04_shap_values_{기간}.csv     : 날짜별 시드 평균 SHAP (변수 15개 + 묶음 5개)
- results/tables/04_large_error_days_{기간}.csv: 오차 큰 날 목록과 주요 SHAP 변수
- results/tables/04_ice_temp_{기간}.csv        : 기온 반응 평균 곡선 (전체, 평일/쉬는 날, 계절)
- results/figures/04_importance_{기간}.png, 04_segment_{기간}.png, 04_rain_scatter_{기간}.png,
  04_ice_temp_{기간}.png (평일/쉬는 날 평균), 04_ice_temp_season_{기간}.png (계절 평균),
  04_ice_temp_days_{기간}.png (날짜별 곡선 + 전체 평균), 04_error_days_{기간}.png
  (--quick 이면 그림 이름에도 _quick 이 붙는다)
"""

import argparse
import time

import numpy as np
import pandas as pd
import shap

from analysis_rules import select_large_error_days, temperature_ice
from common import (FIGURES, MODEL_FEATURES, PERIOD_LABEL, SEEDS, TABLES, TEST_STAGES, WEATHER_GROUPS,
                    ensure_dirs, fit_predict_rf, get_split, load_dataset, load_selected, make_X,
                    setup_matplotlib, shap_by_variable, shap_by_weather_group)

VAR_NAMES = {
    "year": "연도", "month": "월", "dow": "요일", "is_weekend": "주말", "is_holiday": "공휴일",
    "is_offday": "쉬는 날", "max_temp": "최고기온", "min_temp": "최저기온", "rain_mm": "일강수량",
    "rain_hr": "강수 계속시간", "humidity": "상대습도", "sunshine_hr": "일조시간", "cloud": "전운량",
    "wind_ms": "풍속", "new_snow_cm": "신적설",
}
GROUP_NAMES = {"temperature": "기온", "precipitation": "강수", "humidity_sky": "습도·하늘",
               "wind": "바람", "snow": "눈"}
SEASON_NAMES = {"spring": "봄", "summer": "여름", "autumn": "가을", "winter": "겨울"}
OFFDAY_NAMES = {0: "평일", 1: "쉬는 날"}
PERIOD_NAMES = {"2025": "2025년", "2026H1": "2026년 상반기"}
HOT_LIMIT = 35.0   # 이 기온 초과 구간은 해석하지 않는다 (DECISIONS.md 6번)
FIG_SUFFIX = ""    # --quick 이면 "_quick"


# ---------------------------------------------------------------------------
# 학습과 SHAP
# ---------------------------------------------------------------------------
def to_count(pred_log_or_raw, target):
    return np.expm1(pred_log_or_raw) if target == "log1p" else pred_log_or_raw


def fit_models(params, seeds, X_tr, y_tr, X_ev, n_estimators):
    models, preds = [], []
    for seed in seeds:
        model, p = fit_predict_rf(params, seed, X_tr, y_tr, X_ev, n_estimators)
        models.append((seed, model))
        preds.append(p)
    return models, np.column_stack(preds)


def explain(models, X_ev, features):
    """시드별 변수 단위 SHAP 목록, 시드 평균 SHAP, 평균 기준값"""
    per_seed, bases = [], []
    for seed, model in models:
        explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
        sv = np.asarray(explainer.shap_values(X_ev))
        if sv.ndim == 3:            # 일부 shap 버전은 (n, p, 1) 로 돌려준다
            sv = sv[..., 0]
        per_seed.append(shap_by_variable(sv, X_ev.columns, features))
        bases.append(float(np.ravel(explainer.expected_value)[0]))
        print(f"    SHAP 시드 {seed} 완료")
    mean = sum(per_seed) / len(per_seed)
    return per_seed, mean, float(np.mean(bases))


def importance_rows(sv_var: pd.DataFrame, target: str):
    rows = []
    groups = shap_by_weather_group(sv_var)
    for level, table, names in (("variable", sv_var, VAR_NAMES), ("group", groups, GROUP_NAMES)):
        for col in table.columns:
            signed = float(table[col].mean())
            rows.append({"level": level, "name": col, "name_ko": names[col],
                         "mean_abs": float(table[col].abs().mean()), "mean_signed": signed,
                         "mean_signed_pct": (np.expm1(signed) * 100) if target == "log1p" else np.nan})
    out = pd.DataFrame(rows)
    out["rank"] = out.groupby("level")["mean_abs"].rank(ascending=False, method="first").astype(int)
    return out.sort_values(["level", "rank"], ascending=[False, True])


def segment_rows(sv_var: pd.DataFrame, ev: pd.DataFrame):
    groups = shap_by_weather_group(sv_var)
    both = pd.concat([sv_var, groups], axis=1)
    rows = []
    for seg_col, names in (("is_offday", OFFDAY_NAMES), ("season", SEASON_NAMES)):
        for val, idx in ev.reset_index(drop=True).groupby(seg_col).groups.items():
            part = both.loc[idx]
            for col in both.columns:
                level = "group" if col in WEATHER_GROUPS else "variable"
                rows.append({"segment_type": seg_col, "segment": val, "segment_ko": names[val],
                             "n_days": len(idx), "level": level, "name": col,
                             "name_ko": GROUP_NAMES.get(col, VAR_NAMES.get(col)),
                             "mean_abs": float(part[col].abs().mean()),
                             "mean_signed": float(part[col].mean())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 그림
# ---------------------------------------------------------------------------
def unit_label(target):
    return "SHAP (log(1+대여건수) 단위)" if target == "log1p" else "SHAP (건)"


def plot_importance(plt, imp, period, target):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [3, 2]})
    for ax, level, title in ((axes[0], "variable", "변수별"), (axes[1], "group", "날씨 묶음별")):
        d = imp[imp["level"] == level].sort_values("mean_abs")
        colors = ["#1D9E75" if (level == "group" or n in sum(WEATHER_GROUPS.values(), [])) else "#888780"
                  for n in d["name"]]
        ax.barh(d["name_ko"], d["mean_abs"], color=colors)
        ax.set_title(f"{title} 평균 |SHAP|")
        ax.set_xlabel(unit_label(target))
    fig.suptitle(f"기상 추가 랜덤 포레스트 예측 근거 중요도 ({PERIOD_NAMES[period]}, 시드 평균)  초록=날씨, 회색=달력")
    fig.tight_layout()
    fig.savefig(FIGURES / f"04_importance_{period}{FIG_SUFFIX}.png")
    plt.close(fig)


def plot_segments(plt, seg, period, target):
    g = seg[seg["level"] == "group"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, seg_col, order in ((axes[0], "is_offday", [0, 1]), (axes[1], "season", list(SEASON_NAMES))):
        d = g[g["segment_type"] == seg_col]
        order = [o for o in order if o in set(d["segment"])]
        width = 0.8 / len(order)
        x = np.arange(len(GROUP_NAMES))
        for i, val in enumerate(order):
            part = d[d["segment"] == val].set_index("name").reindex(list(GROUP_NAMES))
            label = f"{part['segment_ko'].iloc[0]} ({int(part['n_days'].iloc[0])}일)"
            ax.bar(x + i * width - 0.4 + width / 2, part["mean_abs"], width, label=label)
        ax.set_xticks(x)
        ax.set_xticklabels(GROUP_NAMES.values())
        ax.legend(fontsize=8)
        ax.set_title("평일 · 쉬는 날" if seg_col == "is_offday" else "계절")
    axes[0].set_ylabel(f"평균 |묶음 SHAP|\n{unit_label(target)}")
    fig.suptitle(f"날씨 묶음 기여도의 조건별 비교 ({PERIOD_NAMES[period]})")
    fig.tight_layout()
    fig.savefig(FIGURES / f"04_segment_{period}{FIG_SUFFIX}.png")
    plt.close(fig)


def plot_rain(plt, ev, groups, period, target):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for val, name, color in ((0, "평일", "#378ADD"), (1, "쉬는 날", "#D85A30")):
        m = ev["is_offday"].to_numpy() == val
        ax.scatter(ev["rain_mm"].to_numpy()[m], groups["precipitation"].to_numpy()[m], s=12, alpha=0.6,
                   label=name, color=color)
    ax.set_xscale("symlog", linthresh=1)
    ax.axhline(0, color="#888780", lw=0.8)
    ax.set_xlabel("관측 일강수량 (mm, 1mm 이상은 로그 눈금)")
    ax.set_ylabel(f"강수 묶음 SHAP\n{unit_label(target)}")
    ax.set_title(f"실제 강수량과 강수 묶음 SHAP ({PERIOD_NAMES[period]})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / f"04_rain_scatter_{period}{FIG_SUFFIX}.png")
    plt.close(fig)


def plot_ice(plt, centered, ice_table, info, period, groups_spec, fname, title):
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    if groups_spec is None:
        for _, row in centered.iterrows():
            ax.plot(centered.columns, row.to_numpy(), color="#B4B2A9", lw=0.5, alpha=0.15)
    x = ice_table["max_temp"]
    for col, label, color in groups_spec or []:
        ax.plot(x, ice_table[col], lw=2, label=label, color=color)
    ax.plot(x, ice_table["all_mean"], lw=2.5, color="#2C2C2A", label="전체 평균")
    ax.axvline(info["x0_reference"], color="#888780", ls="--", lw=0.8)
    ax.axvspan(HOT_LIMIT, x.max() + 0.5, color="#F7C1C1", alpha=0.4, label=f"{HOT_LIMIT:.0f}°C 초과(해석 제외)")
    ax.axhline(0, color="#888780", lw=0.6)
    ax.set_ylabel(f"예측 변화 (건)\n기준: 최고기온 {info['x0_reference']:.1f}°C")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax2.plot(x, ice_table["n_valid_days"], color="#5F5E5A")
    ax2.set_ylabel("평균에 든\n날짜 수")
    ax2.set_xlabel("옮긴 뒤 최고기온 (°C, 일교차 유지)")
    fig.tight_layout()
    fig.savefig(FIGURES / fname.replace(".png", f"{FIG_SUFFIX}.png"))
    plt.close(fig)


def plot_error_days(plt, show, sv_var, ev_dates, period, target):
    if show.empty:
        return
    n = len(show)
    fig, axes = plt.subplots(n, 1, figsize=(8, 2.6 * n), squeeze=False)
    pos = {d: i for i, d in enumerate(pd.to_datetime(ev_dates))}
    for ax, (_, r) in zip(axes[:, 0], show.iterrows()):
        s = sv_var.iloc[pos[r["date"]]]
        s = s.reindex(s.abs().sort_values(ascending=False).index)[:8][::-1]
        ax.barh([VAR_NAMES[c] for c in s.index], s.to_numpy(),
                color=["#1D9E75" if v > 0 else "#D85A30" for v in s.to_numpy()])
        ax.axvline(0, color="#888780", lw=0.6)
        kind = "과대예측" if r["direction"] == "over" else "과소예측"
        ax.set_title(f"{r['date']:%Y-%m-%d} ({kind}) 실제 {r['actual']:,.0f} / 예측 {r['pred_mean']:,.0f}, "
                     f"z = {r['robust_z']:.1f}", fontsize=9)
    axes[-1, 0].set_xlabel(f"{unit_label(target)}  (초록 = 예측을 높임, 주황 = 낮춤, 상위 8개 변수)")
    fig.tight_layout()
    fig.savefig(FIGURES / f"04_error_days_{period}{FIG_SUFFIX}.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 기간별 실행
# ---------------------------------------------------------------------------
def run_period(plt, df, stage, params, seeds, n_estimators, suffix):
    period = PERIOD_LABEL[stage]
    features = MODEL_FEATURES["P"]
    target = params["target"]
    train, ev = get_split(df, stage)
    ev = ev.reset_index(drop=True)
    X_tr, X_ev = make_X(train, features), make_X(ev, features)

    t0 = time.time()
    models, pred_by_seed = fit_models(params, seeds, X_tr, train["rentals"], X_ev, n_estimators)
    print(f"[{period}] 모델 {len(models)}개 학습 ({time.time() - t0:.0f}초)")

    # 02_main 과 같은 예측인지 확인
    pred_path = TABLES / f"02_predictions{suffix}.csv"
    preds02 = None
    if pred_path.exists():
        preds02 = pd.read_csv(pred_path, parse_dates=["date"])
        preds02 = preds02[preds02["period"] == period].reset_index(drop=True)
        cols = [f"pred_rf_weather_s{s}" for s in seeds]
        if all(c in preds02.columns for c in cols) and len(preds02) == len(ev):
            gap = np.abs(preds02[cols].to_numpy() - pred_by_seed).max()
            print(f"  02_main 기상 추가 RF 예측과의 최대 차이: {gap:.6f}  (0이면 같은 모델)")

    t0 = time.time()
    per_seed, sv_var, base = explain(models, X_ev, features)
    groups = shap_by_weather_group(sv_var)
    print(f"  SHAP 계산 ({time.time() - t0:.0f}초), 기준값 {base:.4f}")

    # 1) 날짜별 SHAP, 중요도
    out = pd.concat([ev[["date", "rentals", "is_offday", "season", "rain_mm", "max_temp", "min_temp"]],
                     sv_var.add_prefix("shap_"), groups.add_prefix("group_")], axis=1)
    out["base_value"] = base
    out["pred_mean"] = pred_by_seed.mean(axis=1)
    out.to_csv(TABLES / f"04_shap_values_{period}{suffix}.csv", index=False, encoding="utf-8-sig")

    imp = importance_rows(sv_var, target)
    imp.insert(0, "period", period)
    by_seed = []
    for (seed, _), sv in zip(models, per_seed):
        r = importance_rows(sv, target)
        r.insert(0, "seed", seed)
        r.insert(0, "period", period)
        by_seed.append(r)
    seg = segment_rows(sv_var, ev)
    seg.insert(0, "period", period)

    plot_importance(plt, imp, period, target)
    plot_segments(plt, seg, period, target)
    plot_rain(plt, ev, groups, period, target)

    # 2) 기온 반응 곡선
    def predict_fn(X):
        return np.mean([to_count(m.predict(X), target) for _, m in models], axis=0)

    t0 = time.time()
    _, centered, mean_curve, info = temperature_ice(predict_fn, X_ev, X_tr, step=0.5)
    ice = mean_curve.rename(columns={"centered_mean": "all_mean"})
    for val, name in OFFDAY_NAMES.items():
        ice[f"offday_{val}_mean"] = centered[ev["is_offday"].to_numpy() == val].mean(axis=0).to_numpy()
    for val in SEASON_NAMES:
        m = ev["season"].to_numpy() == val
        if m.any():
            ice[f"season_{val}_mean"] = centered[m].mean(axis=0).to_numpy()
    ice["x0_reference"] = info["x0_reference"]
    ice.to_csv(TABLES / f"04_ice_temp_{period}{suffix}.csv", index=False, encoding="utf-8-sig")
    print(f"  기온 반응 곡선 ({time.time() - t0:.0f}초), 기준점 {info['x0_reference']:.1f}°C, "
          f"학습 최고기온 범위 {info['train_max_temp_range'][0]:.1f}~{info['train_max_temp_range'][1]:.1f}°C")
    plot_ice(plt, centered, ice, info, period,
             [("offday_0_mean", "평일 평균", "#378ADD"), ("offday_1_mean", "쉬는 날 평균", "#D85A30")],
             f"04_ice_temp_{period}.png",
             f"최고·최저기온을 함께 옮긴 예측 반응 ({PERIOD_NAMES[period]}, 평일·쉬는 날 평균)")
    season_spec = [(f"season_{v}_mean", f"{SEASON_NAMES[v]} 평균", c) for v, c in
                   zip(SEASON_NAMES, ["#639922", "#D85A30", "#BA7517", "#378ADD"]) if f"season_{v}_mean" in ice]
    plot_ice(plt, centered, ice, info, period, season_spec, f"04_ice_temp_season_{period}.png",
             f"계절별 기온 반응 평균 곡선 ({PERIOD_NAMES[period]})")
    plot_ice(plt, centered, ice, info, period, None, f"04_ice_temp_days_{period}.png",
             f"날짜별 기온 반응 곡선과 전체 평균 ({PERIOD_NAMES[period]})")

    # 3) 오차 큰 날
    large = pd.DataFrame()
    if preds02 is not None:
        cols = [f"pred_rf_weather_s{s}" for s in seeds]
        flagged, show, summary = select_large_error_days(
            preds02["date"], preds02["rentals"], preds02[cols].to_numpy(),
            extra=preds02[["is_offday", "season"]])
        pos = {d: i for i, d in enumerate(ev["date"])}
        top = []
        for d in flagged["date"]:
            s = sv_var.iloc[pos[d]]
            s = s.reindex(s.abs().sort_values(ascending=False).index)[:3]
            top.append(", ".join(f"{VAR_NAMES[k]} {v:+.3f}" for k, v in s.items()))
        large = flagged.assign(top3_shap=top, shown_in_figure=flagged["date"].isin(show["date"]))
        large.insert(0, "period", period)
        large.to_csv(TABLES / f"04_large_error_days_{period}{suffix}.csv", index=False, encoding="utf-8-sig")
        plot_error_days(plt, show, sv_var, ev["date"], period, target)
        print(f"  오차 큰 날: {summary['n_flagged']}일 (과대 {summary['n_flagged_over']}, "
              f"과소 {summary['n_flagged_under']}), 치우침(중앙값) {summary['median_error(bias)']:,.0f}건")
    else:
        print(f"  {pred_path.name} 이 없어 오차 큰 날 분석을 건너뜀 (먼저 02_main.py 실행)")

    return imp, pd.concat(by_seed, ignore_index=True), seg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="dataset.csv 경로 (기본: data/processed/dataset.csv)")
    ap.add_argument("--quick", action="store_true", help="시드 2개, 트리 50개로 코드만 점검")
    args = ap.parse_args()

    seeds = SEEDS[:2] if args.quick else SEEDS
    n_estimators = 50 if args.quick else None
    suffix = "_quick" if args.quick else ""
    global FIG_SUFFIX
    FIG_SUFFIX = suffix

    ensure_dirs()
    plt = setup_matplotlib()
    df = load_dataset(args.data)
    params = load_selected()["P"]
    print(f"shap {shap.__version__}, P 설정: n_estimators={n_estimators or params['n_estimators']}, "
          f"max_depth={params['max_depth']}, min_samples_leaf={params['min_samples_leaf']}, "
          f"max_features={params['max_features']:.3f}, target={params['target']}")

    imps, seeds_imp, segs = [], [], []
    for stage in TEST_STAGES:
        imp, by_seed, seg = run_period(plt, df, stage, params, seeds, n_estimators, suffix)
        imps.append(imp)
        seeds_imp.append(by_seed)
        segs.append(seg)
    imp = pd.concat(imps, ignore_index=True)
    imp.to_csv(TABLES / f"04_importance{suffix}.csv", index=False, encoding="utf-8-sig")
    pd.concat(seeds_imp, ignore_index=True).to_csv(TABLES / f"04_importance_by_seed{suffix}.csv",
                                                    index=False, encoding="utf-8-sig")
    pd.concat(segs, ignore_index=True).to_csv(TABLES / f"04_by_segment{suffix}.csv",
                                              index=False, encoding="utf-8-sig")

    show_cols = ["period", "rank", "name_ko", "mean_abs", "mean_signed", "mean_signed_pct"]
    print("\n변수 중요도 상위 5개 (평균 |SHAP|, 평균 SHAP, 평균 SHAP의 % 환산)")
    print(imp[(imp["level"] == "variable") & (imp["rank"] <= 5)][show_cols].round(4).to_string(index=False))
    print("\n날씨 묶음 중요도")
    print(imp[imp["level"] == "group"][show_cols].round(4).to_string(index=False))
    print("\n연도(year)의 평균 SHAP — 과대예측과의 관계 확인용")
    print(imp[imp["name"] == "year"][["period", "mean_signed", "mean_signed_pct"]].round(4).to_string(index=False))
    print(f"\n[저장] {TABLES} (04_*{suffix}.csv), {FIGURES} (04_*.png)")


if __name__ == "__main__":
    main()
