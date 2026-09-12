"""问 1：附件 1 点预测曲线下的确定性日前购电计划。"""

from __future__ import annotations

from .data import SOC_INIT, load_attachment1
from .optimizer import solve_deterministic
from .results import print_summary, summarize_q1, write_result1
from .value_dp import periodic_value, recover_periodic_actions


def main(data_path: str = "data/附件1.xlsx", out_path: str = "results/result1.xlsx") -> None:
    att = load_attachment1(data_path)
    res = solve_deterministic(att.price, att.load_kwh, att.pv_kwh, SOC_INIT, periodic=True)
    # Independent exact continuous-DP check.  The terminal is the literal
    # one-point constraint S_144=6000, not a SOC mesh or a penalty.
    dp = periodic_value(att.price, att.load_kwh, att.pv_kwh, SOC_INIT)
    recovered = recover_periodic_actions(att.price, att.load_kwh, att.pv_kwh, SOC_INIT)
    if abs(recovered.objective - res.purchase_cost) > 1e-6:
        raise RuntimeError("Q1 LP and independently recovered DP actions disagree")
    path = write_result1(recovered, SOC_INIT, out_path)
    print_summary(summarize_q1(recovered, att.price, SOC_INIT))
    print(f"\n周期约束对偶价格 {res.duals['periodic']:.4f} 元/kWh")
    print(f"LP/DP 互证值：LP={res.purchase_cost:.4f}，DP={dp(SOC_INIT):.4f}，差={dp(SOC_INIT)-res.purchase_cost:.4f} 元")
    print(f"已写出 {path}")


if __name__ == "__main__":
    main()
