"""问 1：附件 1 点预测曲线下的确定性日前购电计划。"""

from __future__ import annotations

import numpy as np

from .data import SOC_INIT, load_attachment1
from .optimizer import solve_deterministic
from .results import print_summary, summarize_q1, write_result1
from .value_dp import periodic_value


def main(data_path: str = "data/附件1.xlsx", out_path: str = "results/result1.xlsx") -> None:
    att = load_attachment1(data_path)
    res = solve_deterministic(att.price, att.load_kwh, att.pv_kwh, SOC_INIT, periodic=True)
    # Independent dynamic-programming cross-check (continuous action, fine
    # SOC mesh) used for the report's LP/DP validation.
    dp = periodic_value(att.price, att.load_kwh, att.pv_kwh, SOC_INIT,
                        grid=np.linspace(1200.0, 10800.0, 1921))
    path = write_result1(res, SOC_INIT, out_path)
    print_summary(summarize_q1(res, att.price, SOC_INIT))
    print(f"\n周期约束对偶价格 {res.duals['periodic']:.4f} 元/kWh")
    print(f"LP/DP 互证值：LP={res.purchase_cost:.4f}，DP≈{dp(SOC_INIT):.4f}，差={dp(SOC_INIT)-res.purchase_cost:.4f} 元")
    print(f"已写出 {path}")


if __name__ == "__main__":
    main()
