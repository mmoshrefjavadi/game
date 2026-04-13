from __future__ import annotations

import json
import pandas as pd
import streamlit as st

from capacity_game_v3 import SimulationConfig, CapacityStrategySimulationV3, CapacityOption

st.set_page_config(page_title="Capacity Strategy Game V5", layout="wide")

# -----------------------------
# Defaults and helpers
# -----------------------------

DEFAULTS = {
    "seed": 42,
    "initial_demand": 100.0,
    "initial_capacity": 108.0,
    "baseline_growth": 0.065,
    "demand_shock_std": 0.04,
    "forecast_noise_std": 0.03,
    "forecast_adjustment_pct": 0.10,
    "unit_revenue": 11.0,
    "lost_sales_penalty": 7.0,
    "overcapacity_penalty": 2.0,
    "annual_budget": 65.0,
    "small_size_factor": 0.10,
    "medium_size_factor": 0.20,
    "large_size_factor": 0.32,
    "small_fixed_cost": 4.5,
    "medium_fixed_cost": 7.0,
    "large_fixed_cost": 10.0,
    "small_var_cost": 2.0,
    "medium_var_cost": 1.6,
    "large_var_cost": 1.35,
}


def build_config_from_session() -> tuple[SimulationConfig, int]:
    seed = int(st.session_state.get("seed", DEFAULTS["seed"]))
    options = {
        "small": CapacityOption("small", lead_time=1,
                                 size_factor=float(st.session_state.get("small_size_factor", DEFAULTS["small_size_factor"])),
                                 fixed_cost=float(st.session_state.get("small_fixed_cost", DEFAULTS["small_fixed_cost"])),
                                 variable_cost=float(st.session_state.get("small_var_cost", DEFAULTS["small_var_cost"]))),
        "medium": CapacityOption("medium", lead_time=1,
                                  size_factor=float(st.session_state.get("medium_size_factor", DEFAULTS["medium_size_factor"])),
                                  fixed_cost=float(st.session_state.get("medium_fixed_cost", DEFAULTS["medium_fixed_cost"])),
                                  variable_cost=float(st.session_state.get("medium_var_cost", DEFAULTS["medium_var_cost"]))),
        "large": CapacityOption("large", lead_time=2,
                                 size_factor=float(st.session_state.get("large_size_factor", DEFAULTS["large_size_factor"])),
                                 fixed_cost=float(st.session_state.get("large_fixed_cost", DEFAULTS["large_fixed_cost"])),
                                 variable_cost=float(st.session_state.get("large_var_cost", DEFAULTS["large_var_cost"]))),
    }
    config = SimulationConfig(
        initial_demand=float(st.session_state.get("initial_demand", DEFAULTS["initial_demand"])),
        initial_capacity=float(st.session_state.get("initial_capacity", DEFAULTS["initial_capacity"])),
        baseline_growth=float(st.session_state.get("baseline_growth", DEFAULTS["baseline_growth"])),
        demand_shock_std=float(st.session_state.get("demand_shock_std", DEFAULTS["demand_shock_std"])),
        forecast_noise_std=float(st.session_state.get("forecast_noise_std", DEFAULTS["forecast_noise_std"])),
        forecast_adjustment_pct=float(st.session_state.get("forecast_adjustment_pct", DEFAULTS["forecast_adjustment_pct"])),
        unit_revenue=float(st.session_state.get("unit_revenue", DEFAULTS["unit_revenue"])),
        lost_sales_penalty=float(st.session_state.get("lost_sales_penalty", DEFAULTS["lost_sales_penalty"])),
        overcapacity_penalty=float(st.session_state.get("overcapacity_penalty", DEFAULTS["overcapacity_penalty"])),
        annual_budget=float(st.session_state.get("annual_budget", DEFAULTS["annual_budget"])),
        options=options,
    )
    return config, seed


def init_defaults() -> None:
    for key, value in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def init_game(force: bool = False) -> None:
    config, seed = build_config_from_session()
    if force or "sim" not in st.session_state:
        st.session_state.sim = CapacityStrategySimulationV3(config=config, seed=seed)
        st.session_state.active_config_json = json.dumps({
            "seed": seed,
            **{k: getattr(config, k) for k in [
                "initial_demand", "initial_capacity", "baseline_growth", "demand_shock_std",
                "forecast_noise_std", "forecast_adjustment_pct", "unit_revenue",
                "lost_sales_penalty", "overcapacity_penalty", "annual_budget"
            ]},
            "options": {name: vars(opt) for name, opt in config.options.items()}
        }, sort_keys=True)


def current_config_changed() -> bool:
    config, seed = build_config_from_session()
    current_json = json.dumps({
        "seed": seed,
        **{k: getattr(config, k) for k in [
            "initial_demand", "initial_capacity", "baseline_growth", "demand_shock_std",
            "forecast_noise_std", "forecast_adjustment_pct", "unit_revenue",
            "lost_sales_penalty", "overcapacity_penalty", "annual_budget"
        ]},
        "options": {name: vars(opt) for name, opt in config.options.items()}
    }, sort_keys=True)
    return current_json != st.session_state.get("active_config_json", "")


def format_num(x: float) -> str:
    return f"{x:,.1f}"


def outcome_badge(label: str, tone: str = "neutral") -> str:
    colors = {
        "good": ("#dcfce7", "#166534"),
        "warn": ("#fef3c7", "#92400e"),
        "bad": ("#fee2e2", "#991b1b"),
        "neutral": ("#e5e7eb", "#374151"),
        "info": ("#dbeafe", "#1d4ed8"),
    }
    bg, fg = colors[tone]
    return f"<span style='background:{bg};color:{fg};padding:0.28rem 0.58rem;border-radius:999px;font-size:0.85rem;font-weight:650;'>{label}</span>"


def render_scenario_cards(sim: CapacityStrategySimulationV3, year: int):
    baseline = sim.get_baseline_forecast(year)
    current_capacity = sim.current_capacity
    scenarios = [
        ("Conservative", "conservative", "Lower demand view. Useful when you worry forecasts are overstated."),
        ("Base", "base", "Use the baseline signal. Best when the market looks stable."),
        ("Aggressive", "aggressive", "Higher demand view. Useful when strong growth feels credible."),
    ]
    cols = st.columns(3)
    for col, (title, mode, desc) in zip(cols, scenarios):
        adj = sim.adjust_forecast(baseline, mode)
        ref = sim.get_reference_demand(adj, year)
        gap = ref - current_capacity
        if gap > current_capacity * 0.08:
            tone, risk = "bad", "Undercapacity risk"
        elif gap < -current_capacity * 0.08:
            tone, risk = "warn", "Overcapacity risk"
        else:
            tone, risk = "good", "Roughly aligned"
        with col:
            st.markdown(
                f"""
                <div style='border:1px solid #e5e7eb;border-radius:18px;padding:1rem;background:white;height:100%;'>
                    <div style='font-size:1.05rem;font-weight:700;margin-bottom:0.2rem;'>{title}</div>
                    <div style='color:#4b5563;font-size:0.9rem;min-height:45px'>{desc}</div>
                    <div style='margin-top:0.8rem;'>
                        <div style='font-size:0.82rem;color:#6b7280;'>Next-year reference demand</div>
                        <div style='font-size:1.45rem;font-weight:750;'>{ref:,.1f}</div>
                    </div>
                    <div style='margin-top:0.45rem;'>
                        <div style='font-size:0.82rem;color:#6b7280;'>Gap vs current capacity</div>
                        <div style='font-size:1.05rem;font-weight:650;'>{gap:+,.1f}</div>
                    </div>
                    <div style='margin-top:0.75rem;'>{outcome_badge(risk, tone)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def recommendation_text(sim: CapacityStrategySimulationV3, year: int, adjustment_mode: str, action: str) -> tuple[str, str]:
    baseline = sim.get_baseline_forecast(year)
    adjusted = sim.adjust_forecast(baseline, adjustment_mode)
    ref = sim.get_reference_demand(adjusted, year)
    details = sim.compute_project_details(action, ref)
    current_capacity = sim.current_capacity
    gap = ref - current_capacity

    if action == "none":
        msg = "You are choosing to wait this year."
    else:
        msg = f"You are proposing a {action} expansion that adds about {details['capacity_added']:.1f} units."

    if details["investment_cost"] > sim.config.annual_budget:
        tone = "Budget warning: this choice will be rejected under the current annual budget."
    elif gap > current_capacity * 0.12 and action in ["none", "small"]:
        tone = "Forecasted demand is meaningfully above current capacity, so this may leave you exposed to lost sales."
    elif gap < -current_capacity * 0.10 and action in ["medium", "large"]:
        tone = "Forecasted demand does not justify a large expansion right now, so overcapacity risk may rise."
    elif action == "large":
        tone = "Large expansion is a strategic bet: cheaper per unit, but slow and risky if growth fades."
    elif action == "small":
        tone = "Small expansion keeps flexibility high, but may be too cautious if strong growth persists."
    else:
        tone = "This looks like a balanced move given the current forecast and budget."
    return msg, tone


def render_dashboard(sim: CapacityStrategySimulationV3):
    results_df = sim.get_results_df()
    summary = sim.get_summary_metrics()
    st.subheader("Performance Dashboard")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cumulative Profit", format_num(summary["cumulative_profit"]))
    c2.metric("Avg Service Level", f"{summary['average_service_level']:.1%}")
    c3.metric("Avg Utilization", f"{summary['average_utilization']:.1%}")
    c4.metric("Total Lost Sales", format_num(summary["total_lost_sales"]))

    if results_df.empty:
        st.info("No years completed yet. Make your first decision to start the game.")
        return

    chart_df = sim.get_chart_data().copy()
    tl, tr = st.columns(2)
    with tl:
        st.markdown("#### Capacity vs Demand")
        st.line_chart(chart_df.set_index("year")[["realized_demand", "capacity_start_of_year", "sales"]])
    with tr:
        st.markdown("#### Profit and Investment")
        st.bar_chart(chart_df.set_index("year")[["annual_profit", "investment_cost"]])

    bl, br = st.columns(2)
    with bl:
        st.markdown("#### Service and Utilization")
        st.line_chart(chart_df.set_index("year")[["service_level", "utilization"]])
    with br:
        st.markdown("#### Lost Sales and Unused Capacity")
        st.bar_chart(chart_df.set_index("year")[["lost_sales", "unused_capacity"]])

    display_cols = [
        "year", "adjustment_mode", "requested_action", "action", "realized_demand",
        "capacity_start_of_year", "sales", "lost_sales", "unused_capacity",
        "investment_cost", "annual_profit", "service_level", "utilization"
    ]
    st.markdown("#### Year-by-Year Table")
    st.dataframe(results_df[display_cols].round(2), use_container_width=True, hide_index=True)


def render_final_results(sim: CapacityStrategySimulationV3):
    summary = sim.get_summary_metrics()
    results_df = sim.get_results_df()
    bench_df = sim.compare_benchmarks()
    player_profit = summary["cumulative_profit"]
    best_row = bench_df.sort_values("cumulative_profit", ascending=False).iloc[0]

    st.subheader("Final Results")
    a, b, c, d = st.columns(4)
    a.metric("Final Profit", format_num(summary["cumulative_profit"]))
    b.metric("Avg Service Level", f"{summary['average_service_level']:.1%}")
    c.metric("Avg Utilization", f"{summary['average_utilization']:.1%}")
    d.metric("Total Investment", format_num(summary["total_investment_cost"]))

    if player_profit >= best_row["cumulative_profit"]:
        score_msg = outcome_badge("Top benchmark performance", "good")
    elif summary["average_service_level"] < 0.93:
        score_msg = outcome_badge("Service suffered from tight capacity", "bad")
    elif summary["average_utilization"] < 0.78:
        score_msg = outcome_badge("Expansion may have been too aggressive", "warn")
    else:
        score_msg = outcome_badge("Balanced but improvable", "info")

    st.markdown(f"<div style='margin:0.25rem 0 1rem 0'>{score_msg}</div>", unsafe_allow_html=True)

    compare_df = pd.concat([
        bench_df,
        pd.DataFrame([{"policy_name": "Current Player", **summary}])
    ], ignore_index=True)
    st.markdown("#### Benchmark Comparison")
    st.bar_chart(compare_df.set_index("policy_name")[["cumulative_profit"]])
    st.dataframe(compare_df.round(2), use_container_width=True, hide_index=True)

    csv_data = results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download results CSV",
        data=csv_data,
        file_name="capacity_game_results.csv",
        mime="text/csv",
        use_container_width=True,
    )


def render_instructor_review(sim: CapacityStrategySimulationV3):
    review = sim.get_instructor_review()
    summary = review["summary"]
    rank = review["player_rank_by_profit"]
    bench_df = review["benchmarks"].copy()

    st.subheader("Instructor Review")
    a, b, c, d = st.columns(4)
    a.metric("Player Profit", format_num(summary["cumulative_profit"]))
    b.metric("Service Level", f"{summary['average_service_level']:.1%}")
    c.metric("Utilization", f"{summary['average_utilization']:.1%}")
    d.metric("Profit Rank", "-" if rank is None else f"#{rank}")

    if not bench_df.empty:
        player_row = pd.DataFrame([{"policy_name": "Current Player", **summary}])
        compare_df = pd.concat([bench_df, player_row], ignore_index=True)
        st.markdown("#### Benchmark Comparison")
        st.bar_chart(compare_df.set_index("policy_name")[["cumulative_profit"]])
        st.dataframe(compare_df.round(2), use_container_width=True, hide_index=True)

    if not review["results_table"].empty:
        df = review["results_table"]
        notes = []
        if df["service_level"].mean() < 0.95:
            notes.append("Average service level is below 95%, so the student likely under-expanded at key moments.")
        if df["utilization"].mean() < 0.80:
            notes.append("Average utilization is below 80%, suggesting expansions may have been too early or too large.")
        if df["action_rejected_for_budget"].sum() > 0:
            notes.append("At least one decision was rejected for budget reasons, which shows the student did not fully adapt to financing constraints.")
        if not notes:
            notes.append("The pattern looks balanced. Review whether expansions were proactive rather than reactive.")
        st.markdown("#### Instructor Notes")
        for note in notes:
            st.write(f"- {note}")
        st.markdown("#### Results Table")
        st.dataframe(df.round(2), use_container_width=True, hide_index=True)


def render_ai_panel(sim: CapacityStrategySimulationV3):
    st.subheader("AI Coach")
    st.info("The app is ready for a real LLM later. For now, use the coach note and the prompt templates below.")
    st.markdown("#### Current Coach Note")
    st.write(sim.get_rule_based_coach_message())
    st.markdown("#### System Prompt")
    st.code(sim.get_ai_system_prompt())
    st.markdown("#### User Prompt")
    st.code(sim.get_ai_user_prompt())
    st.markdown("#### Context Payload Preview")
    st.json(sim.build_ai_context(include_full_history=False))


# -----------------------------
# Initialize
# -----------------------------

init_defaults()
init_game(force=("sim" not in st.session_state))
sim = st.session_state.sim

st.markdown(
    """
    <style>
    .stMetric {background: #fafafa; border: 1px solid #e5e7eb; padding: 0.45rem 0.65rem; border-radius: 14px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Sidebar controls
# -----------------------------

with st.sidebar:
    st.header("Game Setup")
    st.number_input("Random seed", min_value=0, step=1, key="seed")
    st.number_input("Initial demand", min_value=10.0, step=5.0, key="initial_demand")
    st.number_input("Initial capacity", min_value=10.0, step=5.0, key="initial_capacity")
    st.slider("Baseline growth", min_value=0.00, max_value=0.20, step=0.005, key="baseline_growth")
    st.slider("Demand volatility", min_value=0.00, max_value=0.15, step=0.005, key="demand_shock_std")
    st.slider("Forecast noise", min_value=0.00, max_value=0.12, step=0.005, key="forecast_noise_std")
    st.slider("Forecast adjustment %", min_value=0.00, max_value=0.25, step=0.01, key="forecast_adjustment_pct")
    st.number_input("Annual budget", min_value=10.0, step=5.0, key="annual_budget")

    with st.expander("Economics", expanded=False):
        st.number_input("Revenue per unit", min_value=1.0, step=0.5, key="unit_revenue")
        st.number_input("Lost sales penalty", min_value=0.0, step=0.5, key="lost_sales_penalty")
        st.number_input("Overcapacity penalty", min_value=0.0, step=0.5, key="overcapacity_penalty")

    with st.expander("Expansion options", expanded=False):
        st.slider("Small size factor", min_value=0.05, max_value=0.30, step=0.01, key="small_size_factor")
        st.slider("Medium size factor", min_value=0.10, max_value=0.45, step=0.01, key="medium_size_factor")
        st.slider("Large size factor", min_value=0.15, max_value=0.60, step=0.01, key="large_size_factor")
        st.number_input("Small fixed cost", min_value=0.0, step=0.5, key="small_fixed_cost")
        st.number_input("Medium fixed cost", min_value=0.0, step=0.5, key="medium_fixed_cost")
        st.number_input("Large fixed cost", min_value=0.0, step=0.5, key="large_fixed_cost")
        st.number_input("Small variable cost", min_value=0.0, step=0.05, key="small_var_cost")
        st.number_input("Medium variable cost", min_value=0.0, step=0.05, key="medium_var_cost")
        st.number_input("Large variable cost", min_value=0.0, step=0.05, key="large_var_cost")

    if current_config_changed():
        st.warning("Scenario settings changed. Apply and restart to use them.")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Apply & Restart", use_container_width=True):
            init_game(force=True)
            st.rerun()
    with col_b:
        if st.button("Reset Defaults", use_container_width=True):
            for key, value in DEFAULTS.items():
                st.session_state[key] = value
            init_game(force=True)
            st.rerun()

    st.markdown("---")
    st.markdown("### Current Option Table")
    option_rows = []
    for key, opt in sim.config.options.items():
        option_rows.append({
            "Option": key.title(),
            "Lead Time": opt.lead_time,
            "Size Factor": opt.size_factor,
            "Fixed Cost": opt.fixed_cost,
            "Variable Cost": opt.variable_cost,
        })
    st.dataframe(pd.DataFrame(option_rows), use_container_width=True, hide_index=True)

# -----------------------------
# Header
# -----------------------------

st.title("Capacity Strategy Simulation Game")
st.caption("Version 5: student inputs, polished gameplay, final score screen, instructor review, and AI coach hooks.")
summary = sim.get_summary_metrics()
header_cols = st.columns(5)
header_cols[0].metric("Current Year", f"{min(sim.current_year, sim.config.horizon)} / {sim.config.horizon}")
header_cols[1].metric("Current Capacity", format_num(sim.current_capacity))
header_cols[2].metric("Cumulative Profit", format_num(summary["cumulative_profit"]))
header_cols[3].metric("Avg Service Level", f"{summary['average_service_level']:.1%}")
header_cols[4].metric("Avg Utilization", f"{summary['average_utilization']:.1%}")

student_tab, instructor_tab, ai_tab = st.tabs(["Student Play", "Instructor Review", "AI Coach"])

# -----------------------------
# Student tab
# -----------------------------

with student_tab:
    if sim.current_year <= sim.config.horizon:
        year = sim.current_year
        state = sim.get_available_state_for_player(year)
        baseline = sim.get_baseline_forecast(year)

        st.subheader(f"Decision Round: Year {year}")
        left, right = st.columns([1.45, 1])
        with left:
            st.markdown("#### Scenario Outlook")
            render_scenario_cards(sim, year)
        with right:
            pipeline_total = sum(state["pipeline"].values()) if state["pipeline"] else 0.0
            st.markdown(
                f"""
                <div style='border:1px solid #e5e7eb;border-radius:18px;padding:1rem;background:white;'>
                    <div style='font-size:0.85rem;color:#6b7280;'>Current capacity</div>
                    <div style='font-size:1.65rem;font-weight:750;margin-bottom:0.55rem;'>{state['current_capacity']:,.1f}</div>
                    <div style='font-size:0.85rem;color:#6b7280;'>Pipeline arriving later</div>
                    <div style='font-size:1.2rem;font-weight:650;margin-bottom:0.55rem;'>{pipeline_total:,.1f}</div>
                    <div style='font-size:0.85rem;color:#6b7280;'>Annual budget</div>
                    <div style='font-size:1.2rem;font-weight:650;'>{state['annual_budget']:,.1f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if state["pipeline"]:
                st.markdown("**Pipeline by arrival year**")
                st.json(state["pipeline"])

        with st.expander("Demand outlook and history", expanded=True):
            mid_left, mid_right = st.columns(2)
            with mid_left:
                forecast_df = pd.DataFrame({"year": list(baseline.keys()), "baseline_forecast": list(baseline.values())})
                st.line_chart(forecast_df.set_index("year"))
                st.dataframe(forecast_df.round(2), use_container_width=True, hide_index=True)
            with mid_right:
                history = state["demand_history"]
                if history:
                    history_df = pd.DataFrame({"year": list(history.keys()), "realized_demand": list(history.values())})
                    st.line_chart(history_df.set_index("year"))
                    st.dataframe(history_df.round(2), use_container_width=True, hide_index=True)
                else:
                    st.info("No realized demand history yet.")

        st.markdown("#### Make Your Decision")
        c1, c2 = st.columns(2)
        with c1:
            adjustment_mode = st.radio("Forecast adjustment", options=state["valid_adjustments"], horizontal=True, index=1)
        with c2:
            action = st.radio("Capacity action", options=state["valid_actions"], horizontal=True, index=0)

        adjusted = sim.adjust_forecast(baseline, adjustment_mode)
        ref = sim.get_reference_demand(adjusted, year)
        details = sim.compute_project_details(action, ref)
        intro, interpretation = recommendation_text(sim, year, adjustment_mode, action)

        pc1, pc2, pc3, pc4 = st.columns(4)
        pc1.metric("Reference Demand", format_num(ref))
        pc2.metric("Capacity Added", format_num(details["capacity_added"]))
        pc3.metric("Investment Cost", format_num(details["investment_cost"]))
        pc4.metric("Lead Time", f"{details['lead_time']} year(s)")

        st.markdown(
            f"""
            <div style='border:1px solid #e5e7eb;border-radius:18px;padding:1rem;background:#f9fafb;'>
                <div style='font-weight:700;margin-bottom:0.35rem;'>Decision Preview</div>
                <div style='margin-bottom:0.35rem;'>{intro}</div>
                <div style='color:#4b5563;'>{interpretation}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("Quick what-if sandbox", expanded=False):
            sandbox_demand = st.number_input("Try a custom next-year demand level", min_value=0.0, value=float(ref), step=5.0)
            sandbox_gap = sandbox_demand - sim.current_capacity
            st.write(f"Gap vs current capacity: {sandbox_gap:+.1f}")
            what_if_rows = []
            for candidate in state["valid_actions"]:
                d = sim.compute_project_details(candidate, sandbox_demand)
                projected_capacity = sim.current_capacity + d["capacity_added"]
                what_if_rows.append({
                    "action": candidate,
                    "capacity_added": round(d["capacity_added"], 2),
                    "investment_cost": round(d["investment_cost"], 2),
                    "projected_capacity_after_arrival": round(projected_capacity, 2),
                    "within_budget": d["investment_cost"] <= sim.config.annual_budget,
                })
            st.dataframe(pd.DataFrame(what_if_rows), use_container_width=True, hide_index=True)

        if details["investment_cost"] > sim.config.annual_budget:
            st.error("This action exceeds the annual budget and will be replaced with 'none' if submitted.")

        if st.button("Submit Year Decision", type="primary", use_container_width=True):
            result = sim.play_one_year(adjustment_mode=adjustment_mode, action=action)
            if result["action_rejected_for_budget"]:
                st.warning("Your requested action exceeded budget and was replaced with 'none'.")
            else:
                st.success(f"Year {result['year']} completed successfully.")
            st.rerun()
    else:
        render_final_results(sim)

    st.markdown("---")
    render_dashboard(sim)

with instructor_tab:
    render_instructor_review(sim)

with ai_tab:
    render_ai_panel(sim)
