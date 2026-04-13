from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
import copy
import numpy as np
import pandas as pd


@dataclass
class CapacityOption:
    name: str
    lead_time: int
    size_factor: float
    fixed_cost: float
    variable_cost: float


@dataclass
class SimulationConfig:
    horizon: int = 10
    initial_demand: float = 100.0
    initial_capacity: float = 108.0

    baseline_growth: float = 0.065
    demand_shock_std: float = 0.04
    forecast_noise_std: float = 0.03

    forecast_adjustment_pct: float = 0.10

    unit_revenue: float = 11.0
    lost_sales_penalty: float = 7.0
    overcapacity_penalty: float = 2.0

    annual_budget: float = 65.0

    options: Dict[str, CapacityOption] = field(default_factory=lambda: {
        "small": CapacityOption("small", lead_time=1, size_factor=0.10, fixed_cost=4.5, variable_cost=2.0),
        "medium": CapacityOption("medium", lead_time=1, size_factor=0.20, fixed_cost=7.0, variable_cost=1.6),
        "large": CapacityOption("large", lead_time=2, size_factor=0.32, fixed_cost=10.0, variable_cost=1.35),
    })


class CapacityStrategySimulationV3:
    """
    Version 3 simulation engine.

    Improvements over V2:
    - more balanced base parameters so medium and large decisions are more often feasible
    - AI coach hooks: structured context builder + rule-based coach text
    - cleaner student/instructor review support methods
    - chart-ready time series outputs
    """

    VALID_ACTIONS = ["none", "small", "medium", "large"]
    VALID_ADJUSTMENTS = ["conservative", "base", "aggressive"]

    def __init__(self, config: SimulationConfig, seed: Optional[int] = None):
        self.config = config
        self.seed = seed
        self.reset(seed=seed)

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.seed = seed
        self.rng = np.random.default_rng(self.seed)

        self.current_year = 1
        self.current_capacity = float(self.config.initial_capacity)
        self.pipeline: Dict[int, List[float]] = {}
        self.true_demand_history: Dict[int, float] = {}
        self.results: List[Dict[str, Any]] = []

        self.true_demand_path = self._generate_true_demand_path()
        self.forecast_cache = self._precompute_baseline_forecasts()

    # -------------------------
    # Hidden demand and forecast generation
    # -------------------------

    def _generate_true_demand_path(self) -> Dict[int, float]:
        demand = {1: float(self.config.initial_demand)}
        for year in range(2, self.config.horizon + 1):
            shock = self.rng.normal(0, self.config.demand_shock_std)
            growth = max(self.config.baseline_growth + shock, -0.90)
            demand[year] = max(0.0, demand[year - 1] * (1 + growth))
        return demand

    def _generate_baseline_forecast_for_year(self, year: int) -> Dict[int, float]:
        forecast = {}
        for future_year in range(year, self.config.horizon + 1):
            true_value = self.true_demand_path[future_year]
            horizon_gap = future_year - year
            scaled_std = self.config.forecast_noise_std * (1 + 0.20 * horizon_gap)
            noise = self.rng.normal(0, scaled_std)
            forecast[future_year] = max(0.0, true_value * (1 + noise))
        return forecast

    def _precompute_baseline_forecasts(self) -> Dict[int, Dict[int, float]]:
        return {
            year: self._generate_baseline_forecast_for_year(year)
            for year in range(1, self.config.horizon + 1)
        }

    # -------------------------
    # Public player-facing state
    # -------------------------

    def get_baseline_forecast(self, year: int) -> Dict[int, float]:
        self._validate_year(year)
        return copy.deepcopy(self.forecast_cache[year])

    def adjust_forecast(self, baseline_forecast: Dict[int, float], adjustment_mode: str) -> Dict[int, float]:
        self._validate_adjustment(adjustment_mode)
        delta = self.config.forecast_adjustment_pct
        multipliers = {
            "conservative": 1 - delta,
            "base": 1.0,
            "aggressive": 1 + delta,
        }
        factor = multipliers[adjustment_mode]
        return {year: value * factor for year, value in baseline_forecast.items()}

    def get_reference_demand(self, adjusted_forecast: Dict[int, float], year: int) -> float:
        next_year = year + 1
        if next_year in adjusted_forecast:
            return adjusted_forecast[next_year]
        return adjusted_forecast[year]

    def get_available_state_for_player(self, year: Optional[int] = None) -> Dict[str, Any]:
        if year is None:
            year = self.current_year
        self._validate_year(year)
        if year < self.current_year:
            raise ValueError("Cannot request pre-decision state for a year already simulated.")

        baseline_forecast = self.get_baseline_forecast(year)
        return {
            "year": year,
            "current_capacity": round(self.current_capacity, 4),
            "annual_budget": self.config.annual_budget,
            "pipeline": {k: round(sum(v), 4) for k, v in sorted(self.pipeline.items())},
            "baseline_forecast": {k: round(v, 4) for k, v in baseline_forecast.items()},
            "demand_history": {k: round(v, 4) for k, v in self.true_demand_history.items()},
            "valid_actions": self.VALID_ACTIONS,
            "valid_adjustments": self.VALID_ADJUSTMENTS,
            "config_snapshot": {
                "forecast_adjustment_pct": self.config.forecast_adjustment_pct,
                "unit_revenue": self.config.unit_revenue,
                "lost_sales_penalty": self.config.lost_sales_penalty,
                "overcapacity_penalty": self.config.overcapacity_penalty,
                "annual_budget": self.config.annual_budget,
            },
        }

    # -------------------------
    # Pipeline and cost helpers
    # -------------------------

    def activate_completed_projects(self, year: int) -> float:
        arriving_capacity = float(sum(self.pipeline.get(year, [])))
        self.current_capacity += arriving_capacity
        if year in self.pipeline:
            del self.pipeline[year]
        return arriving_capacity

    def compute_project_details(self, action: str, reference_demand: float) -> Dict[str, float]:
        self._validate_action(action)
        if action == "none":
            return {"capacity_added": 0.0, "investment_cost": 0.0, "lead_time": 0}

        option = self.config.options[action]
        capacity_added = option.size_factor * reference_demand
        investment_cost = option.fixed_cost + option.variable_cost * capacity_added
        return {
            "capacity_added": float(capacity_added),
            "investment_cost": float(investment_cost),
            "lead_time": option.lead_time,
        }

    def register_project(self, year: int, action: str, capacity_added: float) -> None:
        if action == "none" or capacity_added <= 0:
            return
        option = self.config.options[action]
        arrival_year = year + option.lead_time
        self.pipeline.setdefault(arrival_year, []).append(float(capacity_added))

    # -------------------------
    # Main yearly loop
    # -------------------------

    def simulate_year(self, year: int, adjustment_mode: str, action: str) -> Dict[str, Any]:
        self._validate_year(year)
        self._validate_adjustment(adjustment_mode)
        self._validate_action(action)

        if year != self.current_year:
            raise ValueError(f"Expected year {self.current_year}, received {year}.")

        capacity_activated = self.activate_completed_projects(year)
        baseline_forecast = self.get_baseline_forecast(year)
        adjusted_forecast = self.adjust_forecast(baseline_forecast, adjustment_mode)
        reference_demand = self.get_reference_demand(adjusted_forecast, year)

        project = self.compute_project_details(action, reference_demand)
        capacity_added = project["capacity_added"]
        investment_cost = project["investment_cost"]
        lead_time = project["lead_time"]

        feasible = investment_cost <= self.config.annual_budget
        rejected_for_budget = False
        original_action = action

        if not feasible:
            rejected_for_budget = True
            action = "none"
            capacity_added = 0.0
            investment_cost = 0.0
            lead_time = 0

        self.register_project(year, action, capacity_added)

        demand = float(self.true_demand_path[year])
        self.true_demand_history[year] = demand

        sales = min(demand, self.current_capacity)
        lost_sales = max(0.0, demand - self.current_capacity)
        unused_capacity = max(0.0, self.current_capacity - demand)

        revenue = self.config.unit_revenue * sales
        lost_sales_cost = self.config.lost_sales_penalty * lost_sales
        overcapacity_cost = self.config.overcapacity_penalty * unused_capacity
        annual_profit = revenue - investment_cost - lost_sales_cost - overcapacity_cost

        service_level = sales / demand if demand > 0 else 1.0
        utilization = sales / self.current_capacity if self.current_capacity > 0 else 0.0

        result = {
            "year": year,
            "baseline_forecast_current": baseline_forecast[year],
            "adjusted_forecast_current": adjusted_forecast[year],
            "reference_demand_for_sizing": reference_demand,
            "adjustment_mode": adjustment_mode,
            "requested_action": original_action,
            "action": action,
            "action_rejected_for_budget": rejected_for_budget,
            "capacity_start_of_year": self.current_capacity,
            "capacity_activated_this_year": capacity_activated,
            "capacity_added_to_pipeline": capacity_added,
            "project_lead_time": lead_time,
            "realized_demand": demand,
            "sales": sales,
            "lost_sales": lost_sales,
            "unused_capacity": unused_capacity,
            "revenue": revenue,
            "investment_cost": investment_cost,
            "lost_sales_cost": lost_sales_cost,
            "overcapacity_cost": overcapacity_cost,
            "annual_profit": annual_profit,
            "service_level": service_level,
            "utilization": utilization,
            "pipeline_snapshot": {k: round(sum(v), 4) for k, v in sorted(self.pipeline.items())},
        }

        self.results.append(result)
        self.current_year += 1
        return result

    def play_one_year(self, adjustment_mode: str, action: str) -> Dict[str, Any]:
        return self.simulate_year(self.current_year, adjustment_mode, action)

    # -------------------------
    # Summaries, charts, instructor review
    # -------------------------

    def get_results_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.results)

    def get_summary_metrics(self) -> Dict[str, float]:
        if not self.results:
            return {
                "cumulative_profit": 0.0,
                "average_service_level": 0.0,
                "average_utilization": 0.0,
                "total_lost_sales": 0.0,
                "total_unused_capacity": 0.0,
                "total_investment_cost": 0.0,
                "final_capacity": float(self.current_capacity),
            }
        df = self.get_results_df()
        return {
            "cumulative_profit": float(df["annual_profit"].sum()),
            "average_service_level": float(df["service_level"].mean()),
            "average_utilization": float(df["utilization"].mean()),
            "total_lost_sales": float(df["lost_sales"].sum()),
            "total_unused_capacity": float(df["unused_capacity"].sum()),
            "total_investment_cost": float(df["investment_cost"].sum()),
            "final_capacity": float(self.current_capacity),
        }

    def get_chart_data(self) -> pd.DataFrame:
        df = self.get_results_df().copy()
        if df.empty:
            return pd.DataFrame(columns=[
                "year", "realized_demand", "capacity_start_of_year",
                "sales", "annual_profit", "service_level", "utilization"
            ])
        return df[[
            "year", "realized_demand", "capacity_start_of_year",
            "sales", "annual_profit", "service_level", "utilization",
            "investment_cost", "lost_sales", "unused_capacity"
        ]].copy()

    def get_instructor_review(self) -> Dict[str, Any]:
        df = self.get_results_df()
        summary = self.get_summary_metrics()
        benchmark_df = self.compare_benchmarks()
        player_rank = None
        if not df.empty and not benchmark_df.empty:
            combined = pd.concat([
                benchmark_df[["policy_name", "cumulative_profit"]],
                pd.DataFrame([{"policy_name": "Current Player", "cumulative_profit": summary["cumulative_profit"]}])
            ], ignore_index=True)
            combined = combined.sort_values("cumulative_profit", ascending=False).reset_index(drop=True)
            player_rank = int(combined.index[combined["policy_name"] == "Current Player"][0] + 1)

        return {
            "summary": summary,
            "benchmarks": benchmark_df,
            "player_rank_by_profit": player_rank,
            "chart_data": self.get_chart_data(),
            "results_table": df,
        }

    # -------------------------
    # AI coach hooks
    # -------------------------

    def build_ai_context(self, include_full_history: bool = True) -> Dict[str, Any]:
        state = self.get_available_state_for_player() if self.current_year <= self.config.horizon else None
        payload = {
            "game": "capacity_strategy_simulation",
            "version": 3,
            "config": {
                k: v for k, v in asdict(self.config).items() if k != "options"
            },
            "options": {name: asdict(opt) for name, opt in self.config.options.items()},
            "current_year": self.current_year,
            "current_capacity": self.current_capacity,
            "pipeline": {k: sum(v) for k, v in self.pipeline.items()},
            "summary_metrics": self.get_summary_metrics(),
            "current_state": state,
        }
        if include_full_history:
            payload["results"] = self.results.copy()
            payload["demand_history"] = self.true_demand_history.copy()
        return payload

    def get_ai_system_prompt(self) -> str:
        return (
            "You are an operations strategy teaching coach for a capacity strategy simulation. "
            "Explain trade-offs clearly. Do not invent hidden information. Use only the provided game context. "
            "Focus on timing, forecast uncertainty, budget constraints, lost sales risk, and overcapacity risk. "
            "Be concise, actionable, and educational."
        )

    def get_ai_user_prompt(self) -> str:
        return (
            "Review the current game state and the player's past decisions. "
            "Give a short coaching note with: (1) what is going well, (2) the main current risk, "
            "and (3) one concrete recommendation for the next decision."
        )

    def get_rule_based_coach_message(self) -> str:
        if not self.results:
            return (
                "You are at the start of the game. Look at next year's forecast versus current capacity. "
                "If growth looks modest, keep flexibility. If the gap is already large, waiting may create lost sales."
            )

        last = self.results[-1]
        notes: List[str] = []

        if last["action_rejected_for_budget"]:
            notes.append("Your last requested expansion was rejected because it exceeded the annual budget.")

        if last["service_level"] < 0.95:
            notes.append("Service level fell below 95%, so capacity was likely too tight relative to realized demand.")
        elif last["utilization"] < 0.75:
            notes.append("Utilization was below 75%, which suggests excess capacity or overly early expansion.")
        else:
            notes.append("Your latest year stayed in a healthy operating range with reasonably aligned demand and capacity.")

        upcoming = None
        if self.current_year <= self.config.horizon:
            state = self.get_available_state_for_player()
            baseline = state["baseline_forecast"]
            next_year = min(self.current_year + 1, self.config.horizon)
            upcoming = baseline.get(next_year, baseline.get(self.current_year, 0.0))
            current_capacity = state["current_capacity"]
            if upcoming > current_capacity * 1.12:
                notes.append("The next-year baseline forecast is meaningfully above current capacity, so waiting may increase lost sales risk.")
            elif upcoming < current_capacity * 0.92:
                notes.append("The next-year forecast is below current capacity, so another expansion may increase overcapacity cost.")
            else:
                notes.append("The next-year forecast is fairly close to current capacity, so a cautious move may be reasonable.")

        notes.append("Use large capacity only when you are confident growth will stay strong long enough to justify the longer lead time.")
        return " ".join(notes)

    # -------------------------
    # Full runs and benchmark policies
    # -------------------------

    def run_simulation(self, policy: List[Dict[str, str]], reset_first: bool = True) -> pd.DataFrame:
        if len(policy) != self.config.horizon:
            raise ValueError("Policy length must equal simulation horizon.")
        if reset_first:
            self.reset(seed=self.seed)
        for year in range(1, self.config.horizon + 1):
            decision = policy[year - 1]
            self.simulate_year(year=year, adjustment_mode=decision["adjustment_mode"], action=decision["action"])
        return pd.DataFrame(self.results)

    def benchmark_policy_none(self) -> List[Dict[str, str]]:
        return [{"adjustment_mode": "base", "action": "none"} for _ in range(self.config.horizon)]

    def benchmark_policy_cautious(self) -> List[Dict[str, str]]:
        sim = self._clone_clean()
        policy = []
        for year in range(1, self.config.horizon + 1):
            state = sim.get_available_state_for_player(year)
            adjusted = sim.adjust_forecast(state["baseline_forecast"], "conservative")
            ref = sim.get_reference_demand(adjusted, year)
            action = "small" if ref > sim.current_capacity * 1.02 else "none"
            if sim.compute_project_details(action, ref)["investment_cost"] > sim.config.annual_budget:
                action = "none"
            policy.append({"adjustment_mode": "conservative", "action": action})
            sim.simulate_year(year, "conservative", action)
        return policy

    def benchmark_policy_balanced(self) -> List[Dict[str, str]]:
        sim = self._clone_clean()
        policy = []
        for year in range(1, self.config.horizon + 1):
            baseline = sim.get_baseline_forecast(year)
            adjusted = sim.adjust_forecast(baseline, "base")
            ref = sim.get_reference_demand(adjusted, year)
            gap_ratio = (ref - sim.current_capacity) / max(ref, 1e-9)
            if gap_ratio <= 0.00:
                action = "none"
            elif gap_ratio <= 0.07:
                action = "small"
            elif gap_ratio <= 0.16:
                action = "medium"
            else:
                action = "large"
            if sim.compute_project_details(action, ref)["investment_cost"] > sim.config.annual_budget:
                for candidate in ["medium", "small", "none"]:
                    if sim.compute_project_details(candidate, ref)["investment_cost"] <= sim.config.annual_budget:
                        action = candidate
                        break
            policy.append({"adjustment_mode": "base", "action": action})
            sim.simulate_year(year, "base", action)
        return policy

    def benchmark_policy_growth(self) -> List[Dict[str, str]]:
        sim = self._clone_clean()
        policy = []
        for year in range(1, self.config.horizon + 1):
            baseline = sim.get_baseline_forecast(year)
            adjusted = sim.adjust_forecast(baseline, "aggressive")
            ref = sim.get_reference_demand(adjusted, year)
            current = sim.current_capacity
            if ref <= current * 1.00:
                action = "none"
            elif ref <= current * 1.08:
                action = "small"
            elif ref <= current * 1.18:
                action = "medium"
            else:
                action = "large"
            if sim.compute_project_details(action, ref)["investment_cost"] > sim.config.annual_budget:
                for candidate in ["medium", "small", "none"]:
                    if sim.compute_project_details(candidate, ref)["investment_cost"] <= sim.config.annual_budget:
                        action = candidate
                        break
            policy.append({"adjustment_mode": "aggressive", "action": action})
            sim.simulate_year(year, "aggressive", action)
        return policy

    def evaluate_policy(self, policy: List[Dict[str, str]]) -> Dict[str, Any]:
        sim = self._clone_clean()
        df = sim.run_simulation(policy, reset_first=False)
        return {"results": df, "summary": sim.get_summary_metrics()}

    def compare_benchmarks(self) -> pd.DataFrame:
        benchmarks = {
            "Do Nothing": self.benchmark_policy_none(),
            "Cautious": self.benchmark_policy_cautious(),
            "Balanced": self.benchmark_policy_balanced(),
            "Growth": self.benchmark_policy_growth(),
        }
        rows = []
        for name, policy in benchmarks.items():
            evaluation = self.evaluate_policy(policy)
            rows.append({"policy_name": name, **evaluation["summary"]})
        return pd.DataFrame(rows)

    # -------------------------
    # Helpers
    # -------------------------

    def _clone_clean(self) -> "CapacityStrategySimulationV3":
        return CapacityStrategySimulationV3(config=copy.deepcopy(self.config), seed=self.seed)

    def _validate_year(self, year: int) -> None:
        if year < 1 or year > self.config.horizon:
            raise ValueError(f"Year must be between 1 and {self.config.horizon}.")

    def _validate_action(self, action: str) -> None:
        if action not in self.VALID_ACTIONS:
            raise ValueError(f"Invalid action: {action}")

    def _validate_adjustment(self, adjustment_mode: str) -> None:
        if adjustment_mode not in self.VALID_ADJUSTMENTS:
            raise ValueError(f"Invalid adjustment mode: {adjustment_mode}")


def demo_run() -> None:
    config = SimulationConfig()
    sim = CapacityStrategySimulationV3(config=config, seed=42)
    policy = [
        {"adjustment_mode": "base", "action": "none"},
        {"adjustment_mode": "aggressive", "action": "small"},
        {"adjustment_mode": "base", "action": "medium"},
        {"adjustment_mode": "base", "action": "large"},
        {"adjustment_mode": "conservative", "action": "none"},
        {"adjustment_mode": "aggressive", "action": "small"},
        {"adjustment_mode": "base", "action": "medium"},
        {"adjustment_mode": "conservative", "action": "none"},
        {"adjustment_mode": "base", "action": "small"},
        {"adjustment_mode": "base", "action": "none"},
    ]
    results = sim.run_simulation(policy)
    print(results.round(2).to_string(index=False))
    print("\nSummary:")
    for k, v in sim.get_summary_metrics().items():
        print(f"{k}: {v:.2f}")
    print("\nCoach note:")
    print(sim.get_rule_based_coach_message())


if __name__ == "__main__":
    demo_run()
