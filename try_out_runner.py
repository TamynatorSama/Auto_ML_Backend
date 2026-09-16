"""Run the AutoML agent over every dataset in try_out_data/.

The two subgraphs are invoked separately rather than through `main.app` for one
reason: the config generator echoes `id` 1 for every run, and `run_dir` is
`runs/<id>`, so four runs through the combined graph would write four models'
scripts, pickles and reports into the same directory. The context is re-stamped
with a per-dataset run id and directory between the two phases; nothing else
about the flow changes.
"""

import json
import os
import sys
import time
import traceback
from pathlib import Path

from models import DataColumn, DataSchema, DataType
from information_agent.base import app as information_app
from code_gen_eval.base import app as code_gen_eval_app

NUM = DataType.NUMERIC
CAT = DataType.CATEGORICAL
TXT = DataType.TEXT


# ---------------------------------------------------------------------------
# building materials — revenue
# ---------------------------------------------------------------------------

building_materials = DataSchema(
    name="building_materials_transactions",
    description=(
        "Building materials distributor transactions, 2019 onwards, one row per "
        "order line, with the housing and lumber market conditions of that week"
    ),
    columns=[
        DataColumn(name="date", data_type=TXT, description="Transaction date (weekly, ISO format)"),
        DataColumn(name="year", data_type=NUM, description="Calendar year of the transaction"),
        DataColumn(name="month", data_type=NUM, description="Calendar month of the transaction (1-12)"),
        DataColumn(name="week_of_year", data_type=NUM, description="ISO week number of the transaction (1-53)"),
        DataColumn(name="region", data_type=CAT, description="US census division the order shipped to"),
        DataColumn(name="product_category", data_type=CAT, description="Material family sold (lumber, roofing, concrete, and so on)"),
        DataColumn(name="sku", data_type=TXT, description="Stock keeping unit identifier of the item sold"),
        DataColumn(name="channel", data_type=CAT, description="Sales channel (pro_dealer or retail)"),
        DataColumn(name="customer_type", data_type=CAT, description="Buyer type (contractor, remodeler, builder, DIY)"),
        DataColumn(name="units", data_type=NUM, description="Quantity of the item sold on this line"),
        DataColumn(name="unit_price", data_type=NUM, description="Price charged per unit, in dollars"),
        DataColumn(name="housing_starts_index", data_type=NUM, description="National housing starts index for that week"),
        DataColumn(name="lumber_price_index", data_type=NUM, description="Lumber commodity price index for that week"),
        DataColumn(name="mortgage_rate", data_type=NUM, description="Average 30 year mortgage rate that week, in percent"),
        DataColumn(name="revenue", data_type=NUM, description="Line revenue in dollars", is_target=True),
    ],
)


# ---------------------------------------------------------------------------
# EV battery — battery_failure
# ---------------------------------------------------------------------------

ev_battery = DataSchema(
    name="ev_battery_failure",
    description=(
        "Electric vehicle battery telemetry, one row per vehicle, covering pack "
        "state, charging behaviour, driving style, environment and service history"
    ),
    columns=[
        DataColumn(name="vehicle_id", data_type=TXT, description="Unique vehicle identifier"),
        DataColumn(name="vehicle_brand", data_type=CAT, description="Vehicle manufacturer"),
        DataColumn(name="vehicle_model", data_type=CAT, description="Vehicle model name"),
        DataColumn(name="vehicle_type", data_type=CAT, description="Body style (sedan, SUV, truck, and so on)"),
        DataColumn(name="manufacturing_year", data_type=NUM, description="Year the vehicle was built"),
        DataColumn(name="battery_manufacturer", data_type=CAT, description="Company that made the battery pack"),
        DataColumn(name="battery_chemistry", data_type=CAT, description="Cell chemistry (NMC, NCA, LFP, and so on)"),
        DataColumn(name="battery_capacity_kwh", data_type=NUM, description="Nameplate pack capacity in kilowatt hours"),
        DataColumn(name="drive_type", data_type=CAT, description="Drivetrain layout (FWD, RWD, AWD)"),
        DataColumn(name="odometer_km", data_type=NUM, description="Distance driven to date, in kilometres"),
        DataColumn(name="vehicle_age_years", data_type=NUM, description="Vehicle age in years"),
        DataColumn(name="fleet_or_private", data_type=CAT, description="Whether the vehicle is fleet or privately owned"),
        DataColumn(name="battery_serial", data_type=TXT, description="Serial number of the battery pack"),
        DataColumn(name="cycle_count", data_type=NUM, description="Total charge/discharge cycles the pack has seen"),
        DataColumn(name="battery_health_percent", data_type=NUM, description="Reported pack health as a percentage of new"),
        DataColumn(name="state_of_charge", data_type=NUM, description="Current charge level, percent"),
        DataColumn(name="depth_of_discharge", data_type=NUM, description="Typical discharge depth per cycle, percent"),
        DataColumn(name="state_of_health", data_type=NUM, description="Pack state of health, percent of original capacity"),
        DataColumn(name="cell_voltage_avg", data_type=NUM, description="Mean cell voltage across the pack, volts"),
        DataColumn(name="cell_voltage_std", data_type=NUM, description="Standard deviation of cell voltage, volts"),
        DataColumn(name="pack_voltage", data_type=NUM, description="Total pack voltage, volts"),
        DataColumn(name="cell_temperature_avg", data_type=NUM, description="Mean cell temperature, Celsius"),
        DataColumn(name="cell_temperature_max", data_type=NUM, description="Maximum recorded cell temperature, Celsius"),
        DataColumn(name="internal_resistance", data_type=NUM, description="Pack internal resistance, ohms"),
        DataColumn(name="charge_efficiency", data_type=NUM, description="Energy retained when charging, percent"),
        DataColumn(name="discharge_efficiency", data_type=NUM, description="Energy delivered when discharging, percent"),
        DataColumn(name="remaining_capacity", data_type=NUM, description="Usable capacity left, kilowatt hours"),
        DataColumn(name="capacity_loss_percent", data_type=NUM, description="Capacity lost since new, percent"),
        DataColumn(name="charging_cycles_last_month", data_type=NUM, description="Charge cycles in the last 30 days"),
        DataColumn(name="fast_charge_ratio", data_type=NUM, description="Share of charging sessions done on DC fast chargers, 0-1"),
        DataColumn(name="slow_charge_ratio", data_type=NUM, description="Share of charging sessions done on AC slow chargers, 0-1"),
        DataColumn(name="average_charge_power_kw", data_type=NUM, description="Mean charging power, kilowatts"),
        DataColumn(name="average_charging_time", data_type=NUM, description="Mean session length, minutes"),
        DataColumn(name="overnight_charging_ratio", data_type=NUM, description="Share of charging done overnight, 0-1"),
        DataColumn(name="home_charging_ratio", data_type=NUM, description="Share of charging done at home, 0-1"),
        DataColumn(name="charging_interruptions", data_type=NUM, description="Count of interrupted charging sessions"),
        DataColumn(name="overcharge_events", data_type=NUM, description="Count of overcharge events recorded"),
        DataColumn(name="average_speed", data_type=NUM, description="Mean driving speed, km/h"),
        DataColumn(name="average_trip_distance", data_type=NUM, description="Mean trip length, kilometres"),
        DataColumn(name="aggressive_acceleration_score", data_type=NUM, description="Aggressive acceleration score, higher is harsher"),
        DataColumn(name="hard_braking_score", data_type=NUM, description="Hard braking score, higher is harsher"),
        DataColumn(name="regenerative_braking_usage", data_type=NUM, description="Regenerative braking usage, percent"),
        DataColumn(name="highway_driving_ratio", data_type=NUM, description="Share of distance driven on highways, 0-1"),
        DataColumn(name="city_driving_ratio", data_type=NUM, description="Share of distance driven in cities, 0-1"),
        DataColumn(name="daily_distance", data_type=NUM, description="Mean distance driven per day, kilometres"),
        DataColumn(name="average_ambient_temperature", data_type=NUM, description="Mean ambient temperature at the vehicle, Celsius"),
        DataColumn(name="maximum_temperature", data_type=NUM, description="Maximum ambient temperature seen, Celsius"),
        DataColumn(name="minimum_temperature", data_type=NUM, description="Minimum ambient temperature seen, Celsius"),
        DataColumn(name="humidity", data_type=NUM, description="Mean ambient humidity, percent"),
        DataColumn(name="altitude", data_type=NUM, description="Mean operating altitude, metres"),
        DataColumn(name="terrain_type", data_type=CAT, description="Dominant terrain driven (flat, hilly, mountainous, coastal, desert)"),
        DataColumn(name="dust_exposure", data_type=NUM, description="Dust exposure index of the operating environment"),
        DataColumn(name="last_service_days", data_type=NUM, description="Days since the last service visit"),
        DataColumn(name="cooling_system_health", data_type=NUM, description="Thermal management system health, percent"),
        DataColumn(name="firmware_updates", data_type=NUM, description="Count of firmware updates applied"),
        DataColumn(name="previous_faults", data_type=NUM, description="Count of faults logged before this observation"),
        DataColumn(name="maintenance_score", data_type=NUM, description="Maintenance compliance score, higher is better kept"),
        DataColumn(name="thermal_runaway_risk", data_type=NUM, description="Modelled thermal runaway risk index"),
        DataColumn(name="voltage_imbalance", data_type=NUM, description="Cell voltage imbalance across the pack, millivolts"),
        DataColumn(name="temperature_variance", data_type=NUM, description="Variance of cell temperature across the pack"),
        DataColumn(name="sensor_fault_count", data_type=NUM, description="Count of faulty sensors reported"),
        DataColumn(name="BMS_warning_count", data_type=NUM, description="Count of battery management system warnings"),
        DataColumn(name="abnormal_voltage_events", data_type=NUM, description="Count of abnormal voltage events"),
        DataColumn(name="battery_stress_index", data_type=NUM, description="Composite battery stress index"),
        DataColumn(name="aging_score", data_type=NUM, description="Composite aging score"),
        DataColumn(name="thermal_health_score", data_type=NUM, description="Composite thermal health score"),
        DataColumn(name="charging_quality_score", data_type=NUM, description="Composite charging behaviour quality score"),
        DataColumn(name="driving_stress_score", data_type=NUM, description="Composite driving stress score"),
        DataColumn(name="predicted_remaining_life_cycles", data_type=NUM, description="Vendor estimate of cycles of life left in the pack"),
        DataColumn(name="battery_failure", data_type=CAT, description="Whether the battery failed (0 = no, 1 = yes)", is_target=True),
    ],
)


# ---------------------------------------------------------------------------
# sales — Profit
# ---------------------------------------------------------------------------

sales = DataSchema(
    name="sales_transactions_2022_2025",
    description=(
        "Retail and online sales transactions across six countries, 2022 to 2025, "
        "one row per transaction line with customer, store, product and fulfilment detail"
    ),
    columns=[
        DataColumn(name="Transaction_ID", data_type=TXT, description="Unique transaction line identifier"),
        DataColumn(name="Order_ID", data_type=TXT, description="Order the line belongs to"),
        DataColumn(name="Customer_ID", data_type=TXT, description="Unique customer identifier"),
        DataColumn(name="Customer_Name", data_type=TXT, description="Customer full name"),
        DataColumn(name="Customer_Age", data_type=NUM, description="Customer age in years"),
        DataColumn(name="Customer_Gender", data_type=CAT, description="Customer gender as recorded"),
        DataColumn(name="Customer_Segment", data_type=CAT, description="Customer segment (Consumer, Corporate, Home Office)"),
        DataColumn(name="Order_Date", data_type=TXT, description="Date the order was placed (ISO format)"),
        DataColumn(name="Order_Time", data_type=TXT, description="Time of day the order was placed"),
        DataColumn(name="Sales_Channel", data_type=CAT, description="Channel the order came through (Online, In-Store, and so on)"),
        DataColumn(name="Store_ID", data_type=CAT, description="Identifier of the store fulfilling the order"),
        DataColumn(name="Store_Name", data_type=CAT, description="Name of the store fulfilling the order"),
        DataColumn(name="Country", data_type=CAT, description="Country of the order"),
        DataColumn(name="Region", data_type=CAT, description="Region or state of the order"),
        DataColumn(name="City", data_type=CAT, description="City of the order"),
        DataColumn(name="Product_ID", data_type=CAT, description="Identifier of the product sold"),
        DataColumn(name="Product_Name", data_type=CAT, description="Name of the product sold"),
        DataColumn(name="Product_Category", data_type=CAT, description="Product category"),
        DataColumn(name="Product_Subcategory", data_type=CAT, description="Product subcategory"),
        DataColumn(name="Quantity", data_type=NUM, description="Units sold on this line"),
        DataColumn(name="Unit_Price", data_type=NUM, description="List price per unit before discount"),
        DataColumn(name="Discount_Percentage", data_type=NUM, description="Discount applied to the line, percent"),
        DataColumn(name="Sales_Amount", data_type=NUM, description="Revenue booked on the line after discount"),
        DataColumn(name="Cost_Amount", data_type=NUM, description="Cost of goods sold on the line"),
        DataColumn(name="Payment_Method", data_type=CAT, description="How the customer paid"),
        DataColumn(name="Order_Status", data_type=CAT, description="Fulfilment status of the order"),
        DataColumn(name="Shipping_Method", data_type=CAT, description="Shipping service used"),
        DataColumn(name="Delivery_Days", data_type=NUM, description="Days taken to deliver"),
        DataColumn(name="Return_Flag", data_type=CAT, description="Whether the line was returned (Yes or No)"),
        DataColumn(name="Return_Reason", data_type=CAT, description="Reason given for the return, blank when not returned"),
        DataColumn(name="Sales_Representative", data_type=CAT, description="Sales representative credited with the order"),
        DataColumn(name="Promotion_Code", data_type=CAT, description="Promotion code applied, blank when none"),
        DataColumn(name="Customer_Rating", data_type=NUM, description="Rating the customer left, 1 to 5"),
        DataColumn(name="Inventory_Level", data_type=NUM, description="Stock on hand for the product at order time"),
        DataColumn(name="Order_Year", data_type=NUM, description="Calendar year of the order"),
        DataColumn(name="Profit", data_type=NUM, description="Profit booked on the line, in dollars", is_target=True),
    ],
)


# ---------------------------------------------------------------------------
# telecom — churn
# ---------------------------------------------------------------------------

telecom = DataSchema(
    name="telecom_churn",
    description=(
        "Telecom subscribers, one row per customer, with plan, billing, usage, "
        "support and satisfaction history, labelled with whether they churned"
    ),
    columns=[
        DataColumn(name="customer_id", data_type=TXT, description="Unique customer identifier"),
        DataColumn(name="signup_date", data_type=TXT, description="Date the customer signed up (ISO format)"),
        DataColumn(name="age", data_type=NUM, description="Customer age in years"),
        DataColumn(name="region", data_type=CAT, description="Region the customer is served from"),
        DataColumn(name="plan", data_type=CAT, description="Subscription tier (Basic, Standard, Premium, and so on)"),
        DataColumn(name="tenure_months", data_type=NUM, description="Months the customer has been subscribed"),
        DataColumn(name="num_devices", data_type=NUM, description="Devices on the account"),
        DataColumn(name="monthly_charges", data_type=NUM, description="Current monthly bill, in dollars"),
        DataColumn(name="discount_pct", data_type=NUM, description="Discount on the monthly bill, percent"),
        DataColumn(name="is_autopay", data_type=CAT, description="Whether the customer pays automatically (0 = no, 1 = yes)"),
        DataColumn(name="avg_session_minutes", data_type=NUM, description="Mean session length, minutes"),
        DataColumn(name="sessions_last_30d", data_type=NUM, description="Sessions in the last 30 days"),
        DataColumn(name="last_login_days_ago", data_type=NUM, description="Days since the customer last logged in"),
        DataColumn(name="support_tickets_90d", data_type=NUM, description="Support tickets raised in the last 90 days"),
        DataColumn(name="payment_failures_12m", data_type=NUM, description="Failed payments in the last 12 months"),
        DataColumn(name="satisfaction_score", data_type=NUM, description="Latest satisfaction survey score, 1 to 5"),
        DataColumn(name="total_charges", data_type=NUM, description="Lifetime amount billed to the customer, in dollars"),
        DataColumn(name="churn", data_type=CAT, description="Whether the customer churned (0 = no, 1 = yes)", is_target=True),
    ],
)


RUNS = [
    {
        "run_id": 201,
        "slug": "telecom_churn",
        "topic": "Telecom customer churn prediction",
        "data_path": "try_out_data/telecom/telecom_churn.csv",
        "schema": telecom,
    },
    {
        "run_id": 202,
        "slug": "ev_battery_failure",
        "topic": "Electric vehicle battery failure prediction",
        "data_path": "try_out_data/ev_battery/ev_battery_failure.csv",
        "schema": ev_battery,
    },
    {
        "run_id": 203,
        "slug": "sales_profit",
        "topic": "Retail sales profit prediction",
        "data_path": "try_out_data/sales/Sales_transactions_2022_2025.csv",
        "schema": sales,
    },
    {
        "run_id": 204,
        "slug": "building_materials_revenue",
        "topic": "Building materials transaction revenue prediction",
        "data_path": "try_out_data/building_materials/building_materials_transactions.csv",
        "schema": building_materials,
    },
]


def execute(spec: dict) -> dict:
    run_dir = Path("runs") / spec["slug"]
    started = time.time()

    print(f"\n{'=' * 78}\n>>> {spec['slug']}: {spec['topic']}\n{'=' * 78}", flush=True)

    information = information_app.invoke(
        {"data_path": spec["data_path"], "schema": spec["schema"], "topic": spec["topic"]}
    )

    # the generator echoes id 1 for every dataset; without this the four runs
    # share one directory and overwrite each other's scripts and models
    context = information["context"].model_copy(
        update={"run_id": spec["run_id"], "run_dir": str(run_dir)}
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "context.json").write_text(
        context.model_dump_json(indent=2), encoding="utf-8"
    )
    (run_dir / "profile.md").write_text(context.summary, encoding="utf-8")
    (run_dir / "profile_full.md").write_text(information["full_summary"], encoding="utf-8")

    # one worker is the safe setting on a large dataset when memory is tight:
    # two workers on 306k training rows ran the machine out of memory
    result = code_gen_eval_app.invoke(
        {"topic": spec["topic"], "context": context, "config": information["config"]},
        {"max_concurrency": int(os.environ.get("AUTOML_CONCURRENCY", "2"))},
    )

    (run_dir / "report.md").write_text(result["report"], encoding="utf-8")
    elapsed = time.time() - started
    print(f"\n<<< {spec['slug']} finished in {elapsed / 60:.1f} min -> {run_dir}", flush=True)

    report = result["run_report"]
    return {
        "slug": spec["slug"],
        "status": report.status,
        "task_type": report.dataset.task_type,
        "target": report.dataset.target,
        "selected_model": report.selected_model,
        "primary_metric": next((m.name for m in report.metrics if m.primary), None),
        "scores": next(
            (row.test_scores for row in report.comparison if row.selected), {}
        ),
        "run_dir": str(run_dir),
        "minutes": round(elapsed / 60, 1),
    }


if __name__ == "__main__":
    wanted = sys.argv[1:]
    selected = [r for r in RUNS if not wanted or r["slug"] in wanted]

    summary = []
    for spec in selected:
        try:
            summary.append(execute(spec))
        except Exception as error:
            print(f"!!! {spec['slug']} failed: {error!r}", flush=True)
            traceback.print_exc()
            summary.append({"slug": spec["slug"], "status": f"crashed: {error!r}"})

        Path("runs/try_out_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

    print("\n===============> all runs <===============")
    print(json.dumps(summary, indent=2))
