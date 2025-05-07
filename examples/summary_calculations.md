# WOMBAT examples

For all the examples, I will be working with the following set of assumptions

```python
import pandas as pd
from wombat import Simulation

library = "library/corewind"
config = "morro_bay_in_situ_consolidated.yaml"

sim = Simulation(library, config)
sim.run()
sim.env.cleanup_log_files()

# Years is dependent on if the weather profile contains leap years. In the COREWIND
# data, this is the case, so 365.25 should be used in place of 365
years = round((sim.env.end_datetime - sim.env.start_datetime).days / 365.25, 2)

metrics = sim.metrics
ev = metrics.events
```

## Annual average materials costs

The `metrics.component_costs` does not fully delineate the breakdown available in your
slides, so I would recommend the following calculation to get the subassembly costs by
failure mode. I will plan to implement this in WOMBAT so the math can be behind the
scenes.

```python
materials = (
    ev
    .loc[
        ev.materials_cost.gt(0)
        & ev.request_id.startsiwth("RPR"),  # NOTE: remove this filter to include maintenance costs
        ["part_name", "reason", "materials_cost"]
]
)
materials.loc[materials.part_name.str.startswith("ARR"), "part_name"] = "array cable"
materials.loc[materials.part_name.str.startswith("EXP"), "part_name"] = "export cable"
total_materials = materials.groupby(["part_name", "reason"]).sum() / years
```

## Average annual failures

This number of failures in a simulation can be computed using the `metrics.process_times()`
method, but it does not include the subassembly category. To attach the subassembly data
for plotting, we can join the annual occurrences with the above materials costs breakdown.
As a note, if the maintenance tasks are filtered out of the materials costs, then the
default left join will filter out the maintenance occurrences as well.

```python
timing = (
    metrics
    .process_times()[["N"]]
    .rename(columns={"N": "annual_occurrences"})
    / years
)

average_failures_costs = (
    total_materials
    .reset_index(drop=False)
    .rename(columns={"part_name": "subassembly", "reason": "category"})
    .set_index("category")
    .join(timing)
    .reset_index(drop=False)
    .set_index(["subassembly", "category"])
)
```

# Vessel delay summary

The best way to address the specific utilization of each vessel is to break down
the cause of delays, which I am outlining below. As a note, I am shortening the 
actual delay messages to common categories so that the delays are appropriately
grouped.

```python
delay_summary = (
    ev
    .loc[
        ev.agent.isin(sim.service_equipment)
        & ev.duration.gt(0)
        & ev.action.eq("delay"),
        ["agent", "additional", "duration"]
    ]
    .groupby(["agent", "additional"])
    .sum()
    .reset_index(drop=False)
    .replace({
        "no work requests submitted by start of shift": "no requests",
        "no work requests, waiting until the next shift": "no requests",
        "weather unsuitable to transfer crew": "weather delay",
        "work shift has ended; waiting for next shift to start": "end of shift",
        "insufficient time to complete travel before end of the shift": "end of shift",
        "will return next year": "end of charter",
    })
    .set_index(["agent", "additional"])
    / 24
)
```

# Vessel mobilization summary

The builtin mobilization cost summary in the service equipment costs likely does not
provide all the information needed, so below demonstrates how to get the number of
mobilizations, days spent mobilizing, and the total cost of mobilization by vessel.

```python
mobilization_summary = (
    ev
    .loc[ev.action.eq("mobilization") & ev.duration.gt(0), ["agent", "duration"]]
    .groupby("agent")
    .count()
    .rename(columns={"duration": "mobilizations"})
    .join(
        ev
        .loc[ev.action.eq("mobilization"), ["agent", "duration", "equipment_cost"]]
        .groupby("agent")
        .sum()
    )
)
mobilization_summary.duration /= 24
```
