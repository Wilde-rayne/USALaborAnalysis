# Chart-library examples for the image-writer agent

The `ImageWriterAgent` (utils/agents/image_writer.py) needs a small,
curated corpus of example chart kinds the orchestrator can serve up
when grounding a `FigureSpec`. This file is the seed corpus — a
deliberately narrow set of canonical labor-market visuals expressed
in the four libraries we use.

The goal is **bounded vocabulary**: the LLM picks one of these kinds,
emits a structured `FigureSpec`, and a Python builder turns the spec
into a real figure. The LLM does not write arbitrary plotting code.

## Bounded chart kinds

### 1. `line_with_forecast` (Plotly / Dash)

Time-series line with historic actual + forecast line + 95% CI fan +
a hatched "data-lag" gap region. Used by LFP forecast panel today.

```python
import plotly.graph_objs as go

fig = go.Figure([
    go.Scatter(x=hist_x, y=hist_y, mode="lines", name="historic",
               line=dict(color="rgb(31,119,180)", width=2)),
    go.Scatter(x=fc_x, y=fc_upper, mode="lines",
               line=dict(width=0), showlegend=False, hoverinfo="skip"),
    go.Scatter(x=fc_x, y=fc_lower, mode="lines", name="95% CI",
               fill="tonexty", fillcolor="rgba(31,119,180,0.18)",
               line=dict(width=0), hoverinfo="skip"),
    go.Scatter(x=fc_x, y=fc_point, mode="lines+markers",
               name=f"forecast ({model} · RMSE {rmse:.2f})",
               line=dict(color="rgb(31,119,180)", width=2.5)),
])
fig.add_shape(type="rect", xref="x", yref="paper",
              x0=gap_start, x1=gap_end, y0=0, y1=1,
              fillcolor="rgba(120,120,120,0.10)", line=dict(width=0),
              layer="below")
```

**FigureSpec keys**: `kind="line_with_forecast"`, `series=[
{name, x, y, role: "historic"|"forecast_point"|"forecast_ci"}]`,
`annotations=[{x, text, role: "gap"|"forecast_start"}]`.

### 2. `bar_with_ci` (Plotly / Dash)

Per-state forecast bars with asymmetric error-bar caps for the 95%
prediction interval. Used by Super (supersector) tab today.

```python
fig = go.Figure([go.Bar(
    x=labels, y=points, marker=dict(color=colors),
    error_y=dict(type="data", symmetric=False,
                 array=upper_offsets, arrayminus=lower_offsets,
                 color="rgba(60,70,90,0.55)", thickness=1.5, width=6),
    text=[f"{v:,.0f}" for v in points], textposition="auto",
)])
```

**FigureSpec keys**: `kind="bar_with_ci"`, `series=[{label, point,
ci_lower, ci_upper, model}]`.

### 3. `kde` (matplotlib / seaborn)

Per-state distribution overlay — kernel density estimate. Useful for
the EDA distribution panel.

```python
import seaborn as sns
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 4))
for label, values in series.items():
    sns.kdeplot(values, ax=ax, label=label, fill=True, alpha=0.25)
ax.set_xlabel(measure_name); ax.set_ylabel("density")
ax.legend(loc="upper right", fontsize=9)
```

**FigureSpec keys**: `kind="kde"`, `series=[{label, values}]`,
`x_field=measure_name`.

### 4. `regplot` (sklearn / seaborn)

Scatter + regression line for human-geography overlays — wages vs
LFP, education vs employment. Adds the OLS slope + R² in the title.

```python
from sklearn.linear_model import LinearRegression
import seaborn as sns

X = features_x.reshape(-1, 1)
model = LinearRegression().fit(X, target_y)
r2 = model.score(X, target_y)
slope = model.coef_[0]

fig, ax = plt.subplots(figsize=(7, 5))
sns.regplot(x=features_x, y=target_y, ax=ax,
            scatter_kws={"s": 30, "alpha": 0.6})
ax.set_title(f"{x_label} vs {y_label} — slope {slope:.3f}, R² {r2:.2f}")
```

**FigureSpec keys**: `kind="regplot"`, `series=[{x_field, y_field}]`,
`color_scheme="brand"`.

### 5. `confusion_or_score_matrix` (sklearn)

For requirements / pass-fail panels — a state×criterion grid showing
which thresholds each state clears. Uses sklearn's metric vocabulary.

```python
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# Rows: states; Cols: thresholds. Cell = passes/fails.
matrix = np.array(grid)  # shape (n_states, n_thresholds)
fig, ax = plt.subplots(figsize=(6, 4))
im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
ax.set_xticks(range(len(thresholds))); ax.set_xticklabels(thresholds)
ax.set_yticks(range(len(states)));     ax.set_yticklabels(states)
fig.colorbar(im, ax=ax, label="passes")
```

**FigureSpec keys**: `kind="confusion_or_score_matrix"`,
`series=[{row_label, col_label, value}]`.

### 6. `training_curve` (keras / pytorch)

For LSTM forecast diagnostics — train + validation loss across
epochs. Reserves the cell for when LSTM telemetry is exposed.

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(history.history["loss"], label="train")
ax.plot(history.history["val_loss"], label="val")
ax.set_xlabel("epoch"); ax.set_ylabel("loss")
ax.legend()
```

**FigureSpec keys**: `kind="training_curve"`, `series=[{name: "train"|"val", values}]`.

## Statistical-rigor invariants

Each kind ships with a small invariant set in `utils/forecasting/invariants.py`:

| Invariant | Applies to | Caught failure |
|---|---|---|
| `point_within_historical_envelope` | line_with_forecast, bar_with_ci | model produced a forecast outside ±3σ of history |
| `ci_contains_point` | line_with_forecast, bar_with_ci | the model's CI doesn't bracket its own point |
| `ci_width_reasonable` | line_with_forecast, bar_with_ci | the model's CI is wider than ±8σ — i.e. uninformative |
| `rmse_against_baseline` | line_with_forecast, bar_with_ci | "winning" model didn't actually beat naive |

The orchestrator runs `audit_forecast(...)` on each forecast view_state
before shipping. Any failures are appended to the reviewer's critique
context so the narrative acknowledges the caveat instead of claiming
certainty the data doesn't support.

## How the image-writer slice will use this

1. `ImageWriterAgent.write_spec(view_state)` reads `view_state["_kind"]`,
   maps to a chart kind from the bounded list above, and emits a
   strict-JSON `FigureSpec`.
2. A small `tabs/_charts.py` builder (next slice) takes the
   `FigureSpec` and produces the actual `go.Figure` / matplotlib
   figure using the templates above.
3. `TileReviewer` (utils/agents/reviewer.py) audits the
   (figure_spec, blurb) pair for consistency before shipping.
4. The orchestrator's reviewers see invariant failures from
   `utils/forecasting/invariants.py` and fold them into critique
   prompts so the narrative stays honest about model uncertainty.
