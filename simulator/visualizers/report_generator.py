"""Generate HTML reports for simulation results."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict

try:
    from jinja2 import Template
except ImportError:
    Template = None


def _create_latency_chart(results: Dict[str, Any]) -> str:
    responses = results.get("responses", [])
    latencies = [r.get("latency_ms", 0) for r in responses if "latency_ms" in r]
    try:
        import plotly.graph_objs as go
        fig = go.Figure(data=[go.Histogram(x=latencies, nbinsx=50)])
        fig.update_layout(
            title="Latency Distribution",
            xaxis_title="Latency (ms)",
            yaxis_title="Count",
        )
        return fig.to_html(include_plotlyjs="cdn", div_id="latency-chart")
    except ImportError:
        return f"<p>Latency samples: n={len(latencies)}</p>"


def _create_throughput_chart(results: Dict[str, Any]) -> str:
    try:
        import plotly.graph_objs as go
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                y=[results.get("peak_rps", 0)],
                mode="lines+markers",
                name="RPS",
            )
        )
        fig.update_layout(title="Throughput")
        return fig.to_html(include_plotlyjs="cdn", div_id="throughput-chart")
    except ImportError:
        return f"<p>Peak RPS: {results.get('peak_rps', 0):.0f}</p>"


def _create_error_chart(results: Dict[str, Any]) -> str:
    error_rate = results.get("error_rate", 0) * 100
    try:
        import plotly.graph_objs as go
        fig = go.Figure(data=[go.Bar(x=["Error Rate"], y=[error_rate])])
        fig.update_layout(title="Error Rate", yaxis_title="Percentage (%)")
        return fig.to_html(include_plotlyjs="cdn", div_id="error-chart")
    except ImportError:
        return f"<p>Error rate: {error_rate:.2f}%</p>"


def _fraud_display(results: Dict[str, Any]) -> tuple[float, str]:
    """Return (value as percentage 0-100, label) for the fraud metric to show."""
    if "fraud_precision" in results:
        return (float(results["fraud_precision"]) * 100, "Fraud Precision")
    if "fraud_recall" in results:
        return (float(results["fraud_recall"]) * 100, "Fraud Recall")
    return (float(results.get("fraud_detected_pct", 0)), "Fraud Detected")


def _create_fraud_chart(results: Dict[str, Any]) -> str:
    fraud_pct, fraud_label = _fraud_display(results)
    try:
        import plotly.graph_objs as go
        fig = go.Figure(
            data=[go.Bar(x=[fraud_label], y=[fraud_pct], marker_color="green")]
        )
        fig.update_layout(
            title="Fraud Detection Rate",
            yaxis_title="Percentage (%)",
        )
        return fig.to_html(include_plotlyjs="cdn", div_id="fraud-chart")
    except ImportError:
        return f"<p>{fraud_label}: {fraud_pct:.0f}%</p>"


TEMPLATE_STR = """
<!DOCTYPE html>
<html>
<head>
    <title>Simulation Report - {{ scenario_name }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { background: #2c3e50; color: white; padding: 20px; border-radius: 5px; }
        .metrics { display: grid; grid-template-columns: repeat(5, 1fr); gap: 20px; margin: 20px 0; }
        .metric-card { background: #ecf0f1; padding: 20px; border-radius: 5px; text-align: center; }
        .metric-value { font-size: 2em; font-weight: bold; color: #3498db; }
        .status-pass { color: #27ae60; }
        .status-fail { color: #e74c3c; }
        .chart-container { margin: 30px 0; }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{ scenario_name }}</h1>
        <p>{{ description }}</p>
        <p><small>Generated: {{ timestamp }}</small></p>
    </div>
    <h2>Overall Status:
        <span class="{% if passed %}status-pass{% else %}status-fail{% endif %}">
            {% if passed %}PASSED{% else %}FAILED{% endif %}
        </span>
    </h2>
    <div class="metrics">
        <div class="metric-card">
            <div class="metric-value">{{ "%.0f"|format(metric_results.get('p99_latency_ms', 0)) }}</div>
            <div>p99 Latency (ms)</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{{ "%.2f"|format(metric_results.get('error_rate', 0) * 100) }}%</div>
            <div>Error Rate</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{{ "%.0f"|format(metric_results.get('peak_rps', 0)) }}</div>
            <div>Peak RPS</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{{ metric_results.get('timeout_count', 0) }}</div>
            <div>Timeout Count</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{{ "%.1f"|format(fraud_metric_value) }}%</div>
            <div>{{ fraud_metric_label }}</div>
        </div>
    </div>
    <div class="chart-container">{{ charts.latency|safe }}</div>
    <div class="chart-container">{{ charts.throughput|safe }}</div>
    <div class="chart-container">{{ charts.errors|safe }}</div>
    <div class="chart-container">{{ charts.fraud|safe }}</div>
    {% if failures %}
    <h2>Failures:</h2>
    <ul>
    {% for failure in failures %}
        <li class="status-fail">{{ failure }}</li>
    {% endfor %}
    </ul>
    {% endif %}
</body>
</html>
"""


class ReportGenerator:
    """Generate HTML reports for simulation results."""

    def __init__(self) -> None:
        self.template = Template(TEMPLATE_STR) if Template else None

    def generate_html(
        self,
        scenario: Any,
        results: Dict[str, Any],
        output_path: str,
    ) -> None:
        """Generate HTML report. results is the validation dict from scenario.execute()."""
        metric_results = getattr(scenario, "results", {})
        passed = results.get("passed", False)
        failures = list(results.get("failures", []))
        timeout_count = metric_results.get("timeout_count")
        if isinstance(timeout_count, (int, float)) and timeout_count > 0:
            failures.append(f"timeouts: expected 0, got {int(timeout_count)}")
            passed = False
        fraud_metric_value, fraud_metric_label = _fraud_display(metric_results)
        charts = {
            "latency": _create_latency_chart(metric_results),
            "throughput": _create_throughput_chart(metric_results),
            "errors": _create_error_chart(metric_results),
            "fraud": _create_fraud_chart(metric_results),
        }
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if self.template:
            html = self.template.render(
                scenario_name=scenario.name,
                description=scenario.description,
                timestamp=timestamp,
                passed=passed,
                failures=failures,
                metric_results=metric_results,
                fraud_metric_value=fraud_metric_value,
                fraud_metric_label=fraud_metric_label,
                charts=charts,
            )
        else:
            html = (
                f"<html><body><h1>{scenario.name}</h1>"
                f"<p>Status: {'PASSED' if passed else 'FAILED'}</p>"
                f"<p>Generated: {timestamp}</p></body></html>"
            )
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(html)
