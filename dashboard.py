# dashboard.py – polished professional UI
import os
import json
import sys
import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
from config import METRICS_FILE, NUM_CLIENTS

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
app.title = "TrustFlux – Secure Federated Learning"


CARD_STYLE = {
    "borderRadius": "12px",
    "boxShadow": "0 4px 12px rgba(0,0,0,0.3)",
    "padding": "10px"
}

app.layout = dbc.Container([

    dbc.Row(dbc.Col(
        html.Div([
            html.H2("🛡️ TrustFlux", className="text-center"),
            html.P("Secure Federated Learning Monitoring",
                   className="text-center text-muted")
        ], className="mt-4 mb-3"),
        width=12
    )),

    dbc.Row(dbc.Col(html.Div(id="status-alert", className="text-center mb-3"), width=12)),
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([html.H6("Round"), html.H3(id="kpi-round")])
        ], style={**CARD_STYLE, "backgroundColor": "#12B1BC", "color": "black"}), width=4),

        dbc.Col(dbc.Card([
            dbc.CardBody([html.H6("Accuracy"), html.H3(id="kpi-acc")])
        ], style={**CARD_STYLE, "backgroundColor": "#F45577", "color": "black"}), width=4),

        dbc.Col(dbc.Card([
            dbc.CardBody([html.H6("Selected Clients"), html.H3(id="kpi-clients")])
        ], style={**CARD_STYLE, "backgroundColor": "#9C13D7", "color": "black"}), width=4),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col(
            dbc.Card([
                dbc.CardHeader("Experiment Status"),
                dbc.CardBody(html.Div(id="experiment-info"))
            ],
            style={**CARD_STYLE, "backgroundColor": "#31EE7D", "color": "black"}, className="mb-3"), width=12)
    ]),

    dbc.Tabs([

        dbc.Tab([
            dbc.Row([
                dbc.Col(dbc.Card([
                    dbc.CardHeader("Test Accuracy"),
                    dbc.CardBody(dcc.Graph(id="accuracy-gauge"))
                ], style=CARD_STYLE), width=6),

                dbc.Col(dbc.Card([
                    dbc.CardHeader("Merkle Root"),
                    dbc.CardBody(html.Div(id="merkle-display", className="text-center"))
                ], style=CARD_STYLE), width=6)
            ], className="mb-3"),

            dbc.Row(dbc.Col(dbc.Card([
                dbc.CardHeader("Trust Score Distribution"),
                dbc.CardBody(dcc.Graph(id="trust-heatmap"))
            ], style=CARD_STYLE), width=12)),

            dbc.Row(dbc.Col(dbc.Card([
                dbc.CardHeader("Client Selection"),
                dbc.CardBody(dcc.Graph(id="selected-bar"))
            ], style=CARD_STYLE), width=12))
        ], label="Overview"),

        dbc.Tab(dbc.Card([
            dbc.CardHeader("Trust Scores"),
            dbc.CardBody(dcc.Graph(id="trust-history"))
        ], style=CARD_STYLE), label="Trust Analytics"),

        dbc.Tab(dbc.Card([
            dbc.CardHeader("Audit Trail"),
            dbc.CardBody(dash_table.DataTable(
                id="audit-table",
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'center'},
                style_header={'fontWeight': 'bold'}
            ))
        ], style=CARD_STYLE), label="Audit Logs"),

        dbc.Tab(dbc.Card([
            dbc.CardHeader("Configuration"),
            dbc.CardBody(html.Pre(id="config-display",
                                  style={'background': '#1e1e1e', 'padding': '15px'}))
        ], style=CARD_STYLE), label="Config")

    ]),

    dcc.Interval(id='interval', interval=2000),

    html.Footer("© TrustFlux | Secure Federated Learning System",
                className="text-center text-muted mt-4")

], fluid=True)


@app.callback(
    [Output("status-alert", "children"),
     Output("experiment-info", "children"),
     Output("accuracy-gauge", "figure"),
     Output("merkle-display", "children"),
     Output("trust-heatmap", "figure"),
     Output("selected-bar", "figure"),
     Output("trust-history", "figure"),
     Output("audit-table", "data"),
     Output("audit-table", "columns"),
     Output("config-display", "children"),
     Output("kpi-round", "children"),
     Output("kpi-acc", "children"),
     Output("kpi-clients", "children")],
    [Input("interval", "n_intervals")]
)
def update_dashboard(n):

    try:
        with open(METRICS_FILE, "r") as f:
            data = json.load(f)
    except Exception:
        return ("⏳ Waiting for data...",
                "No data",
                empty_gauge("Waiting"),
                html.H4("⏳"),
                empty_heatmap(),
                empty_bar(),
                empty_line("Waiting"),
                [], [], "No config",
                "-", "-", "-")

    exp_name = data.get("experiment", "unknown")
    rnd = data.get("round", 0)
    acc = data.get("test_accuracy", 0)
    trust = data.get("trust_scores", {})
    selected = data.get("selected", [])
    merkle_root = data.get("merkle_root", "")

    clients = list(range(NUM_CLIENTS))
    trust_vals = [trust.get(str(c), trust.get(c, 0.5)) for c in clients]

    status_text = html.Span(
        f"● Round {rnd} | Accuracy: {acc:.4f}",
        style={
            "color": "#00ffcc" if acc > 0.5 else "#ffcc00",
            "fontWeight": "bold",
            "fontSize": "16px"
        }
    )

    exp_info = html.Div([
        html.P(f"Experiment: {exp_name}"),
        html.P(f"Round: {rnd} / {os.getenv('NUM_ROUNDS', '20')}"),
        html.P(f"Test Accuracy: {acc:.4f}")
    ])

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=acc,
        number={'font': {'size': 45}},
        gauge={
            'axis': {'range': [0, 1]},
            'bar': {'color': "#00d4ff"},
            'steps': [
                {'range': [0, 0.5], 'color': '#ff4d4d'},
                {'range': [0.5, 0.8], 'color': '#ffd11a'},
                {'range': [0.8, 1], 'color': '#00cc66'}
            ]
        }
    ))
    gauge.update_layout(
        height=360,
        margin=dict(l=40, r=40, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "white"}
    )

    merkle_short = (merkle_root[:16] + "...") if len(merkle_root) > 16 else merkle_root
    merkle = html.H4(merkle_short, className="text-info")

    heatmap = go.Figure(data=go.Heatmap(
        z=[trust_vals],
        x=[f"C{c}" for c in clients],
        y=["Trust"],
        colorscale=[
            [0, "#ff4d4d"],
            [0.5, "#ffd11a"],
            [1, "#00cc66"]
        ],
        zmin=0, zmax=1
    ))

    bar = go.Figure(data=go.Bar(
        x=[f"C{c}" for c in clients],
        y=[1 if c in selected else 0 for c in clients],
        marker_color=['#00cc66' if c in selected else '#ff4d4d' for c in clients]
    ))

    trust_hist = go.Figure(data=go.Bar(
        x=[f"C{c}" for c in clients],
        y=trust_vals,
        marker_color="#00d4ff",
        text=[f"{v:.2f}" for v in trust_vals],
        textposition='outside'
    ))

    audit_data = [{
        "Round": rnd,
        "Merkle Root": merkle_root,
        "Selected Clients": ", ".join(map(str, selected)),
        "Accuracy": f"{acc:.4f}"
    }]
    audit_columns = [{"name": k, "id": k} for k in audit_data[0]]

    config_text = json.dumps(
        {k: v for k, v in os.environ.items() if not k.startswith('_')},
        indent=2
    )

    return (status_text, exp_info, gauge, merkle, heatmap, bar,
            trust_hist, audit_data, audit_columns, config_text,
            rnd, f"{acc:.3f}", len(selected))


def empty_gauge(msg):
    return go.Figure(go.Indicator(mode="number", value=0, title={'text': msg}))

def empty_heatmap():
    return go.Figure()

def empty_bar():
    return go.Figure()

def empty_line(msg):
    return go.Figure()


if __name__ == "__main__":
    print("🌐 Dashboard running at http://localhost:8050", flush=True)
    app.run_server(host='0.0.0.0', port=8050, debug=False, use_reloader=False)
