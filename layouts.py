from dash import dcc, html
import dash_bootstrap_components as dbc

def create_layout(app):
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("Options Visualizer", className="header-title"),
                html.P("Visualize option payoffs and greeks for different strategies", className="header-description"),
            ], width=12)
        ], className="header mb-4"),
        
        # Mode Toggle
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Mode Selection"),
                    dbc.CardBody([
                        dbc.ButtonGroup([
                            dbc.Button("Strategy Mode", id="strategy-mode-btn", color="primary", active=True),
                            dbc.Button("Leg Builder Mode", id="leg-builder-mode-btn", color="secondary"),
                        ], className="d-grid gap-2 d-md-flex justify-content-md-center"),
                        html.Hr(),
                        html.Div([
                            dbc.Alert([
                                html.Strong("Strategy Mode: "), 
                                "Use predefined option strategies like Long Call, Bull Call Spread, Iron Condor, etc."
                            ], color="info", dismissable=False, className="mb-2"),
                            dbc.Alert([
                                html.Strong("Leg Builder Mode: "), 
                                "Build custom option structures by adding individual call and put legs."
                            ], color="info", dismissable=False, className="mb-0"),
                        ], id="mode-description"),
                    ], className="mode-toggle"),
                ]),
            ], width=12),
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                # Strategy Mode Panel
                html.Div([
                    dbc.Card([
                        dbc.CardHeader("Option Parameters"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Stock Price ($)"),
                                    dbc.Input(id="stock-price", type="number", value=100, min=1, step=1),
                                ], width=6),
                                dbc.Col([
                                    html.Label("Strike Price ($)"),
                                    dbc.Input(id="strike-price", type="number", value=100, min=1, step=1),
                                ], width=6),
                            ], className="mb-3"),
                            
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Volatility (%)"),
                                    dbc.Input(id="volatility", type="number", value=30, min=1, max=200, step=1),
                                ], width=6),
                                dbc.Col([
                                    html.Label("Days to Expiration"),
                                    dbc.Input(id="days-to-expiration", type="number", value=30, min=1, max=1000, step=1),
                                ], width=6),
                            ], className="mb-3"),
                            
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Risk-Free Rate (%)"),
                                    dbc.Input(id="risk-free-rate", type="number", value=2, min=0, max=20, step=0.1),
                                ], width=6),
                                dbc.Col([
                                    html.Label("Option Type"),
                                    dbc.Select(
                                        id="option-type",
                                        options=[
                                            {"label": "Call", "value": "call"},
                                            {"label": "Put", "value": "put"},
                                        ],
                                        value="call",
                                    ),
                                ], width=6),
                            ], className="mb-3"),
                            
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Strategy"),
                                    dbc.Select(
                                        id="option-strategy",
                                        options=[
                                            {"label": "Long Call", "value": "long_call"},
                                            {"label": "Long Put", "value": "long_put"},
                                            {"label": "Short Call", "value": "short_call"},
                                            {"label": "Short Put", "value": "short_put"},
                                            {"label": "Covered Call", "value": "covered_call"},
                                            {"label": "Protective Put", "value": "protective_put"},
                                            {"label": "Bull Call Spread", "value": "bull_call_spread"},
                                            {"label": "Bear Put Spread", "value": "bear_put_spread"},
                                            {"label": "Iron Condor", "value": "iron_condor"},
                                            {"label": "Straddle", "value": "straddle"},
                                            {"label": "Strangle", "value": "strangle"},
                                            {"label": "Butterfly Spread", "value": "butterfly_spread"},
                                        ],
                                        value="long_call",
                                    ),
                                ], width=12),
                            ], className="mb-3"),
                            
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Second Strike Price ($) (for spreads)"),
                                    dbc.Input(id="second-strike", type="number", value=110, min=1, step=1),
                                ], width=6, id="second-strike-container"),
                            ], className="mb-3"),
                            
                            dbc.Row([
                                dbc.Col([
                                    dbc.Button("Calculate", id="calculate-button", color="primary", className="w-100"),
                                ], width=12),
                            ]),
                        ]),
                    ], className="mb-4"),
                ], id="strategy-mode-panel"),
                
                # Leg Builder Mode Panel
                html.Div([
                    dbc.Card([
                        dbc.CardHeader("Global Parameters"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Stock Price ($)"),
                                    dbc.Input(id="leg-stock-price", type="number", value=100, min=1, step=1),
                                ], width=4),
                                dbc.Col([
                                    html.Label("Volatility (%)"),
                                    dbc.Input(id="leg-volatility", type="number", value=30, min=1, max=200, step=1),
                                ], width=4),
                                dbc.Col([
                                    html.Label("Days to Expiration"),
                                    dbc.Input(id="leg-days-to-expiration", type="number", value=30, min=1, max=1000, step=1),
                                ], width=4),
                            ], className="mb-3"),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Risk-Free Rate (%)"),
                                    dbc.Input(id="leg-risk-free-rate", type="number", value=2, min=0, max=20, step=0.1),
                                ], width=4),
                                dbc.Col([
                                    dbc.Button("Update All Legs", id="update-legs-button", color="secondary", className="mt-4"),
                                ], width=4),
                                dbc.Col([
                                    dbc.Button("Clear All Legs", id="clear-legs-button", color="danger", className="mt-4"),
                                ], width=4),
                            ]),
                        ]),
                    ], className="mb-4"),
                    
                    dbc.Card([
                        dbc.CardHeader("Add New Leg"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Option Type"),
                                    dbc.Select(
                                        id="new-leg-type",
                                        options=[
                                            {"label": "Call", "value": "call"},
                                            {"label": "Put", "value": "put"},
                                        ],
                                        value="call",
                                    ),
                                ], width=6),
                                dbc.Col([
                                    html.Label("Position"),
                                    dbc.Select(
                                        id="new-leg-position",
                                        options=[
                                            {"label": "Long (+)", "value": "long"},
                                            {"label": "Short (-)", "value": "short"},
                                        ],
                                        value="long",
                                    ),
                                ], width=6),
                            ], className="mb-3"),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Strike Price ($)"),
                                    dbc.Input(id="new-leg-strike", type="number", value=100, min=1, step=1),
                                ], width=6),
                                dbc.Col([
                                    html.Label("Quantity"),
                                    dbc.Input(id="new-leg-quantity", type="number", value=1, min=1, step=1),
                                ], width=6),
                            ], className="mb-3"),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Button("Add Leg", id="add-leg-button", color="success", className="w-100"),
                                ], width=12),
                            ]),
                        ]),
                    ], className="mb-4"),
                    
                    dbc.Card([
                        dbc.CardHeader("Current Legs"),
                        dbc.CardBody([
                            html.Div(id="legs-container", children=[
                                html.P("No legs added yet. Add some legs above to build your custom option structure.", 
                                       style={"textAlign": "center", "color": "#718096", "fontStyle": "italic"})
                            ]),
                        ]),
                    ], className="mb-4"),
                ], id="leg-builder-mode-panel", style={"display": "none"}),
            ], width=4),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Payoff Diagram"),
                    dbc.CardBody([
                        dcc.Loading(
                            dcc.Graph(id="payoff-graph", config={"displayModeBar": False}),
                            type="circle",
                        ),
                    ]),
                ], className="mb-4"),
                
                dbc.Card([
                    dbc.CardHeader("Greeks"),
                    dbc.CardBody([
                        dbc.Tabs([
                            dbc.Tab([
                                dcc.Loading(
                                    dcc.Graph(id="delta-graph", config={"displayModeBar": False}),
                                    type="circle",
                                ),
                            ], label="Delta"),
                            dbc.Tab([
                                dcc.Loading(
                                    dcc.Graph(id="gamma-graph", config={"displayModeBar": False}),
                                    type="circle",
                                ),
                            ], label="Gamma"),
                            dbc.Tab([
                                dcc.Loading(
                                    dcc.Graph(id="theta-graph", config={"displayModeBar": False}),
                                    type="circle",
                                ),
                            ], label="Theta"),
                            dbc.Tab([
                                dcc.Loading(
                                    dcc.Graph(id="vega-graph", config={"displayModeBar": False}),
                                    type="circle",
                                ),
                            ], label="Vega"),
                        ]),
                    ]),
                ]),
            ], width=8),
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Option Values"),
                    dbc.CardBody([
                        html.Div(id="option-values-content", children=[
                            dbc.Row([
                                dbc.Col([
                                    html.Div([
                                        html.H5("Call Option", className="text-center"),
                                        html.Table([
                                            html.Tr([html.Td("Price:"), html.Td(id="call-price")]),
                                            html.Tr([html.Td("Delta:"), html.Td(id="call-delta")]),
                                            html.Tr([html.Td("Gamma:"), html.Td(id="call-gamma")]),
                                            html.Tr([html.Td("Theta:"), html.Td(id="call-theta")]),
                                            html.Tr([html.Td("Vega:"), html.Td(id="call-vega")]),
                                        ], className="table table-sm"),
                                    ]),
                                ], width=6),
                                dbc.Col([
                                    html.Div([
                                        html.H5("Put Option", className="text-center"),
                                        html.Table([
                                            html.Tr([html.Td("Price:"), html.Td(id="put-price")]),
                                            html.Tr([html.Td("Delta:"), html.Td(id="put-delta")]),
                                            html.Tr([html.Td("Gamma:"), html.Td(id="put-gamma")]),
                                            html.Tr([html.Td("Theta:"), html.Td(id="put-theta")]),
                                            html.Tr([html.Td("Vega:"), html.Td(id="put-vega")]),
                                        ], className="table table-sm"),
                                    ]),
                                ], width=6),
                            ]),
                        ])
                    ]),
                ]),
            ], width=12),
        ], className="mb-4"),
        
        # Store for leg data
        dcc.Store(id="legs-data", data=[]),
        dcc.Store(id="current-mode", data="strategy"),
        
        html.Footer([
            html.P("© 2025 Options Visualizer", className="text-center"),
        ], className="footer"),
    ], fluid=True)
