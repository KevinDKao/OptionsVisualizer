from dash import Input, Output, State, callback_context, ALL, MATCH, html
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm
import uuid

def register_callbacks(app):
    # Mode switching callback
    @app.callback(
        [
            Output("strategy-mode-btn", "color"),
            Output("leg-builder-mode-btn", "color"),
            Output("strategy-mode-panel", "style"),
            Output("leg-builder-mode-panel", "style"),
            Output("current-mode", "data"),
        ],
        [
            Input("strategy-mode-btn", "n_clicks"),
            Input("leg-builder-mode-btn", "n_clicks"),
        ],
        [State("current-mode", "data")]
    )
    def switch_mode(strategy_clicks, leg_clicks, current_mode):
        ctx = callback_context
        if not ctx.triggered:
            return "primary", "secondary", {}, {"display": "none"}, "strategy"
        
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
        if button_id == "strategy-mode-btn":
            return "primary", "secondary", {}, {"display": "none"}, "strategy"
        else:
            return "secondary", "primary", {"display": "none"}, {}, "leg_builder"

    # Leg management callbacks
    @app.callback(
        Output("legs-data", "data"),
        [
            Input("add-leg-button", "n_clicks"),
            Input("clear-legs-button", "n_clicks"),
            Input({"type": "remove-leg", "index": ALL}, "n_clicks"),
        ],
        [
            State("new-leg-type", "value"),
            State("new-leg-position", "value"),
            State("new-leg-strike", "value"),
            State("new-leg-quantity", "value"),
            State("legs-data", "data"),
        ]
    )
    def manage_legs(add_clicks, clear_clicks, remove_clicks, leg_type, position, strike, quantity, current_legs):
        ctx = callback_context
        if not ctx.triggered:
            return current_legs
        
        button_id = ctx.triggered[0]["prop_id"]
        
        if "add-leg-button" in button_id and add_clicks:
            leg_id = str(uuid.uuid4())
            new_leg = {
                "id": leg_id,
                "type": leg_type,
                "position": position,
                "strike": strike,
                "quantity": quantity,
            }
            return current_legs + [new_leg]
        
        elif "clear-legs-button" in button_id and clear_clicks:
            return []
        
        elif "remove-leg" in button_id:
            # Parse the leg ID from the button that was clicked
            import json
            button_data = json.loads(button_id.split('.')[0])
            leg_id_to_remove = button_data["index"]
            return [leg for leg in current_legs if leg["id"] != leg_id_to_remove]
        
        return current_legs

    # Update legs display
    @app.callback(
        Output("legs-container", "children"),
        [Input("legs-data", "data")]
    )
    def update_legs_display(legs_data):
        if not legs_data:
            return [html.P("No legs added yet. Add some legs above to build your custom option structure.", 
                          style={"textAlign": "center", "color": "#718096", "fontStyle": "italic"})]
        
        legs_components = []
        for i, leg in enumerate(legs_data):
            position_symbol = "+" if leg["position"] == "long" else "-"
            position_color = "#48bb78" if leg["position"] == "long" else "#f56565"
            
            leg_component = html.Div([
                html.Div([
                    html.Span(f"Leg {i+1}: ", style={"fontWeight": "bold"}),
                    html.Span(f"{position_symbol}{leg['quantity']} {leg['type'].title()}", 
                             style={"color": position_color, "fontWeight": "bold"}),
                    html.Span(f" @ ${leg['strike']}", style={"marginLeft": "10px"}),
                    dbc.Button("Remove", 
                              id={"type": "remove-leg", "index": leg["id"]}, 
                              color="danger", size="sm", className="float-end"),
                ], className="leg-header"),
            ], className="leg-container")
            legs_components.append(leg_component)
        
        return legs_components

    # Main calculation callback for both modes
    @app.callback(
        [
            Output("payoff-graph", "figure"),
            Output("delta-graph", "figure"),
            Output("gamma-graph", "figure"),
            Output("theta-graph", "figure"),
            Output("vega-graph", "figure"),
            Output("second-strike-container", "style"),
            Output("option-values-content", "children"),
        ],
        [
            Input("calculate-button", "n_clicks"),
            Input("option-strategy", "value"),
            Input("legs-data", "data"),
            Input("update-legs-button", "n_clicks"),
            Input("current-mode", "data"),
            Input("add-leg-button", "n_clicks"),
            Input("clear-legs-button", "n_clicks"),
        ],
        [
            State("stock-price", "value"),
            State("strike-price", "value"),
            State("volatility", "value"),
            State("days-to-expiration", "value"),
            State("risk-free-rate", "value"),
            State("option-type", "value"),
            State("second-strike", "value"),
            State("leg-stock-price", "value"),
            State("leg-volatility", "value"),
            State("leg-days-to-expiration", "value"),
            State("leg-risk-free-rate", "value"),
        ]
    )
    def update_calculations(calc_clicks, strategy, legs_data, update_legs_clicks, current_mode, add_leg_clicks, clear_leg_clicks,
                          s, k, v, t, r, option_type, k2,
                          leg_s, leg_v, leg_t, leg_r):
        ctx = callback_context
        
        # Determine which mode we're in and use appropriate parameters
        if current_mode == "leg_builder":
            s = leg_s or 100
            v = (leg_v or 30) / 100
            t = (leg_t or 30) / 365
            r = (leg_r or 2) / 100
            k = 100  # Default for single option display
        else:
            s = s or 100
            k = k or 100
            k2 = k2 or 110
            v = (v or 30) / 100
            t = (t or 30) / 365
            r = (r or 2) / 100

        # Calculate basic option metrics for display
        call_price, call_delta, call_gamma, call_theta, call_vega = bs_option_metrics(s, k, t, r, v, "call")
        put_price, put_delta, put_gamma, put_theta, put_vega = bs_option_metrics(s, k, t, r, v, "put")
        
        # Price range for graphs
        price_range = np.linspace(max(0.5 * s, 1), 1.5 * s, 100)
        
        # Generate graphs based on mode
        if current_mode == "leg_builder":
            # Check if we have legs and they are valid
            if legs_data and len(legs_data) > 0:
                try:
                    payoff_fig = generate_legs_payoff_graph(s, legs_data, price_range, t, r, v)
                    delta_fig = generate_legs_greek_graph(s, legs_data, price_range, "delta", t, r, v)
                    gamma_fig = generate_legs_greek_graph(s, legs_data, price_range, "gamma", t, r, v)
                    theta_fig = generate_legs_greek_graph(s, legs_data, price_range, "theta", t, r, v)
                    vega_fig = generate_legs_greek_graph(s, legs_data, price_range, "vega", t, r, v)
                    
                    # Create legs summary for option values
                    option_values_content = create_legs_summary(s, legs_data, t, r, v)
                except Exception as e:
                    # If there's an error, create empty graphs with error message
                    error_msg = f"Error calculating legs: {str(e)}"
                    payoff_fig = create_empty_graph(error_msg)
                    delta_fig = create_empty_graph(error_msg)
                    gamma_fig = create_empty_graph(error_msg)
                    theta_fig = create_empty_graph(error_msg)
                    vega_fig = create_empty_graph(error_msg)
                    option_values_content = html.P(error_msg, style={"color": "red", "textAlign": "center"})
            else:
                # Create empty graphs for leg builder mode when no legs
                payoff_fig = create_empty_graph("Add legs above to see payoff diagram")
                delta_fig = create_empty_graph("Add legs above to see delta")
                gamma_fig = create_empty_graph("Add legs above to see gamma")
                theta_fig = create_empty_graph("Add legs above to see theta")
                vega_fig = create_empty_graph("Add legs above to see vega")
                option_values_content = html.P("Add legs above to see option values", 
                                             style={"textAlign": "center", "color": "#718096", "fontStyle": "italic"})
        else:
            # Strategy mode
            payoff_fig = generate_payoff_graph(s, k, k2, call_price, put_price, price_range, strategy, t, r, v)
            delta_fig = generate_greek_graph(s, k, t, r, v, price_range, "delta", strategy)
            gamma_fig = generate_greek_graph(s, k, t, r, v, price_range, "gamma", strategy)
            theta_fig = generate_greek_graph(s, k, t, r, v, price_range, "theta", strategy)
            vega_fig = generate_greek_graph(s, k, t, r, v, price_range, "vega", strategy)
            
            # Standard option values display with actual values
            option_values_content = create_standard_option_values_with_data(
                call_price, call_delta, call_gamma, call_theta, call_vega,
                put_price, put_delta, put_gamma, put_theta, put_vega
            )
        
        # Show/hide second strike input based on strategy
        show_second_strike = strategy in ["bull_call_spread", "bear_put_spread", "iron_condor", "strangle", "butterfly_spread"]
        second_strike_style = {} if show_second_strike else {"display": "none"}
        
        return [
            payoff_fig,
            delta_fig,
            gamma_fig,
            theta_fig,
            vega_fig,
            second_strike_style,
            option_values_content,
        ]

def bs_option_metrics(s, k, t, r, v, option_type="call"):
    """Calculate Black-Scholes option price and greeks"""
    if t <= 0:
        if option_type == "call":
            price = max(s - k, 0)
            delta = 1 if s > k else 0
        else:
            price = max(k - s, 0)
            delta = -1 if s < k else 0
        return price, delta, 0, 0, 0
    
    d1 = (np.log(s/k) + (r + 0.5 * v**2) * t) / (v * np.sqrt(t))
    d2 = d1 - v * np.sqrt(t)
    
    if option_type == "call":
        price = s * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)
        delta = norm.cdf(d1)
        theta = -((s * norm.pdf(d1) * v) / (2 * np.sqrt(t))) - r * k * np.exp(-r * t) * norm.cdf(d2)
    else:  # put
        price = k * np.exp(-r * t) * norm.cdf(-d2) - s * norm.cdf(-d1)
        delta = -norm.cdf(-d1)
        theta = -((s * norm.pdf(d1) * v) / (2 * np.sqrt(t))) + r * k * np.exp(-r * t) * norm.cdf(-d2)
    
    gamma = norm.pdf(d1) / (s * v * np.sqrt(t))
    vega = s * np.sqrt(t) * norm.pdf(d1) * 0.01  # For 1% change in volatility
    
    # Convert theta to daily
    theta = theta / 365
    
    return price, delta, gamma, theta, vega

def create_empty_graph(message):
    """Create an empty graph with a message"""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        xanchor='center', yanchor='middle',
        showarrow=False,
        font=dict(size=16, color="#718096")
    )
    fig.update_layout(
        template="plotly_white",
        height=300,
        margin=dict(l=50, r=50, t=50, b=50),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )
    return fig

def generate_legs_payoff_graph(s, legs_data, price_range, t, r, v):
    """Generate payoff diagram for custom legs"""
    total_payoff = np.zeros_like(price_range)
    total_cost = 0
    
    fig = go.Figure()
    
    for leg in legs_data:
        # Validate leg data structure
        if not all(key in leg for key in ['strike', 'quantity', 'position', 'type']):
            continue
            
        leg_payoffs = np.zeros_like(price_range)
        strike = float(leg["strike"])
        quantity = int(leg["quantity"])
        multiplier = 1 if leg["position"] == "long" else -1
        
        # Calculate option price for premium
        option_price = bs_option_metrics(s, strike, t, r, v, leg["type"])[0]
        total_cost += multiplier * option_price * quantity
        
        # Calculate payoffs
        for i, price in enumerate(price_range):
            if leg["type"] == "call":
                intrinsic = max(price - strike, 0)
            else:  # put
                intrinsic = max(strike - price, 0)
            
            leg_payoffs[i] = multiplier * quantity * (intrinsic - option_price)
        
        total_payoff += leg_payoffs
        
        # Add individual leg trace
        leg_name = f"{'+' if multiplier > 0 else '-'}{quantity} {leg['type'].title()} ${strike}"
        fig.add_trace(go.Scatter(
            x=price_range, 
            y=leg_payoffs, 
            mode="lines", 
            name=leg_name,
            line=dict(dash="dot", width=1),
            opacity=0.6
        ))
    
    # Add total payoff
    fig.add_trace(go.Scatter(
        x=price_range, 
        y=total_payoff, 
        mode="lines", 
        name="Total Payoff",
        line=dict(width=3, color="#667eea")
    ))
    
    # Add current price marker
    current_payoff_value = total_payoff[np.argmin(np.abs(price_range - s))]
    fig.add_trace(go.Scatter(
        x=[s], 
        y=[current_payoff_value], 
        mode="markers", 
        marker=dict(size=10, color="#f56565"), 
        name="Current Price"
    ))
    
    # Add breakeven points
    breakeven_indices = np.where(np.diff(np.signbit(total_payoff)))[0]
    for idx in breakeven_indices:
        if idx < len(price_range) - 1:
            x1, y1 = price_range[idx], total_payoff[idx]
            x2, y2 = price_range[idx + 1], total_payoff[idx + 1]
            if y1 != y2:
                x_breakeven = x1 + (0 - y1) * (x2 - x1) / (y2 - y1)
                fig.add_trace(go.Scatter(
                    x=[x_breakeven], 
                    y=[0], 
                    mode="markers", 
                    marker=dict(size=8, color="#48bb78"), 
                    name=f"Breakeven: ${x_breakeven:.2f}"
                ))
    
    fig.update_layout(
        title="Custom Option Structure Payoff",
        xaxis=dict(title="Stock Price ($)", gridcolor="#e2e8f0"),
        yaxis=dict(title="Profit/Loss ($)", gridcolor="#e2e8f0"),
        template="plotly_white",
        height=300,
        margin=dict(l=50, r=50, t=50, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff"
    )
    
    # Add zero line
    fig.add_shape(
        type="line", 
        line=dict(dash="dash", color="#a0aec0"), 
        x0=price_range[0], y0=0, 
        x1=price_range[-1], y1=0
    )
    
    return fig

def generate_legs_greek_graph(s, legs_data, price_range, greek_type, t, r, v):
    """Generate greek graph for custom legs"""
    total_greeks = np.zeros_like(price_range)
    
    for leg in legs_data:
        # Validate leg data structure
        if not all(key in leg for key in ['strike', 'quantity', 'position', 'type']):
            continue
            
        strike = float(leg["strike"])
        quantity = int(leg["quantity"])
        multiplier = 1 if leg["position"] == "long" else -1
        
        leg_greeks = np.zeros_like(price_range)
        
        for i, price in enumerate(price_range):
            _, delta, gamma, theta, vega = bs_option_metrics(price, strike, t, r, v, leg["type"])
            
            if greek_type == "delta":
                leg_greeks[i] = delta
            elif greek_type == "gamma":
                leg_greeks[i] = gamma
            elif greek_type == "theta":
                leg_greeks[i] = theta
            else:  # vega
                leg_greeks[i] = vega
        
        total_greeks += multiplier * quantity * leg_greeks
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=price_range, y=total_greeks, mode="lines", name=f"Total {greek_type.title()}"))
    
    # Add current value marker
    current_greek_value = total_greeks[np.argmin(np.abs(price_range - s))]
    fig.add_trace(go.Scatter(
        x=[s], 
        y=[current_greek_value], 
        mode="markers", 
        marker=dict(size=10, color="#f56565"), 
        name=f"Current {greek_type.title()}"
    ))
    
    y_title_map = {
        "delta": "Delta",
        "gamma": "Gamma", 
        "theta": "Theta ($/day)",
        "vega": "Vega ($/1% vol)"
    }
    
    fig.update_layout(
        title=f"{greek_type.title()} vs. Stock Price",
        xaxis=dict(title="Stock Price ($)", gridcolor="#e2e8f0"),
        yaxis=dict(title=y_title_map[greek_type], gridcolor="#e2e8f0"),
        template="plotly_white",
        height=300,
        margin=dict(l=50, r=50, t=50, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff"
    )
    
    # Add zero line
    fig.add_shape(
        type="line", 
        line=dict(dash="dash", color="#a0aec0"), 
        x0=price_range[0], y0=0, 
        x1=price_range[-1], y1=0
    )
    
    return fig

def create_legs_summary(s, legs_data, t, r, v):
    """Create summary of all legs"""
    if not legs_data or len(legs_data) == 0:
        return html.P("No legs to display")
    
    legs_summary = []
    total_cost = 0
    
    for i, leg in enumerate(legs_data):
        # Validate leg data structure
        if not all(key in leg for key in ['strike', 'quantity', 'position', 'type']):
            continue
            
        strike = float(leg["strike"])
        quantity = int(leg["quantity"])
        multiplier = 1 if leg["position"] == "long" else -1
        
        price, delta, gamma, theta, vega = bs_option_metrics(s, strike, t, r, v, leg["type"])
        leg_cost = multiplier * price * quantity
        total_cost += leg_cost
        
        position_color = "#48bb78" if leg["position"] == "long" else "#f56565"
        
        leg_summary = dbc.Card([
            dbc.CardHeader(f"Leg {i+1}: {'+' if multiplier > 0 else '-'}{quantity} {leg['type'].title()} ${strike}"),
            dbc.CardBody([
                html.Table([
                    html.Tr([html.Td("Premium:"), html.Td(f"${price:.4f}")]),
                    html.Tr([html.Td("Total Cost:"), html.Td(f"${leg_cost:.2f}", style={"color": position_color})]),
                    html.Tr([html.Td("Delta:"), html.Td(f"{multiplier * delta * quantity:.4f}")]),
                    html.Tr([html.Td("Gamma:"), html.Td(f"{multiplier * gamma * quantity:.4f}")]),
                    html.Tr([html.Td("Theta:"), html.Td(f"${multiplier * theta * quantity:.4f}/day")]),
                    html.Tr([html.Td("Vega:"), html.Td(f"${multiplier * vega * quantity:.4f}")]),
                ], className="table table-sm"),
            ])
        ], className="mb-3")
        
        legs_summary.append(leg_summary)
    
    # Add total summary
    total_summary = dbc.Card([
        dbc.CardHeader("Total Structure"),
        dbc.CardBody([
            html.Table([
                html.Tr([html.Td("Net Premium:"), html.Td(f"${total_cost:.2f}", style={"color": "#48bb78" if total_cost < 0 else "#f56565", "fontWeight": "bold"})]),
            ], className="table table-sm"),
        ])
    ], color="primary", outline=True)
    
    legs_summary.append(total_summary)
    
    return legs_summary

def create_standard_option_values():
    """Create standard option values display for strategy mode"""
    return dbc.Row([
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
    ])

def create_standard_option_values_with_data(call_price, call_delta, call_gamma, call_theta, call_vega,
                                          put_price, put_delta, put_gamma, put_theta, put_vega):
    """Create standard option values display for strategy mode with actual values"""
    return dbc.Row([
        dbc.Col([
            html.Div([
                html.H5("Call Option", className="text-center"),
                html.Table([
                    html.Tr([html.Td("Price:"), html.Td(f"${call_price:.4f}")]),
                    html.Tr([html.Td("Delta:"), html.Td(f"{call_delta:.4f}")]),
                    html.Tr([html.Td("Gamma:"), html.Td(f"{call_gamma:.4f}")]),
                    html.Tr([html.Td("Theta:"), html.Td(f"${call_theta:.4f}/day")]),
                    html.Tr([html.Td("Vega:"), html.Td(f"${call_vega:.4f}")]),
                ], className="table table-sm"),
            ]),
        ], width=6),
        dbc.Col([
            html.Div([
                html.H5("Put Option", className="text-center"),
                html.Table([
                    html.Tr([html.Td("Price:"), html.Td(f"${put_price:.4f}")]),
                    html.Tr([html.Td("Delta:"), html.Td(f"{put_delta:.4f}")]),
                    html.Tr([html.Td("Gamma:"), html.Td(f"{put_gamma:.4f}")]),
                    html.Tr([html.Td("Theta:"), html.Td(f"${put_theta:.4f}/day")]),
                    html.Tr([html.Td("Vega:"), html.Td(f"${put_vega:.4f}")]),
                ], className="table table-sm"),
            ]),
        ], width=6),
    ])

def generate_payoff_graph(s, k, k2, call_price, put_price, price_range, strategy, t, r, v):
    """Generate payoff diagram based on strategy"""
    payoffs = []
    
    if strategy == "long_call":
        payoffs = np.maximum(price_range - k, 0) - call_price
        title = "Long Call Payoff"
    elif strategy == "long_put":
        payoffs = np.maximum(k - price_range, 0) - put_price
        title = "Long Put Payoff"
    elif strategy == "short_call":
        payoffs = call_price - np.maximum(price_range - k, 0)
        title = "Short Call Payoff"
    elif strategy == "short_put":
        payoffs = put_price - np.maximum(k - price_range, 0)
        title = "Short Put Payoff"
    elif strategy == "covered_call":
        payoffs = price_range - s + call_price - np.maximum(price_range - k, 0)
        title = "Covered Call Payoff"
    elif strategy == "protective_put":
        payoffs = price_range - s - put_price + np.maximum(k - price_range, 0)
        title = "Protective Put Payoff"
    elif strategy == "straddle":
        call_price_k = bs_option_metrics(s, k, t, r, v, "call")[0]
        put_price_k = bs_option_metrics(s, k, t, r, v, "put")[0]
        payoffs = np.maximum(price_range - k, 0) + np.maximum(k - price_range, 0) - call_price_k - put_price_k
        title = "Long Straddle Payoff"
    elif strategy == "strangle":
        k_call = max(k, k2)
        k_put = min(k, k2)
        call_price_k2 = bs_option_metrics(s, k_call, t, r, v, "call")[0]
        put_price_k2 = bs_option_metrics(s, k_put, t, r, v, "put")[0]
        payoffs = np.maximum(price_range - k_call, 0) + np.maximum(k_put - price_range, 0) - call_price_k2 - put_price_k2
        title = "Long Strangle Payoff"
    elif strategy == "bull_call_spread":
        call_price_k2 = bs_option_metrics(s, k2, t, r, v, "call")[0]
        payoffs = np.maximum(price_range - k, 0) - call_price - (np.maximum(price_range - k2, 0) - call_price_k2)
        title = "Bull Call Spread Payoff"
    elif strategy == "bear_put_spread":
        put_price_k2 = bs_option_metrics(s, k2, t, r, v, "put")[0]
        payoffs = np.maximum(k - price_range, 0) - put_price - (np.maximum(k2 - price_range, 0) - put_price_k2)
        title = "Bear Put Spread Payoff"
    elif strategy == "butterfly_spread":
        k_low = min(k, k2)
        k_high = max(k, k2)
        k_mid = (k_low + k_high) / 2
        call_low = bs_option_metrics(s, k_low, t, r, v, "call")[0]
        call_mid = bs_option_metrics(s, k_mid, t, r, v, "call")[0]
        call_high = bs_option_metrics(s, k_high, t, r, v, "call")[0]
        payoffs = (np.maximum(price_range - k_low, 0) - call_low + 
                  np.maximum(price_range - k_high, 0) - call_high - 
                  2 * (np.maximum(price_range - k_mid, 0) - call_mid))
        title = "Butterfly Spread Payoff"
    elif strategy == "iron_condor":
        k_low = min(k, k2) * 0.9
        k_high = max(k, k2) * 1.1
        put_price_low = bs_option_metrics(s, k_low, t, r, v, "put")[0]
        call_price_high = bs_option_metrics(s, k_high, t, r, v, "call")[0]
        
        put_spread = np.maximum(k - price_range, 0) - put_price - (np.maximum(k_low - price_range, 0) - put_price_low)
        call_spread = np.maximum(price_range - k, 0) - call_price - (np.maximum(price_range - k_high, 0) - call_price_high)
        
        payoffs = put_spread + call_spread
        title = "Iron Condor Payoff"
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=price_range, y=payoffs, mode="lines", name="Payoff"))
    fig.add_trace(go.Scatter(x=[s], y=[payoffs[np.argmin(np.abs(price_range - s))]], mode="markers", marker=dict(size=10, color="#f56565"), name="Current Price"))
    
    # Add breakeven points
    breakeven_indices = np.where(np.diff(np.signbit(payoffs)))[0]
    for idx in breakeven_indices:
        if idx < len(price_range) - 1:
            x1, y1 = price_range[idx], payoffs[idx]
            x2, y2 = price_range[idx + 1], payoffs[idx + 1]
            if y1 != y2:
                x_breakeven = x1 + (0 - y1) * (x2 - x1) / (y2 - y1)
                fig.add_trace(go.Scatter(x=[x_breakeven], y=[0], mode="markers", 
                                         marker=dict(size=8, color="#48bb78"), 
                                         name=f"Breakeven: ${x_breakeven:.2f}"))
    
    fig.update_layout(
        title=title,
        xaxis=dict(title="Stock Price ($)", gridcolor="#e2e8f0"),
        yaxis=dict(title="Profit/Loss ($)", gridcolor="#e2e8f0"),
        template="plotly_white",
        height=300,
        margin=dict(l=50, r=50, t=50, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff"
    )
    
    # Add zero line
    fig.add_shape(
        type="line", line=dict(dash="dash", color="#a0aec0"), 
        x0=price_range[0], y0=0, 
        x1=price_range[-1], y1=0
    )
    
    return fig

def generate_greek_graph(s, k, t, r, v, price_range, greek_type, strategy):
    """Generate graph for the selected greek"""
    call_greeks = np.zeros_like(price_range)
    put_greeks = np.zeros_like(price_range)
    
    for i, price in enumerate(price_range):
        _, call_delta, call_gamma, call_theta, call_vega = bs_option_metrics(price, k, t, r, v, "call")
        _, put_delta, put_gamma, put_theta, put_vega = bs_option_metrics(price, k, t, r, v, "put")
        
        if greek_type == "delta":
            call_greeks[i] = call_delta
            put_greeks[i] = put_delta
            title = "Delta vs. Stock Price"
            y_title = "Delta"
        elif greek_type == "gamma":
            call_greeks[i] = call_gamma
            put_greeks[i] = put_gamma
            title = "Gamma vs. Stock Price"
            y_title = "Gamma"
        elif greek_type == "theta":
            call_greeks[i] = call_theta
            put_greeks[i] = put_theta
            title = "Theta vs. Stock Price"
            y_title = "Theta ($/day)"
        else:  # vega
            call_greeks[i] = call_vega
            put_greeks[i] = put_vega
            title = "Vega vs. Stock Price"
            y_title = "Vega ($/1% vol)"
    
    # Calculate combined greeks based on strategy
    if strategy == "long_call":
        combined_greeks = call_greeks
    elif strategy == "long_put":
        combined_greeks = put_greeks
    elif strategy == "short_call":
        combined_greeks = -call_greeks
    elif strategy == "short_put":
        combined_greeks = -put_greeks
    elif strategy == "covered_call":
        if greek_type == "delta":
            combined_greeks = 1 - call_greeks
        else:
            combined_greeks = -call_greeks
    elif strategy == "protective_put":
        if greek_type == "delta":
            combined_greeks = 1 + put_greeks
        else:
            combined_greeks = put_greeks
    elif strategy == "straddle":
        combined_greeks = call_greeks + put_greeks
    elif strategy in ["bull_call_spread", "bear_put_spread", "iron_condor", "strangle", "butterfly_spread"]:
        combined_greeks = call_greeks if strategy == "bull_call_spread" else put_greeks
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=price_range, y=combined_greeks, mode="lines", name=f"{strategy.replace('_', ' ').title()}"))
    fig.add_trace(go.Scatter(x=[s], y=[combined_greeks[np.argmin(np.abs(price_range - s))]], 
                             mode="markers", marker=dict(size=10, color="#f56565"), 
                             name=f"Current {greek_type.title()}"))
    
    fig.update_layout(
        title=title,
        xaxis=dict(title="Stock Price ($)", gridcolor="#e2e8f0"),
        yaxis=dict(title=y_title, gridcolor="#e2e8f0"),
        template="plotly_white",
        height=300,
        margin=dict(l=50, r=50, t=50, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff"
    )
    
    # Add zero line
    fig.add_shape(
        type="line", line=dict(dash="dash", color="#a0aec0"), 
        x0=price_range[0], y0=0, 
        x1=price_range[-1], y1=0
    )
    
    return fig
