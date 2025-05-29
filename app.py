import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from layouts import create_layout
from callbacks import register_callbacks

# Initialize the app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

app.title = "Options Visualizer"
server = app.server

# Create app layout
app.layout = create_layout(app)

# Register callbacks
register_callbacks(app)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
