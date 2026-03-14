import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc
from dash import Dash, html, dcc, Input, Output, State
import dash_ag_grid as dag

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

# --- Data Preparation ---
df = pd.read_csv("https://raw.githubusercontent.com/SagarSharma4244/Fetal-Health/main/fetal_health.csv")

def change_value(x):
    mapping = {1.0: "Normal", 2.0: "Suspect", 3.0: "Pathologic"}
    return mapping.get(x, x)

df["fetal_health_label"] = df["fetal_health"].apply(change_value)
X = df.drop(columns=["fetal_health", "fetal_health_label"])
y = df["fetal_health_label"]

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# --- App Setup ---
app = Dash(__name__, external_stylesheets=[dbc.themes.COSMO])

server = app.server

app.layout = dbc.Container([
    html.H2("Fetal Health: Interactive ML Dashboard", className="mt-4 mb-4 text-center"),
    
    dbc.Row([
        # Sidebar Controls
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Model Parameters"),
                dbc.CardBody([
                    dbc.Label("Solver Algorithm:"),
                    dcc.Dropdown(
                        id='solver-selector',
                        options=[{'label': i, 'value': i} for i in ['lbfgs', 'sag', 'saga', 'liblinear', 'newton-cg']],
                        value='lbfgs',
                        clearable=False,
                        className="mb-3"
                    ),
                    dbc.Label("Max Iterations:"),
                    dcc.Input(
                        id='iter-input',
                        type='number',
                        value=200,
                        min=10,
                        max=5000,
                        className="form-control mb-4"
                    ),
                    dbc.Button("Apply & Train", id="run-button", color="primary", n_clicks=0, className="w-100")
                ])
            ], className="shadow-sm")
        ], width=3),

        # Main Display with Spinner
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    # The Spinner wraps the components that wait for the callback
                    dbc.Spinner(
                        children=[
                            html.Div(id='accuracy-display', className="h3 mb-3 text-success"),
                            dcc.Graph(id='importance-graph')
                        ],
                        size="lg",
                        color="primary",
                        type="border",
                        fullscreen=False # Set to True if you want it to cover the whole screen
                    )
                ])
            ], className="shadow-sm")
        ], width=9)
    ], className="mb-5"),

    html.H4("Dataset Explorer"),
    dag.AgGrid(
        id="data-grid",
        rowData=df.head(100).to_dict('records'),
        columnDefs=[{"field": i} for i in df.columns],
        dashGridOptions={"pagination": True},
        style={"height": "400px"}
    ),
], fluid=True)

# --- Callback ---
@app.callback(
    Output('accuracy-display', 'children'),
    Output('importance-graph', 'figure'),
    Input('run-button', 'n_clicks'),
    State('solver-selector', 'value'),
    State('iter-input', 'value')
)
def update_model(n_clicks, solver, max_iter):
    max_iter = max_iter if max_iter else 200

    # Train Model
    model = LogisticRegression(multi_class="multinomial", solver=solver, max_iter=max_iter, random_state=42)
    model.fit(X_train, y_train)

    # Accuracy
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Permutation Importance (n_repeats set to 10 for balance of speed and accuracy)
    perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    perm_df = pd.DataFrame({
        'Feature': X_test.columns,
        'Importance': perm_importance.importances_mean
    }).sort_values(by='Importance', ascending=True)

    fig = px.bar(
        perm_df, x='Importance', y='Feature', orientation='h',
        title="Predictor Impact (Permutation Importance)",
        template="plotly_white",
        color='Importance',
        # color_continuous_scale='Viridis'
    )
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20), coloraxis_showscale=False)

    return f"Model Accuracy: {acc:.2%}", fig

if __name__ == '__main__':
    app.run(debug=True)


