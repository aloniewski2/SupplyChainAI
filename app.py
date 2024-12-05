from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objs as go
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import requests
import json

app = Dash(__name__)

# Initialize ML model
scaler = MinMaxScaler()
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Generate initial sample data
def generate_initial_data():
    dates = pd.date_range(start='2024-01-01', end=datetime.now(), freq='D')
    data = {
        'date': dates,
        'demand': np.random.normal(100, 15, len(dates)),
        'inventory': np.random.normal(500, 50, len(dates)),
        'shipping_cost': np.random.normal(1000, 200, len(dates))
    }
    return pd.DataFrame(data)

df = generate_initial_data()

app.layout = html.Div([
    html.H1("Supply Chain Real-Time Dashboard", style={'textAlign': 'center'}),
    
    dcc.Interval(
        id='interval-component',
        interval=5*1000,  # Update every 5 seconds
        n_intervals=0
    ),
    
    html.Div([
        html.Div([
            dcc.Graph(id='demand-forecast'),
            dcc.Graph(id='inventory-levels'),
        ], style={'width': '49%', 'display': 'inline-block'}),
        
        html.Div([
            dcc.Graph(id='shipping-costs'),
            dcc.Graph(id='anomaly-detection'),
        ], style={'width': '49%', 'display': 'inline-block'}),
    ]),
    
    html.Div([
        html.H3("Real-Time Alerts", style={'textAlign': 'center'}),
        html.Div(id='alerts-section', style={'margin': '20px'})
    ])
])

@app.callback(
    [Output('demand-forecast', 'figure'),
     Output('inventory-levels', 'figure'),
     Output('shipping-costs', 'figure'),
     Output('anomaly-detection', 'figure'),
     Output('alerts-section', 'children')],
    Input('interval-component', 'n_intervals')
)
def update_graphs(n):
    global df
    
    # Fetch new data from API
    try:
        metrics_response = requests.get('http://127.0.0.1:8000/supply-chain/metrics')
        metrics_data = metrics_response.json()
        
        # Add new data point
        new_data = pd.DataFrame([{
            'date': datetime.now(),
            'demand': float(metrics_data['data']['metrics']['inventory_metrics']['stock_turnover_rate']),
            'inventory': np.random.normal(500, 50),
            'shipping_cost': float(metrics_data['data']['metrics']['cost_metrics']['average_cost_per_shipment'].replace('$', ''))
        }])
        
        df = pd.concat([df, new_data], ignore_index=True)
    except:
        # If API call fails, simulate new data
        new_data = pd.DataFrame([{
            'date': datetime.now(),
            'demand': np.random.normal(100, 15),
            'inventory': np.random.normal(500, 50),
            'shipping_cost': np.random.normal(1000, 200)
        }])
        df = pd.concat([df, new_data], ignore_index=True)
    
    # Prepare data for ML
    X = df[['demand', 'inventory']].values
    y = df['shipping_cost'].values
    X_scaled = scaler.fit_transform(X)
    
    # Train model and make predictions
    model.fit(X_scaled, y)
    predictions = model.predict(X_scaled)
    
    # Create demand forecast figure
    demand_fig = px.line(df.tail(30), x='date', y='demand', 
                        title='Demand Trend and Forecast')
    
    # Add forecast
    future_dates = pd.date_range(start=df['date'].max(), 
                               periods=7, freq='D')
    future_demand = model.predict(X_scaled[-7:])
    demand_fig.add_scatter(x=future_dates, y=future_demand,
                          mode='lines', name='Forecast',
                          line=dict(dash='dash'))
    
    # Create inventory levels figure
    inventory_fig = go.Figure()
    inventory_fig.add_trace(go.Scatter(x=df['date'], y=df['inventory'],
                                     mode='lines', name='Current Inventory'))
    inventory_fig.add_hline(y=300, line_dash="dash", line_color="red",
                          annotation_text="Reorder Point")
    inventory_fig.update_layout(title='Inventory Levels')
    
    # Create shipping costs figure
    shipping_fig = px.line(df.tail(30), x='date', y='shipping_cost',
                          title='Shipping Costs Trend')
    
    # Create anomaly detection figure
    residuals = y - predictions
    threshold = 2 * np.std(residuals)
    anomalies = df[abs(residuals) > threshold]
    
    anomaly_fig = go.Figure()
    anomaly_fig.add_trace(go.Scatter(x=df['date'], y=df['shipping_cost'],
                                   mode='lines', name='Actual'))
    anomaly_fig.add_trace(go.Scatter(x=anomalies['date'], 
                                   y=anomalies['shipping_cost'],
                                   mode='markers', name='Anomalies',
                                   marker=dict(color='red', size=10)))
    anomaly_fig.update_layout(title='Anomaly Detection')
    
    # Generate alerts
    alerts = []
    if df['inventory'].iloc[-1] < 300:
        alerts.append(html.Div("⚠️ Low inventory alert!", 
                             style={'color': 'red'}))
    if len(anomalies) > 0:
        alerts.append(html.Div("⚠️ Shipping cost anomalies detected!", 
                             style={'color': 'orange'}))
    if df['demand'].iloc[-1] > df['demand'].mean() + df['demand'].std():
        alerts.append(html.Div("📈 Unusual high demand detected!", 
                             style={'color': 'blue'}))
    
    return demand_fig, inventory_fig, shipping_fig, anomaly_fig, alerts

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)