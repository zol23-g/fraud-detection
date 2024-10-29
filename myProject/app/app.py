import os
import pandas as pd
from flask import Flask, jsonify
from dash import Dash, dcc, html
import dash.dependencies
import plotly.express as px
import requests
import logging
from io import StringIO
import json

# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize Flask application
app = Flask(__name__)

# Load fraud data from CSV file
fraud_data_path = '../data/processed/fraud_data.csv'
fraud_data = pd.read_csv(fraud_data_path)

# Flask endpoint to serve summary statistics and fraud trends
@app.route('/api/fraud_summary', methods=['GET'])
def fraud_summary():
    try:
        total_transactions = len(fraud_data)
        total_fraud_cases = fraud_data['class'].sum()
        fraud_percentage = (total_fraud_cases / total_transactions) * 100
        
        summary = {
            'total_transactions': int(total_transactions),
            'total_fraud_cases': int(total_fraud_cases),
            'fraud_percentage': float(fraud_percentage)
        }
        
        return jsonify(summary)
    except Exception as e:
        logging.error(f"Error in /api/fraud_summary: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/fraud_trends', methods=['GET'])
def fraud_trends():
    try:
        fraud_trends = fraud_data.groupby('purchase_time')['class'].sum().reset_index()
        return fraud_trends.to_json(orient='records')
    except Exception as e:
        logging.error(f"Error in /api/fraud_trends: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/fraud_geography', methods=['GET'])
def fraud_geography():
    try:
        fraud_geography = fraud_data.groupby('country')['class'].sum().reset_index()
        return fraud_geography.to_json(orient='records')
    except Exception as e:
        logging.error(f"Error in /api/fraud_geography: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/fraud_devices_browsers', methods=['GET'])
def fraud_devices_browsers():
    try:
        fraud_devices = fraud_data.groupby('device_id')['class'].sum().reset_index()
        fraud_browsers = fraud_data.groupby('browser')['class'].sum().reset_index()
        
        return jsonify({
            'devices': json.loads(fraud_devices.to_json(orient='records')),
            'browsers': json.loads(fraud_browsers.to_json(orient='records'))
        })
    except Exception as e:
        logging.error(f"Error in /api/fraud_devices_browsers: {e}")
        return jsonify({'error': str(e)}), 500

# Initialize Dash application
dash_app = Dash(__name__, server=app, url_base_pathname='/dashboard/')
dash_app.config.suppress_callback_exceptions = True

# Custom CSS styles for the dashboard
styles = {
    'container': {
        'fontFamily': 'Arial, sans-serif',
        'padding': '20px',
        'backgroundColor': '#f9f9f9'
    },
    'header': {
        'textAlign': 'center',
        'padding': '10px',
        'backgroundColor': '#4CAF50',
        'color': 'white'
    },
    'summaryBox': {
        'display': 'inline-block',
        'width': '30%',
        'padding': '10px',
        'margin': '10px',
        'borderRadius': '5px',
        'backgroundColor': '#ffffff',
        'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.2)',
        'textAlign': 'center'
    },
    'chartContainer': {
        'paddingTop': '20px'
    }
}

# Dash layout
dash_app.layout = html.Div(style=styles['container'], children=[
    html.H1("Fraud Detection Dashboard", style=styles['header']),
    
    # Summary boxes
    html.Div([
        html.Div(id='total-transactions', style=styles['summaryBox']),
        html.Div(id='total-fraud-cases', style=styles['summaryBox']),
        html.Div(id='fraud-percentage', style=styles['summaryBox'])
    ], className='summary-container'),
    
    # Interval component for periodic updates
    dcc.Interval(
        id='interval-component',
        interval=60*1000,  # in milliseconds (1 minute)
        n_intervals=0
    ),
    
    # Line chart for fraud trends over time
    html.Div(dcc.Graph(id='fraud-trends-line-chart'), style=styles['chartContainer']),
    
    # Bar chart for fraud geography
    html.Div(dcc.Graph(id='fraud-geography-bar-chart'), style=styles['chartContainer']),
    
    # Bar chart for fraud cases across devices and browsers
    html.Div(dcc.Graph(id='fraud-devices-bar-chart'), style=styles['chartContainer']),
    html.Div(dcc.Graph(id='fraud-browsers-bar-chart'), style=styles['chartContainer'])
])

# Dash callbacks to update the visualizations
@dash_app.callback(
    [dash.dependencies.Output('total-transactions', 'children'),
     dash.dependencies.Output('total-fraud-cases', 'children'),
     dash.dependencies.Output('fraud-percentage', 'children')],
    [dash.dependencies.Input('interval-component', 'n_intervals')]
)
def update_summary_boxes(n):
    try:
        response = requests.get('http://localhost:5000/api/fraud_summary').json()
        
        total_transactions = f"Total Transactions: {response['total_transactions']}"
        total_fraud_cases = f"Total Fraud Cases: {response['total_fraud_cases']}"
        fraud_percentage = f"Fraud Percentage: {response['fraud_percentage']:.2f}%"
        
        return total_transactions, total_fraud_cases, fraud_percentage
    except Exception as e:
        logging.error(f"Error in update_summary_boxes: {e}")
        return "Error", "Error", "Error"

@dash_app.callback(
    dash.dependencies.Output('fraud-trends-line-chart', 'figure'),
    [dash.dependencies.Input('interval-component', 'n_intervals')]
)
def update_fraud_trends_line_chart(n):
    try:
        response = requests.get('http://localhost:5000/api/fraud_trends').json()
        df = pd.read_json(StringIO(json.dumps(response)))
        
        fig = px.line(df, x='purchase_time', y='class', title='Fraud Cases Over Time')
        
        fig.update_layout(
            plot_bgcolor='#f9f9f9',
            paper_bgcolor='#f9f9f9',
            font=dict(color='#333333')
        )
        
        return fig
    except Exception as e:
        logging.error(f"Error in update_fraud_trends_line_chart: {e}")
        return {}

@dash_app.callback(
    dash.dependencies.Output('fraud-geography-bar-chart', 'figure'),
    [dash.dependencies.Input('interval-component', 'n_intervals')]
)
def update_fraud_geography_bar_chart(n):
    try:
        response = requests.get('http://localhost:5000/api/fraud_geography').json()
        df = pd.read_json(StringIO(json.dumps(response)))
        
        fig = px.bar(df, x='country', y='class', title='Fraud Cases by Geography')
        
        fig.update_layout(
            plot_bgcolor='#f9f9f9',
            paper_bgcolor='#f9f9f9',
            font=dict(color='#333333')
        )
        
        return fig
    except Exception as e:
        logging.error(f"Error in update_fraud_geography_bar_chart: {e}")
        return {}

@dash_app.callback(
    dash.dependencies.Output('fraud-devices-bar-chart', 'figure'),
    [dash.dependencies.Input('interval-component', 'n_intervals')]
)
def update_fraud_devices_bar_chart(n):
    try:
        response = requests.get('http://localhost:5000/api/fraud_devices_browsers').json()
        df_devices = pd.read_json(StringIO(json.dumps(response['devices'])))
        
        fig = px.bar(df_devices, x='device_id', y='class', title='Fraud Cases by Device')
        
        fig.update_layout(
            plot_bgcolor='#f9f9f9',
            paper_bgcolor='#f9f9f9',
            font=dict(color='#333333')
        )
        
        return fig
    except Exception as e:
        logging.error(f"Error in update_fraud_devices_bar_chart: {e}")
        return {}

@dash_app.callback(
    dash.dependencies.Output('fraud-browsers-bar-chart', 'figure'),
    [dash.dependencies.Input('interval-component', 'n_intervals')]
)
def update_fraud_browsers_bar_chart(n):
    try:
        response = requests.get('http://localhost:5000/api/fraud_devices_browsers').json()
        df_browsers = pd.read_json(StringIO(json.dumps(response['browsers'])))
        
        fig = px.bar(df_browsers, x='browser', y='class', title='Fraud Cases by Browser')
        
        fig.update_layout(
            plot_bgcolor='#f9f9f9',
            paper_bgcolor='#f9f9f9',
            font=dict(color='#333333')
        )
        
        return fig
    except Exception as e:
        logging.error(f"Error in update_fraud_browsers_bar_chart: {e}")
        return {}
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
