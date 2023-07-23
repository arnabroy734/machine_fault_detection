from dash import Dash, dcc, html, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime
from data_synthetic.generate_data import HealthyData, FaultyData
import dash_daq as daq
import dash_bootstrap_components as dbc
from path.path import CLUSTERING_MODEL, ISOLATION_MODEL
import pickle
import warnings
warnings.filterwarnings("ignore")
from log.logging import LOGGER
import logging


try:
    healthy_data = HealthyData()
    faulty_data = FaultyData()

    with open(CLUSTERING_MODEL, 'rb') as f:
        clustering_model = pickle.load(f)
        f.close()
        
    with open(ISOLATION_MODEL, 'rb') as f:
        isolation_model = pickle.load(f)
        f.close()
    
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP,  dbc.icons.FONT_AWESOME])
    app.title = "Predict equipment health"

    LOGGER.log_startup("Application initiated successfully", level=logging.INFO)

except Exception as e:
    app = None
    LOGGER.log_startup(f"Error in application initiation - {e}", level=logging.ERROR)




INTERVAL = 1000
DATA_WINDOW = 2000


# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}


# Sidebar
sidebar = html.Div(
    [
        html.H3("Predictive Maintenance"),
        html.Hr(),
        html.P(
            '''This app demonstrates how machine learning can be used in predictive maintenance.
              Here synthetic time series data of equipment parameters are being generated and continuous monitoring is being done to predict equipment health.
              '''),
        dbc.Nav(
            [
                # dbc.NavLink("Monitor Machine Health", href="/", active="exact"),
                # dbc.NavLink("Page 1", href="/page-1", active="exact"),
                # dbc.NavLink("Page 2", href="/page-2", active="exact"),
                
                dbc.Button("Simulate Fault", color="primary", className="me-1", id='button-simulation', n_clicks=2, style={'margin-top':'20px', 'margin-bottom':'20px'}),

                html.H6("Select method of prediction"), 

                dcc.Dropdown(id="model_used", options=[
                    {'label':"Clustering Model", "value":"clustering"},
                    {"label":"Isolation Forest Model", "value": "isolation"}
                ], value='clustering'),
                
                html.A([
                    html.I(className="fa-brands fa-github fa-xl", style={'margin-right':"5px"}),
                    "Github Link"
                ], href='https://github.com/arnabroy734/machine_fault_detection', style={'margin-top':'20px', 'text-align':'center', 'color' : '#1f2328'}),

                html.A([
                    html.I(className="fa-solid fa-database fa-lg", style={'margin-right':"5px"}),
                    "Dataset Used"
                ], href='https://archive.ics.uci.edu/dataset/791/metropt+3+dataset', style={'margin-top':'20px', 'text-align':'center', 'color' : '#1f2328'}),
                
            
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)


# Content
content = html.Div(
    # style={'width':'60%', 'margin-left':'auto', 'margin-right':'auto', 'margin-top': '50px'},
    style=CONTENT_STYLE,
    children= [

        # Heading    
        html.H5(children=["Select the machine parameter"]),

        # Dropdown Menu
        dcc.Dropdown(HealthyData().get_column_names(), id="feature-select", value="TP2"),

        # Show graph
        dcc.Graph(id='machine-trend'), 

        # Interval to rigger update function
        dcc.Interval(id='interval', interval=INTERVAL),

        # Data Store
        dcc.Store(id='latest-healthy-data-index', data=-1),
        dcc.Store(id='latest-fault-data-index', data=-1),
        dcc.Store(id='data-queue'),
        dcc.Store(id="prediction", data=-1),

        # Fault simulation button
        # html.Div(
        #     [dbc.Button("Simulate Fault", color="primary", className="me-1", id='button-simulation', n_clicks=2)]
        # ),

        # Prediction section
        html.H5 (["Machine Health Prediction"], style={'margin-top' : "50px"}),

        dbc.Alert("Machine is running healthy", id='result', color='info')



    ]
)


# Simulate button callback
@app.callback(
        Output("button-simulation", "children"),
        Output("button-simulation", "color"),
        Input("button-simulation", "n_clicks")
)
def simulate_fault(n_clicks):
    if (n_clicks % 2 != 0):
        return "Fault Being Simulated. Press Again to Stop", "danger"
    else:
        return "Simulate Fault", "primary"



# Call back functions
@app.callback(
    Output("machine-trend", "figure"),
    Input("data-queue", "data"),
    State("feature-select", "value")
)
def update_figure(data_queue, feature):

    plot_data = pd.read_json(data_queue) # Create pandas dataframe from JSON data
    plot_data = plot_data.set_index('timestamp') # Set time index

    
    figure = go.Figure(
        data = go.Scatter(x=plot_data.index, y=plot_data[feature])
    )

    figure.update_layout(yaxis=dict(range=[0,plot_data[feature].max()+2]),
                          xaxis_title="Time",
                          yaxis_title=f"Parameter {feature}",
                          title="Trend of machine parameters")
    
    # figure.update_layout(xaxis=dict(range=[10000, 0]))

    return figure


@app.callback(
    Output("latest-healthy-data-index", "data"),
    Output("latest-fault-data-index", "data"),
    Output("data-queue", "data"),
    Output("prediction", "data"),
    Input("interval", "n_intervals"),
    State("latest-healthy-data-index", "data"),
    State("latest-fault-data-index", "data"),
    State("data-queue", "data"),
    State("button-simulation", "n_clicks"),
    State("model_used", "value")
)
def update_data_queue(n_interval, healthy_data_idx, fault_data_index, data_queue, n_clicks, model_used):

    if (n_clicks % 2 != 0): # Get faulty data
        new_data, fault_data_index = faulty_data.get_data_by_id(fault_data_index+1) # Get a new datapoint and update id
    else: 
        new_data, healthy_data_idx = healthy_data.get_data_by_id(healthy_data_idx+1) # Get a new datapoint and update id

    if n_interval == None:
        # Initialise a new pandas dataframe for the first time
        columns=healthy_data.get_column_names().append('timestamp')
        data_queue = pd.DataFrame(columns=columns)
    else:
        # As data queue is in JSON format  - get dataframe from it
        data_queue = pd.read_json(data_queue)
    
    # Getting current time stamp
    now = datetime.now()
    now = now.strftime("%Y-%m-%d %H:%M:%S")
    new_data['timestamp'] = pd.to_datetime(now)

    # add new data to dataframe
    if data_queue.shape[0] == DATA_WINDOW:
        # Remove the first data point from queue
        data_queue = data_queue.drop(index=data_queue.iloc[0].name)
    
    # Append new point to data frame
    data_queue = data_queue.append(new_data, ignore_index=True)

    # Get the prediction from model
    if model_used == "clustering":
        prediction = clustering_model.predict(data_queue)
    elif model_used == "isolation":
        prediction = isolation_model.predict(data_queue)
   
    # Return the jsonified version
    return healthy_data_idx, fault_data_index, data_queue.to_json(), prediction


@app.callback (
    Output("result", "children"),
    Output("result", "color"),
    Input("prediction", "data")
)
def update_prediction(prediction):
    if prediction == -1:
        return "Not enough data to predict machine health, please wait ..", "info"
    elif prediction == 0:
        return "Machine is healthy", "success"
    elif prediction == 1:
        return "Air leak suspected", "danger"




app.layout = html.Div([dcc.Location(id="url"), sidebar, content])




