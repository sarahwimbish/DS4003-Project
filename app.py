############################################################################################################################################
#########################################################            IMPORTS           #####################################################
############################################################################################################################################
from dash import Dash, dcc, html, Input, Output, callback
import pandas as pd
import plotly as py
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import numpy as np
import plotly.tools as tls
from io import BytesIO
import base64




stylesheet = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = Dash(__name__, external_stylesheets=stylesheet)
server = app.server

data = pd.read_csv('data/data.csv')
data['date'] = pd.to_datetime(data['date'])


############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
########################################################           PASTE BELOW           ###################################################
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################



############################################################################################################################################
#######################################################            HTML CHUNKS           ###################################################
############################################################################################################################################

colors = ['#4487b9', '#e4a23a', '#a18b31', '#cae2e9', '#93a5d0', '#eed9cb']
color = '#4487b9'
stroke = 2

left1 = [
    html.H2("My Listening Habits", style={'textAlign': 'center'}),
    html.P(id="streaming_minutes", style={'color':'white', 'textAlign': 'center', 'font-weight': 'bold', 'font-size':'60px',
                                           "text-shadow":f' -1px -{stroke}px 0 {color}, 1px -{stroke}px 0 {color}, \
                                            -1px {stroke}px 0 {color}, 1px {stroke}px 0 {color}, \
                                            -{stroke}px -1px 0 {color}, {stroke}px -1px 0 {color}, \
                                            -{stroke}px 1px 0 {color}, {stroke}px 1px 0 {color},  \
                                            -{stroke}px -{stroke}px 0 {color}, {stroke}px -{stroke}px 0 {color}, \
                                            -{stroke}px {stroke}px 0 {color}, {stroke}px {stroke}px 0 {color}'}),
    html.Center("Total Streaming Minutes"),
    html.P(id="different_songs", style={'color':'white', 'textAlign': 'center', 'font-weight': 'bold',  'font-size':'70px', 
                                           "text-shadow":f' -1px -{stroke}px 0 {color}, 1px -{stroke}px 0 {color}, \
                                            -1px {stroke}px 0 {color}, 1px {stroke}px 0 {color}, \
                                            -{stroke}px -1px 0 {color}, {stroke}px -1px 0 {color}, \
                                            -{stroke}px 1px 0 {color}, {stroke}px 1px 0 {color},  \
                                            -{stroke}px -{stroke}px 0 {color}, {stroke}px -{stroke}px 0 {color}, \
                                            -{stroke}px {stroke}px 0 {color}, {stroke}px {stroke}px 0 {color}'}),
    html.Center("Different Songs"),
    html.P(id="different_artists", style={'color':'white', 'textAlign': 'center', 'font-weight': 'bold',  'font-size':'70px', 
                                          "text-shadow":f' -1px -{stroke}px 0 {color}, 1px -{stroke}px 0 {color}, \
                                            -1px {stroke}px 0 {color}, 1px {stroke}px 0 {color}, \
                                            -{stroke}px -1px 0 {color}, {stroke}px -1px 0 {color}, \
                                            -{stroke}px 1px 0 {color}, {stroke}px 1px 0 {color},  \
                                            -{stroke}px -{stroke}px 0 {color}, {stroke}px -{stroke}px 0 {color}, \
                                            -{stroke}px {stroke}px 0 {color}, {stroke}px {stroke}px 0 {color}'}),
    html.Center("Different Artists"),
    html.Br(),
    html.Center(id = 'date_annotation')
]

############################################################################################################################################
#######################################################            IMAGE EXPORT           ##################################################
############################################################################################################################################

def fig_to_uri(in_fig, close_all=True, **save_args):
    out_img = BytesIO()
    in_fig.savefig(out_img, format='png', **save_args)
    if close_all:
        in_fig.clf()
        plt.close('all')
    out_img.seek(0)  # rewind file
    encoded = base64.b64encode(out_img.read()).decode("ascii").replace("\n", "")
    return "data:image/png;base64,{}".format(encoded)

############################################################################################################################################
###########################################################            APP           #######################################################
############################################################################################################################################

app.layout = html.Div( # entire page
    [
        html.Div( # Top of Page 
            [
                html.H1("Sarah's Spotify Dashboard"), # title 
                html.H3("blah blah blah "), # description
            ],style={"border":"2px solid pink", 'margin':'5px', 'textAlign': 'center'}),

        html.Div(# Bottom of Page 
            [ 
                        html.Div(left1, # left 3 col of left 
                                className= "three columns", 
                                style={'margin':'5px', "border":"2px solid pink", 'align-items':'center', 'justify-content':'center'}), 
                        
                        html.Div(# right 6 col of left 
                            [html.Div([
                                        "WIDGETS",
                                        dcc.DatePickerRange(
                                            id='date-slider',
                                            min_date_allowed= data['date'].min(),
                                            max_date_allowed= data['date'].max(),
                                            start_date = data['date'].min(),
                                            end_date = data['date'].max()),
                                        html.Button('RESET', id='reset-button', n_clicks=0),
                                        ], style={'border':'2px solid pink', 'margin':'5px'}),
                             html.Div([dcc.Graph(id='line-plot')], style={'border':'2px solid pink','margin':'5px'}),
                             html.Div("MORE WIDGETS", style={'border':'2px solid pink', 'margin':'5px'}), 
                            
                            ], className='six columns', style ={'margin':'5px'}),

                        html.Div(
                                [ html.Img(id = 'plot1', src = '', style={'width': '400px', 'height': '400px'}),
                                            dcc.RadioItems(options = ["AM", "PM"],
                                                        value = "AM",
                                                        inline = True,
                                                        id='AMPM-radio',
                                                        style={'textAlign': 'center', 'margin':'5px'})
                                ], className='three columns', style={'margin':'5px'})
    ], style={"border":"2px solid pink"}, className = 'row') ])

if __name__ == '__main__':
    app.run_server(jupyter_mode='tab', debug=True)

############################################################################################################################################
########################################################            CALLBACKS           ####################################################
############################################################################################################################################

@callback(
        Output('streaming_minutes', 'children'),
        Input('date-slider', 'start_date'),
        Input('date-slider', 'end_date')
)
def get_streaming_minutes(start, end):
    data_filtered = data[data['date'].between(start, end)]
    total_mins = data_filtered['minutes'].sum()
    total_mins = round(total_mins)
    return str(total_mins)


@callback(
        Output('different_songs', 'children'),
        Input('date-slider', 'start_date'),
        Input('date-slider', 'end_date')
)
def get_different_songs(start, end):
    data_filtered = data[data['date'].between(start, end)]
    diff_songs = data_filtered['trackName'].nunique()
    return str(diff_songs)


@callback(
        Output('different_artists', 'children'),
        Input('date-slider', 'start_date'),
        Input('date-slider', 'end_date')
)
def get_different_songs(start, end):
    data_filtered = data[data['date'].between(start, end)]
    diff_artists = data_filtered['artistName'].nunique()
    return str(diff_artists)

@callback(
        Output('date_annotation', 'children'),
        Input('date-slider', 'start_date'),
        Input('date-slider', 'end_date')
)
def get_date_annotations(start, end):
    return f'Between \n{start} and {end}'

@app.callback(
    Output('date-slider', 'start_date'),
    Output('date-slider', 'end_date'),
    Input('reset-button', 'n_clicks')
)
def reset_date_range_slider(n_clicks):
    return data['date'].min(), data['date'].max()


############################################################################################################################################
##########################################################            PLOTS           ######################################################
############################################################################################################################################

@callback(
    Output('plot1', 'src'),
    Input('AMPM-radio', 'value'),
    Input('date-slider', 'start_date'),
    Input('date-slider', 'end_date')
)
def create_plot1(hours_selected, start, end):
    ## group by the hour of the day and find the number of minutes 
    data_filtered = data[data['date'].between(start, end)]
    hours = data_filtered.groupby('hour')['minutes'].sum()
    
    # make sure every index is included in the series 
    hours = hours.reindex(range(24), fill_value= 0)

    # change the index 
    hours.index = [12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, \
                    3, 4, 5, 6, 7, 8, 9, 10, 11]
    hoursam = hours.iloc[:12]
    hourspm = hours.iloc[12:]

    hours_to_use = hoursam
    AMPM = "AM"
    color1 = '##e5a13a'

    if(hours_selected=="AM"):
        hours_to_use = hoursam
        AMPM ='AM' 
        color1 = '#fca828'
    else:
        hours_to_use = hourspm
        AMPM='PM'
        color1 = '#4487b9'

        #initialize the plot 
    fig = plt.figure(figsize=(20,20))
    ax = plt.subplot(111, polar=True)
    plt.axis('off')


    # Compute max and min in the dataset
    max = hours_to_use.max()

    # Set the coordinates limits
    upperLimit = max
    lowerLimit = max/3

    # Let's compute heights: they are a conversion of each item value in those new coordinates
    # In our example, 0 in the dataset will be converted to the lowerLimit (10)
    # The maximum will be converted to the upperLimit (100)
    slope = (max - lowerLimit) / max
    heights = slope * hours_to_use.values

    #Compute the width of each bar. In total we have 2*Pi = 360°
    width = 2*np.pi / len(hours_to_use.index)

    # Compute the angle each bar is centered on:
    indexes = list(range(1, len(hours_to_use)+1))
    angles = [element * width * -1 + np.deg2rad(120) for element in indexes]


    # add bars 
    bars = ax.bar(
        x=angles, 
        height=heights, 
        width=width, 
        bottom=2*lowerLimit,
        linewidth=3, 
        edgecolor="white",
        color=color1,
    )

    # little space between the bar and the label
    labelPadding = max/15


    # Add labels
    for bar, angle, height, label in zip(bars, angles, heights, hours_to_use.values):

        # Labels are rotated. Rotation must be specified in degrees :(
        rotation = np.rad2deg(angle)-90

        # Finally add the labels
        ax.text(
            x=angle, 
            y=2*lowerLimit + bar.get_height() + labelPadding, 
            s=f'{round(label)}\nmins', 
            ha='center', 
            va='center', 
            rotation=rotation, 
            size = 30,
            rotation_mode="anchor") 


    # Add more lables labels
    for bar, angle, height, label in zip(bars, angles, heights, hours_to_use.index):

        # Labels are rotated. Rotation must be specified in degrees :(
        rotation = np.rad2deg(angle)-90

        # Finally add the labels
        ax.text(
            x=angle, 
            y=2*lowerLimit + bar.get_height() + 3*labelPadding, 
            s=f'{label} {AMPM}', 
            ha='center', 
            va='center',
            size = 40,
            rotation=rotation, 
            rotation_mode="anchor") 
    out_url = fig_to_uri(fig)
    return out_url


@callback(
    Output('line-plot', 'figure'),
    Input('date-slider', 'start_date'),
    Input('date-slider', 'end_date')
)
def create_line_plot(start, end):
    data_filtered = data[data['date'].between(start, end)]

    totalminutes = data_filtered.groupby('date')['minutes'].sum()
    totalminutes = totalminutes.apply(lambda x: int(x))

    data_filtered = data_filtered[data_filtered['played']==True]

    numberofartists = data_filtered.groupby('date')['artistName'].nunique()
    numberofsongs = data_filtered.groupby('date')['trackName'].nunique()

    alldata = pd.DataFrame(totalminutes)
    alldata['numberofsongs'] = numberofsongs
    alldata['numberofartists'] = numberofartists

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y = alldata.minutes,
        x = alldata.index,
        mode = 'lines',
        name = "Streaming Minutes",
        line_color = "#eba8c6"
    ))

    fig.add_trace(go.Scatter(
        y = alldata.numberofsongs,
        x = alldata.index,
        mode = 'lines',
        name = "Unique Song Count",
        line_color = "#4487b9"
    ))

    fig.add_trace(go.Scatter(
        y = alldata.numberofartists,
        x = alldata.index,
        mode = 'lines',
        name = 'Unique Artist Count',
        line_color = '#e5a13a'
    ))

    fig.update_layout(template='plotly_white',
                    legend=dict(yanchor="bottom", xanchor='center', x = .5, y= -.15, orientation = 'h'))
                    
    fig.update_layout(
        margin=dict(l=50, r=50, b=50, t=0)
    )
    return fig
