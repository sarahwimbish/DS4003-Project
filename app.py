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
import plotly.express as px 
from datetime import date
import datetime



stylesheet = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = Dash(__name__, external_stylesheets=stylesheet)
app.css.append_css({"external_url": "/assets/main.css"})
app.server.static_folder = "assets"

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
curve = 10
border = ' #FFFFFF'
shadowx = 2
shadowy = 2
blur = 4

left1 = [
    html.Br(), 
    html.H2(["My", html.Br(), "Listening", html.Br(), "Habits"], style={'textAlign': 'center'}),
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
    html.Center(id = 'date_annotation', style = {'margin-bottom':"15px"}),
]

top_artists = [
    html.Div(html.H4(["My Top", html.Br(), 'Artists']), className = 'myTop'), 

    html.Div([html.Div(1, className = 'number2'), 
              html.Div(className = 'text', id = 'artists1')], className = 'boxes'), 

    html.Div([html.Div(2, className = 'number2'), 
              html.Div(className = 'text', id = 'artists2')], className = 'boxes'), 
    
    html.Div([html.Div(3, className = 'number2'), 
              html.Div(className = 'text', id = 'artists3')], className = 'boxes'), 

]

top_songs = [
    html.Div(html.H4(["My Top", html.Br(), 'Songs']), className = 'myTop'), 

    html.Div([html.Div(1, className = 'number'), 
              html.Div(className = 'text', id = 'song1')], className = 'boxes'), 

    html.Div([html.Div(2, className = 'number'), 
              html.Div(className = 'text', id = 'song2')], className = 'boxes'), 
    
    html.Div([html.Div(3, className = 'number'), 
              html.Div(className = 'text', id = 'song3')], className = 'boxes'), 

]

############################################################################################################################################
#######################################################            IMAGE EXPORT           ##################################################
############################################################################################################################################

def fig_to_uri(in_fig, close_all=True, **save_args):
    """
    Save a figure as a URI
    :param in_fig:
    :return:
    """
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

border = 'white'

app.layout = html.Div( # entire page
    [
        html.Div( # Top of Page 
            [               html.Div(html.Div([
                            html.Div(html.H1("Sarah's Spotify Dashboard"), className= 'description', style ={'margin':'20px 0 0 50px'}),
                            html.Div(html.A(html.Div([html.Img(id = 'githublink', src="/assets/github-logo.png", style={'width':'30px', 'height':'30px'})
                            ]), href="https://github.com/sarahwimbish/DS4003-Project", target='blank'), className = 'github-link')
                        ]), className = 'description-and-link'), # description
            ], id = 'mainContainer',style={"border":"2px {boarder}", 
                     'textAlign': 'center', 
                     'background-color':'white', 
                     'border-radius': f'{curve}px {curve}px {curve}px {curve}px',
                     'margin':'5px 5px',
                     'filter': f'drop-shadow({shadowx}px {shadowy}px {blur}px #9b7d81)'}),


        html.Div(# Bottom of Page 
            [ 

                html.Div(left1, # left 3 col of left 
                        className= "container1", 
                        style={"border":"2px {boarder}", 
                                'align-items':'center', 
                                'justify-content':'center', 
                                'background-color':'white', 
                                'border-radius': f'{curve}px {curve}px {curve}px {curve}px',
                                'margin':'0px 5px',
                                'filter': f'drop-shadow({shadowx}px {shadowy}px {blur}px #9b7d81)'}), 
                


                html.Div(# right 6 col of left 
                    [html.Div([
                                dcc.DatePickerRange(
                                    id='date-slider',
                                    min_date_allowed= data['date'].min(),
                                    max_date_allowed= data['date'].max(),
                                    start_date = data['date'].min(),
                                    end_date = data['date'].max(), 
                                    className = 'five columns'),
                                html.Button('RESET DATE', id='reset-button', n_clicks=0, style={'background-color': 'white'}, className = 'three columns'),
                                dcc.RadioItems(options = ["Daily", "Weekly", "Monthly"],
                                    value = "Daily",
                                    inline = True,
                                    id='timeframe-radio',
                                    style={'textAlign': 'center', 'vertical-align': 'middle', 'margin': '0px'}, inputStyle={"margin-left": "12px"}, className = 'four columns')
                                ], style={'border':'2px {border}',
                                            'border-radius': f'{curve}px {curve}px {curve}px {curve}px',
                                            'background-color': 'white',
                                            'padding':'10px',
                                            'margin':'0px 0px 5px 0px',
                                            'filter': f'drop-shadow({shadowx}px {shadowy}px {blur}px #9b7d81)',
                                            'height':'40px',
                                            'z-index':'10'}),
                        html.Div([dcc.Graph(id='line-plot')], style={'border':'2px {border}',
                                                                    'border-radius':f'{curve}px {curve}px {curve}px {curve}px',
                                                                    'background-color': 'white',
                                                                    'padding':'5px',
                                                                    'margin':'5px 0px 0px 0px',
                                                                    'filter': f'drop-shadow({shadowx}px {shadowy}px {blur}px #9b7d81',
                                                                    'z-index':'1'}),
                        html.Div(top_artists, style={'border-radius':f'{curve}px {curve}px {curve}px {curve}px',
                                                                    'background-color': 'white',
                                                                    'margin':'5px 0px 0px 0px', 
                                                                    'overflow': 'hidden',
                                                                    'align-items': 'center',
                                                                    'filter': f'drop-shadow({shadowx}px {shadowy}px {blur}px #9b7d81)'}),
                        html.Div(top_songs, style={'border-radius':f'{curve}px {curve}px {curve}px {curve}px',
                                                                    'background-color': 'white',
                                                                    'margin':'5px 0px 0px 0px', 
                                                                    'overflow': 'hidden',
                                                                    'align-items': 'center',
                                                                    'filter': f'drop-shadow({shadowx}px {shadowy}px {blur}px #9b7d81)'})                      
                    ], className='container2', style ={'margin':'0px 5px'}),



                    html.Div( ## right panel 
                        [html.Div([html.Img(id = 'plot1', src = '', style={'width': '320px',  ### want 380px but moves out of the div 
                                                                    'height': '320px',
                                                                    'padding':'5px',
                                                                    'padding-left':'35px',
                                                                    'align-items': 'center',
                                                                     'justify-content': 'center'}),
                        dcc.RadioItems(options = ["AM", "PM"],
                                    value = "AM",
                                    inline = True,
                                    id='AMPM-radio',
                                    style={'textAlign': 'center'}, inputStyle={"margin-left": "20px"})
                        ], style = {'background-color':'white', 
                                    'border-radius': f'{curve}px {curve}px {curve}px {curve}px',
                                    'margin':'0px 5px',
                                    'filter': f'drop-shadow({shadowx}px {shadowy}px {blur}px #9b7d81)'}), 
                        html.Div(dcc.Graph(id='bar-plot'), style ={'background-color':'white', 
                                                                     'border-radius': f'{curve}px {curve}px {curve}px {curve}px',
                                                                     'margin':'5px 5px', 
                                                                     'padding':'5px',
                                                                     'padding-left':'35px', 
                                                                     'filter': f'drop-shadow({shadowx}px {shadowy}px {blur}px #9b7d81)'}),
                        # DashIconify(icon="ion:logo-github", width=30, href='github.com')
                        ], className = 'container3', style={})
    ], style={"border":"2px {border}", 'margin-top': '0px'}) ], className='row')


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
    data_filtered = data_filtered[data_filtered['played']==True]
    diff_songs = data_filtered['trackName'].nunique()
    return str(diff_songs)

@callback(
        Output('different_artists', 'children'),
        Input('date-slider', 'start_date'),
        Input('date-slider', 'end_date')
)
def get_different_artists(start, end):
    data_filtered = data[data['date'].between(start, end)]
    data_filtered = data_filtered[data_filtered['played']==True]
    diff_artists = data_filtered['artistName'].nunique()
    return str(diff_artists)

@callback(
        Output('date_annotation', 'children'),
        Input('date-slider', 'start_date'),
        Input('date-slider', 'end_date')
)
def get_date_annotations(start, end):
    return html.P([f'Between {start.strip("T00:00:00")}', html.Br(), f'{end.strip("T00:00:00")}'])

@callback(
    Output('date-slider', 'start_date'),
    Output('date-slider', 'end_date'),
    Input('reset-button', 'n_clicks')
)
def reset_date_range_slider(n_clicks):
    return data['date'].min(), data['date'].max()

@callback(
    Output('artists1', 'children'),
    Input('date-slider', 'start_date'),
    Input('date-slider', 'end_date')
)
def get_artist1(start, end):
    data_filtered = data[data['date'].between(start, end)]
    top = data_filtered.groupby('artistName', as_index=False)['minutes'].sum()
    top = top.sort_values('minutes', ascending = False).reset_index()
    return [html.H6(f"{top['artistName'][0]}"), html.P(f"{round(top['minutes'][0])} minutes")]

@callback(
    Output('artists2', 'children'),
    Input('date-slider', 'start_date'),
    Input('date-slider', 'end_date')
)
def get_artist2(start, end):
    data_filtered = data[data['date'].between(start, end)]
    top = data_filtered.groupby('artistName', as_index=False)['minutes'].sum()
    top = top.sort_values('minutes', ascending = False).reset_index()
    return [html.H6(f"{top['artistName'][1]}"), html.P(f"{round(top['minutes'][1])} minutes")]

@callback(
    Output('artists3', 'children'),
    Input('date-slider', 'start_date'),
    Input('date-slider', 'end_date')
)
def get_artist3(start, end):
    data_filtered = data[data['date'].between(start, end)]
    top = data_filtered.groupby('artistName', as_index=False)['minutes'].sum()
    top = top.sort_values('minutes', ascending = False).reset_index()
    return [html.H6(f"{top['artistName'][2]}"), html.P(f"{round(top['minutes'][2])} minutes")]


@callback(
    Output('song1', 'children'),
    Input('date-slider', 'start_date'),
    Input('date-slider', 'end_date')
)
def get_song1(start, end):
    data_filtered = data[data['date'].between(start, end)]
    top = data_filtered.groupby(['trackName', 'artistName'], as_index=False)['minutes'].sum()
    top = top.sort_values('minutes', ascending = False).reset_index()
    return [html.H6(f"{top['trackName'][0]}", className='artist'), html.P(f"by {top['artistName'][0]}")]

@callback(
    Output('song2', 'children'),
    Input('date-slider', 'start_date'),
    Input('date-slider', 'end_date')
)
def get_song2(start, end):
    data_filtered = data[data['date'].between(start, end)]
    top = data_filtered.groupby(['trackName', 'artistName'], as_index=False)['minutes'].sum()
    top = top.sort_values('minutes', ascending = False).reset_index()
    return [html.H6(f"{top['trackName'][1]}", className='artist'), html.P(f"by {top['artistName'][1]}")]

@callback(
    Output('song3', 'children'),
    Input('date-slider', 'start_date'),
    Input('date-slider', 'end_date')
)
def get_song3(start, end):
    data_filtered = data[data['date'].between(start, end)]
    top = data_filtered.groupby(['trackName', 'artistName'], as_index=False)['minutes'].sum()
    top = top.sort_values('minutes', ascending = False).reset_index()
    return [html.H6(f"{top['trackName'][2]}", className='artist'), html.P(f"by {top['artistName'][2]}")]


@callback(
    Output('timeframe-radio', 'options'),
    Input('date-slider', 'start_date'), 
    Input('date-slider', 'end_date')
)
def update_radio_items(start, end):
    if start and end:
        # Calculate the duration between start and end dates
        start = datetime.datetime.fromisoformat(start)
        end = datetime.datetime.fromisoformat(end)
        duration = end - start

        # Default radio options
        options = [
            {'label': 'Daily', 'value': 'Daily'},
            {'label': 'Weekly', 'value': 'Weekly'},
            {'label': 'Monthly', 'value': 'Monthly'}
        ]

        # Disable "Monthly" if duration is less than 6 months
        if duration.days < 180:  # 6 months approx
            options = [opt for opt in options if opt['value'] != 'Monthly']

        # Disable "Weekly" if duration is less than 6 weeks
        if duration.days < 42:  # 6 weeks
            options = [opt for opt in options if opt['value'] != 'Weekly']

        return options

    # Fallback to default options in case of errors
    return [
        {'label': 'Daily', 'value': 'Daily'},
        {'label': 'Weekly', 'value': 'Weekly'},
        {'label': 'Monthly', 'value': 'Monthly'}
    ]


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
    color1 = '#fca828'

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
    Input('date-slider', 'end_date'),
    Input('timeframe-radio', 'value')
)
def create_line_plot(start, end, timeframe):
    data_filtered = data[data['date'].between(start, end)]
    data_filtered = data_filtered[data_filtered['played']==True]
    data_filtered['week_number'] = data_filtered['date'].dt.to_period('W')
    data_filtered['week_number'] = data_filtered['week_number'].dt.start_time
    data_filtered['month_name'] = data_filtered['date'].dt.to_period('M').astype(str)

    start = datetime.datetime.fromisoformat(start)
    end = datetime.datetime.fromisoformat(end)
    duration = end - start
    
    # Group by the selected timeframe
    if timeframe == 'Daily':
        data_grouped = data_filtered.groupby('date')
    elif timeframe == 'Weekly':
        # Get week number from date
        data_grouped = data_filtered.groupby('week_number')
    elif timeframe == 'Monthly':
        # Get month name from date
        data_grouped = data_filtered.groupby('month_name')
    else:
        data_grouped = data_filtered.groupby('date')


    totalminutes = data_grouped['minutes'].sum()
    totalminutes = totalminutes.apply(lambda x: int(x))
    numberofartists = data_grouped['artistName'].nunique()
    numberofsongs = data_grouped['trackName'].nunique()

    alldata = pd.DataFrame(totalminutes)
    alldata['numberofsongs'] = numberofsongs
    alldata['numberofartists'] = numberofartists

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y = alldata.minutes,
        x = alldata.index,
        mode = 'lines',
        name = "Streaming Minutes",
        # fill = '#eba8c6',
        line_color = "#eba8c6",
        line_width = 1
    ))

    fig.add_trace(go.Scatter(
        y = alldata.numberofsongs,
        x = alldata.index,
        mode = 'lines',
        name = "Unique Song Count",
        line_color = "#4487b9", 
        line_width = 1
    ))

    fig.add_trace(go.Scatter(
        y = alldata.numberofartists,
        x = alldata.index,
        mode = 'lines',
        name = 'Unique Artist Count',
        line_color = '#e5a13a', 
        line_width = 1
    ))

    fig.update_layout(template='plotly_white',
                    legend=dict(yanchor="bottom", xanchor='center', x = .5, y= -.2, orientation = 'h'),
                    height = 355, 
                    yaxis=dict(title=''),
                    margin=dict(l=50, r=50, b=50, t=0))

    return fig

@callback(
    Output('bar-plot', 'figure'),
    Input('date-slider', 'start_date'),
    Input('date-slider', 'end_date'),
    Input('timeframe-radio', 'value')
)
def create_bar_plot(start, end, timeframe):
    data_filtered = data[data['date'].between(start, end)]

    d0 = pd.to_datetime(start)
    d1 =pd.to_datetime(end)
    days = (d1 - d0).days
    axis = 'Average Daily <br> Listening Minutes'

    if timeframe == 'Daily':
        days = (d1 - d0).days
        axis = 'Average Daily <br> Listening Minutes'
    elif timeframe == 'Weekly':
        days = (d1 - d0).days/7
        axis = 'Average Weekly <br> Listening Minutes'
    elif timeframe == 'Monthly':
        days = (d1 - d0).days/30
        axis = 'Average Monthly <br> Listening Minutes'
    else:
        days = (d1 - d0).days
        axis = 'Average Daily <br> Listening Minutes'

    times = data_filtered.groupby(['hour','timeofday'], as_index=False)['minutes'].sum()
    times['minutes'] = times['minutes']/days

    fig = go.Figure()
    fig = px.bar(times.sort_values('hour'), 
                x='hour', 
                y='minutes',
                color = 'timeofday',
                color_discrete_map ={'Morning': '#eba8c6', 'Afternoon': '#4487b9', 'Night':'#e5a13a'}, 
                template = 'plotly_white', 
                labels = {'minutes': axis,
                        'hour':'Hour of Day',
                        'timeofday':''}, 
                height=265)
    fig.update_layout(showlegend = False, legend_traceorder="reversed",
                      legend=dict(yanchor="auto", xanchor='auto', x = .5, y= -1.5, orientation = 'h'))
    return fig 