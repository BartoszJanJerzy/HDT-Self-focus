import os
import pandas as pd
import heartpy as hp
import numpy as np
import plotly.express as px
import plotly.graph_objects as go



def get_files():
    path = os.path.join(os.getcwd(), 'Heart')
    filenames = os.listdir(path)
    filenames_good = []
    
    for i in range(len(filenames)):
        name = filenames[i]
        if name.endswith('.txt'):
            filenames_good.append(filenames[i])
    
    dfs = []
    for name in filenames_good:
        with open(os.path.join(path, name), 'r') as file:
            example_file = file.read()
        
        tab_file = example_file.split('EndOfHeader\n')[1]

        with open(os.path.join(path, 'temp_file.csv'), 'w') as file:
            file.write(tab_file)
        
        df = pd.read_csv(
            os.path.join(path, 'temp_file.csv'),
            sep='\t',
            names = ['id', 'b', 'c', 'CleanSignal','e', 'Condition', 'g']
        )
        dfs.append(df[['CleanSignal','Condition']])
    
    return dfs


def get_labels(df):
    conditions = df['Condition'].to_list()
    labels_i = []
    labels_cond = []
    prev_pos = 0
    prev_cond = 'brak'
    i = -1
    label_i = 0

    for c in conditions:
        if c in [1,2,3,4]:
            prev_pos = c

        if c==0 and prev_pos:
            label_cond = prev_pos
            
            if prev_cond != prev_pos:
                i += 1
            label_i = i
            prev_cond = prev_pos
            
        elif c in [1,2,3,4]:
            label_cond = c
            if prev_cond != prev_pos:
                i += 1
            label_i = i
            prev_cond = prev_pos
        elif c==5:
            label_cond = prev_pos
            if prev_cond != prev_pos:
                i += 1
            label_i = i
            prev_pos = 0
            prev_cond = prev_pos
        else:
            label_cond = 'brak'
            prev_cond = 'brak'

        labels_cond.append(label_cond)
        labels_i.append(label_i)
    
    return labels_cond, labels_i


def show_conditions(start, end, df):
    df_graph = df[start:end]
    
    fig = px.line(
        df_graph,
        x = df_graph.index,
        y = 'Condition',
        color = 'Warunek',
    )
    fig.show()
    
    
def get_data(df, col, max_good_value, min_good_value, lower, upper):
    my_median = df[col].median()
    data = df[col].apply(lambda x: my_median if x > max_good_value else (my_median if x < min_good_value else x))
    data = hp.preprocessing.scale_data(data, lower=lower, upper=upper)
    return data


def get_working_data(data, hz):
    working_data, measures = hp.process(
        data,
        hz,
        bpmmin = np.min(data),
        bpmmax = np.max(data),
    )
    return working_data


def get_peak_trace(working_data, df_len):
    all_peaks = working_data['peaklist']
    wrong_peaks = working_data['removed_beats']
    trace = [0]*df_len
    
    for i in all_peaks:
        if i not in wrong_peaks:
            trace[i] = 1
    return trace


def show_peaks(df):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        name = 'Sygnał',
        x = df.index,
        y = df['CleanSignal'],
        mode='lines',
        marker_color='silver'
    ))
    df_peaks = df[df['Peak'] == 1]
    
    conditions = df['Warunek'].unique()
    for cond in conditions:
        if cond != 'brak':
            df_cond = df_peaks[df_peaks['Warunek'] == cond]
            fig.add_trace(go.Scatter(
                name = f'Peaks {cond}',
                x = df_cond.index,
                y = df_cond['CleanSignal'],
                mode='markers',
                marker_size=3
            ))
    
    fig.update_layout(
        xaxis_title = 'Pomiar',
        yaxis_title = 'Wartość sygnał'
    )

    fig.show()