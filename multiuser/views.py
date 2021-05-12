from django.shortcuts import render
from django.http import HttpResponse
from os import path
import os
import pickle
import plotly.offline as opy
import plotly.graph_objs as go
import numpy as np
import sqlite3
import dash
import dash_core_components as dcc
import dash_html_components as html
from django_plotly_dash import DjangoDash
from plotly.subplots import make_subplots
import dash_daq as daq
from skimage import data
import plotly.express as px
from dash.dependencies import Input, Output
import json
import datetime
from scipy import signal
from django.contrib import auth
from multiuser.viz_utils import *
from django.views.static import serve

data_db_ = sqlite3.connect('ppg_anno_dataset/WESAD.db')
c = data_db_.cursor()
c.execute("SELECT MAX(_ROWID_) FROM dataset LIMIT 1")
data_total = c.fetchone()[0] - 1
data_db_.close()

current_reanno_idx_user = 0


def login(request):
    return render(request, 'multiuser/login.html')


def process_credential(request):
    msg = dict()
    username = request.POST.get('username', None)
    if 'login' in request.POST:
        rst = log_in_handler(request, username)
        if isinstance(rst, str):
            msg['feedback'] = rst
        else:
            return rst
    elif 'signup' in request.POST:
        rst = sign_up_handler(request, username)
        msg['feedback'] = rst
    return render(request, 'multiuser/login.html', context=msg)


def log_in_handler(request, username):
    cwd = os.getcwd()
    current_user = username
    if os.path.isdir(path.join(cwd, 'multiuser/multiuser_outputs/{}'.format(username))):
        current_idx = fetch_index(current_user)
        request.session['current_user'] = username
        msg = {'data_total': data_total, 'current_idx': min(current_idx, data_total), 'current_user': username,
               'progress':   get_progress(current_user),
               'dash_app': 'SignalPlotDash_{}'.format(request.session['current_user'])}
        return menu(request, msg)
    else:
        return 'User {} does not exist, please sign up first'.format(username)


def sign_up_handler(request, username):
    cwd = os.getcwd()
    if os.path.isdir(path.join(cwd, 'multiuser/multiuser_outputs/{}'.format(username))):
        return 'User {} already exists, please log in'.format(username)
    else:
        os.mkdir(path.join(cwd, 'multiuser/multiuser_outputs/{}'.format(username)))
        os.mkdir(path.join(cwd, 'multiuser/multiuser_outputs/{}/cache'.format(username)))
        return 'User {} signed up successfully, please log in'.format(username)


def fetch_index(current_user):
    cwd = os.getcwd()
    if not path.exists(path.join(cwd, 'multiuser/multiuser_outputs/{}/cache/counter.npy'.format(current_user))):
        np.save(path.join(cwd, 'multiuser/multiuser_outputs/{}/cache/counter.npy'.format(current_user)),
                np.asarray([0]))
        return 0
    else:
        counter = np.load(path.join(cwd, 'multiuser/multiuser_outputs/{}/cache/counter.npy'.format(current_user)))
        return int(counter)


def update_index(idx, current_user):
    cwd = os.getcwd()
    np.save(path.join(cwd, 'multiuser/multiuser_outputs/{}/cache/counter.npy'.format(current_user)), np.asarray([idx]))


def reset_relayout_cache(request):
    cwd = os.getcwd()
    current_user = request.session['current_user']
    np.save(path.join(cwd, 'multiuser/multiuser_outputs/{}/cache/relay_cache.npy'.format(current_user)), np.asarray([]))


def get_relayout_cache(request):
    cwd = os.getcwd()
    current_user = request.session['current_user']
    cache = np.load(path.join(cwd, 'multiuser/multiuser_outputs/{}/cache/relay_cache.npy'.format(current_user)))
    ret = set()
    for idx in range(0, cache.shape[0], 2):
        ret.add(tuple((cache[idx], cache[idx + 1])))
    return ret


def update_relayout_cache(request, start, end):
    cwd = os.getcwd()
    current_user = request.session['current_user']
    cache = np.load(path.join(cwd, 'multiuser/multiuser_outputs/{}/cache/relay_cache.npy'.format(current_user)))
    if cache.shape[0] == 0:
        cache = np.asarray([start, end])
    else:
        cache = np.concatenate((cache, np.asarray([start, end])))
    np.save(path.join(cwd, 'multiuser/multiuser_outputs/{}/cache/relay_cache.npy'.format(current_user)), cache)


def render_plots(request, idx):
    reset_relayout_cache(request)
    ret_data = fetch_data(idx)
    if not ret_data:
        return False
    else:
        ecg, acc0, acc1, acc2, ppg, act, param = ret_data

    times_64hz = calc_time_axis(64)

    ecg_64 = signal.resample(ecg, len(times_64hz))
    acc0_64 = signal.resample(acc0, len(times_64hz))
    acc1_64 = signal.resample(acc1, len(times_64hz))
    acc2_64 = signal.resample(acc2, len(times_64hz))
    act_64 = signal.resample(act, len(times_64hz))
    fig = make_subplots(rows=4, cols=1, vertical_spacing=0.02, shared_xaxes=True, row_heights=[1.2, 1.2, 1, 1])
    fig.append_trace(go.Scatter(x=times_64hz, y=ppg, mode='lines', name='PPG'), row=1, col=1)
    fig.append_trace(go.Scatter(x=times_64hz, y=ecg_64, mode='lines', name='ECG'), row=2, col=1)
    fig.append_trace(go.Scatter(x=times_64hz, y=acc0_64, mode='lines', name='3-Axis_ACC'), row=3, col=1)
    fig.append_trace(go.Scatter(x=times_64hz, y=acc1_64, mode='lines', name='3-Axis_ACC'), row=3, col=1)
    fig.append_trace(go.Scatter(x=times_64hz, y=acc2_64, mode='lines', name='3-Axis_ACC'), row=3, col=1)
    fig.append_trace(go.Scatter(x=times_64hz, y=act_64, mode='lines', name='Activity'), row=4, col=1)
    fig['layout']['yaxis4'].update(tickmode='array',
                                   tickvals=list(range(0, 9)),
                                   ticktext=['Transient', 'Sitting', 'Stairs', 'Table soccer',
                                             'Cycling', 'Driving', 'Lunch Break', 'Walking', 'Working']
                                   )
    fig.update_xaxes(tickmode='array',
                     tickvals=[x for x in range(0, 30 * 64 + 1) if x % 64 == 0],
                     ticktext=['{0:02}.0'.format(s) for s in range(0, 31)]
                     )

    # if existing_annos:
    #     for anno in existing_annos:
    #         start, end = anno
    #         fig.add_vrect(
    #             x0=start, x1=end,
    #             fillcolor="#ff7f0e", opacity=0.5,
    #             layer="below", line_width=0
    # )
    fig.update_yaxes(fixedrange=True)
    fig.update_layout(
        dragmode="drawrect",
        # dragmode="select",
        # selectdirection="h",
        newshape=dict(fillcolor="cyan", opacity=0.3),
        margin=dict(t=0, b=0, l=0),
    )
    config = {'scrollZoom': True, 'displayModeBar': True,
              'modeBarButtonsToRemove': ['toImage', 'zoom2d', 'pan2d', 'select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d',
                                         'autoScale2d'],
              'doubleClick': "reset"
              }
    app = DjangoDash('SignalPlotDash_{}'.format(request.session['current_user']))  # replaces dash.Dash
    app.layout = html.Div(
        [
            # html.H4("Drag and draw rectangle annotations"),
            dcc.Graph(id="graph-picture", figure=fig, config=config),
            html.Pre(id="annotations-data"),
        ],
        style={'width':'1000px'}
    )

    @app.callback(
        Output("annotations-data", "children"),
        [Input("graph-picture", "relayoutData")],
    )
    def on_new_annotation(relayout_data):

        if relayout_data is None:
            return
        if "shapes" in relayout_data:
            for shape_info in relayout_data["shapes"]:
                anno = tuple(sorted((shape_info["x0"] * ((1 / 64) * 1000), shape_info["x1"] * ((1 / 64) * 1000))))
                update_relayout_cache(request, max(anno[0], 0), min(anno[1], 1920 * ((1 / 64) * 1000)))
        else:
            return dash.no_update

    return ppg, param


def anno_home(request):
    current_user = request.session['current_user']
    current_idx = fetch_index(current_user)
    ret = render_plots(request, current_idx)
    if not ret:
        msg = {'feedback': 'This dataset is finished', 'data_total': data_total, 'current_idx': current_idx,
               'progress':   get_progress(current_user), 'finished': True,
               'current_user': current_user,
               'dash_app': 'SignalPlotDash_{}'.format(request.session['current_user'])}
    else:
        ppg, param = ret
        msg = {'param': param, 'current_idx': current_idx, 'current_user': current_user,
               'data_total': data_total, 'progress':   get_progress(current_user)
            , 'dash_app': 'SignalPlotDash_{}'.format(request.session['current_user'])}
    return render(request, 'multiuser/anno.html', context=msg)


def clear_anno(request, reanno=False):
    current_user = request.session['current_user']

    if reanno:
        current_idx = current_reanno_idx_user
    else:
        current_idx = fetch_index(current_user)
    ret = render_plots(request, current_idx)
    if not ret:
        msg = {'feedback': 'This dataset is finished', 'data_total': data_total, 'current_idx': current_idx,
               'progress':   get_progress(current_user), 'finished': True, 'current_user': current_user
            , 'dash_app': 'SignalPlotDash_{}'.format(request.session['current_user'])}
    else:
        ppg, param = ret
        msg = {'feedback': 'Cleared No. {}'.format(fetch_index(current_user)),
               'param': param, 'current_user': current_user,
               'current_idx': current_idx, 'data_total': data_total, 'progress':   get_progress(current_user)
            , 'dash_app': 'SignalPlotDash_{}'.format(request.session['current_user'])}
        reset_relayout_cache(request)
    if reanno:
        return render(request, 'multiuser/reanno.html', context=msg)
    else:
        return render(request, 'multiuser/anno.html', context=msg)


def skip(request):
    current_user = request.session['current_user']

    cwd = os.getcwd()
    if not path.exists(path.join(cwd, 'multiuser/multiuser_outputs/{}/annotation_results.db'.format(current_user))):
        create_annotation_results_db(current_user)

    current_idx = fetch_index(current_user)
    ret = render_plots(request, current_idx + 1)
    update_index(current_idx + 1, current_user)

    conn = sqlite3.connect(path.join(cwd, 'multiuser/multiuser_outputs/{}/annotation_results.db'.format(current_user)))

    c = conn.cursor()
    c.execute("INSERT INTO annotation_results VALUES(?,?,?)", (current_idx, -999, -999))

    conn.commit()
    conn.close()

    reset_relayout_cache(request)
    if not ret:
        msg = {'feedback': 'This dataset is finished', 'data_total': data_total, 'current_user': current_user,
               'progress':   get_progress(current_user), 'finished': True,
               'dash_app': 'SignalPlotDash_{}'.format(request.session['current_user'])}
    else:
        ppg, param = ret
        msg = {'feedback': 'Skipped No. {}'.format(current_idx),
               'param': param, 'current_user': current_user,
               'current_idx': current_idx + 1, 'data_total': data_total,
               'progress':   get_progress(current_user)
            , 'dash_app': 'SignalPlotDash_{}'.format(request.session['current_user'])
               }
    return render(request, 'multiuser/anno.html', context=msg)


def all_artifact(request):
    current_user = request.session['current_user']

    cwd = os.getcwd()
    if not path.exists(path.join(cwd, 'multiuser/multiuser_outputs/{}/annotation_results.db'.format(current_user))):
        create_annotation_results_db(current_user)

    current_idx = fetch_index(current_user)
    ret = render_plots(request, current_idx + 1)
    update_index(current_idx + 1, current_user)

    conn = sqlite3.connect(path.join(cwd, 'multiuser/multiuser_outputs/{}/annotation_results.db'.format(current_user)))

    c = conn.cursor()

    c.execute("INSERT INTO annotation_results VALUES(?,?,?)", (current_idx, 0, 1920 * ((1 / 64) * 1000)))

    conn.commit()
    conn.close()

    reset_relayout_cache(request)

    if not ret:
        msg = {'feedback': 'This dataset is finished', 'data_total': data_total, 'current_user': current_user,
               'progress':   get_progress(current_user), 'finished': True,
               'dash_app': 'SignalPlotDash_{}'.format(request.session['current_user'])}
    else:
        ppg, param = ret
        msg = {'feedback': 'Submitted No. {}'.format(current_idx),
               'param': param, 'current_user': current_user,
               'current_idx': current_idx + 1, 'data_total': data_total,
               'progress':   get_progress(current_user)
            , 'dash_app': 'SignalPlotDash_{}'.format(request.session['current_user'])}
    return render(request, 'multiuser/anno.html', context=msg)


def no_artifact(request):
    current_user = request.session['current_user']

    if not path.exists(
            path.join(os.getcwd(), 'multiuser/multiuser_outputs/{}/annotation_results.db'.format(current_user))):
        create_annotation_results_db(current_user)

    current_idx = fetch_index(current_user)
    ret = render_plots(request, current_idx + 1)
    update_index(current_idx + 1, current_user)

    conn = sqlite3.connect(
        path.join(os.getcwd(), 'multiuser/multiuser_outputs/{}/annotation_results.db'.format(current_user)))

    c = conn.cursor()

    c.execute("INSERT INTO annotation_results VALUES(?,?,?)", (current_idx, 0, 0))

    conn.commit()
    conn.close()

    reset_relayout_cache(request)

    if not ret:
        msg = {'feedback': 'This dataset is finished', 'data_total': data_total, 'current_user': current_user,
               'progress':   get_progress(current_user), 'finished': True,
               'dash_app': 'SignalPlotDash_{}'.format(request.session['current_user'])}
    else:
        ppg, param = ret
        msg = {'feedback': 'Submitted No. {} No artifact'.format(current_idx),
               'param': param, 'current_user': current_user,
               'current_idx': current_idx + 1, 'data_total': data_total,
               'progress':   get_progress(current_user)
            , 'dash_app': 'SignalPlotDash_{}'.format(request.session['current_user'])}
    return render(request, 'multiuser/anno.html', context=msg)


def submit_anno(request):
    current_user = request.session['current_user']
    relay_out_cache = get_relayout_cache(request)

    if len(relay_out_cache) == 0:
        current_idx = fetch_index(current_user)
        ret = render_plots(request, current_idx)
        if not ret:
            msg = {'feedback': 'This dataset is finished', 'data_total': data_total, 'current_idx': current_idx,
                   'progress':   get_progress(current_user), 'finished': True, 'current_user': current_user,
                   'dash_app': 'SignalPlotDash_{}'.format(request.session['current_user'])}
        else:
            ppg, param = ret
            msg = {'feedback': 'Please make annotation before submitting',
                   'param': param, 'current_user': current_user,
                   'current_idx': current_idx, 'data_total': data_total, 'progress':   get_progress(current_user),
                   'dash_app': 'SignalPlotDash_{}'.format(request.session['current_user'])}
        return render(request, 'multiuser/anno.html', context=msg)

    if not path.exists(
            path.join(os.getcwd(), 'multiuser/multiuser_outputs/{}/annotation_results.db'.format(current_user))):
        create_annotation_results_db(current_user)

    current_idx = fetch_index(current_user)
    ret = render_plots(request, current_idx + 1)
    update_index(current_idx + 1, current_user)
    conn = sqlite3.connect(
        path.join(os.getcwd(), 'multiuser/multiuser_outputs/{}/annotation_results.db'.format(current_user)))

    c = conn.cursor()
    for anno in relay_out_cache:
        c.execute("INSERT INTO annotation_results VALUES(?,?,?)", (current_idx, anno[0], anno[1]))

    conn.commit()
    conn.close()

    reset_relayout_cache(request)
    if not ret:
        msg = {'feedback': 'This dataset is finished', 'data_total': data_total, 'current_idx': current_idx,
               'current_user': current_user,
               'progress':   get_progress(current_user), 'finished': True,
               'dash_app': 'SignalPlotDash_{}'.format(request.session['current_user'])}
    else:
        ppg, param = ret
        msg = {'feedback': 'Submitted No. {}'.format(current_idx),
               'param': param, 'current_user': current_user,
               'current_idx': current_idx + 1, 'data_total': data_total, 'progress':   get_progress(current_user)
            , 'dash_app': 'SignalPlotDash_{}'.format(request.session['current_user'])
               }
    return render(request, 'multiuser/anno.html', context=msg)


def export_anno(request):
    current_user = request.session['current_user']
    if not path.exists(
            path.join(os.getcwd(), 'multiuser/multiuser_outputs/{}/annotation_results.db'.format(current_user))):
        create_annotation_results_db(current_user)

    conn = sqlite3.connect(
        path.join(os.getcwd(), 'multiuser/multiuser_outputs/{}/annotation_results.db'.format(current_user)))
    c1 = conn.cursor()
    c1.execute("SELECT idx FROM annotation_results")  # execute a simple SQL select query
    idxs = c1.fetchall()
    result_dict = dict()
    for idx in idxs:
        result_dict[idx[0]] = []

    c2 = conn.cursor()
    c2.execute("SELECT * FROM annotation_results")
    rows = c2.fetchall()

    for row in rows:
        idx, start, end = row
        result_dict[idx].append([start, end])

    conn.close()

    json_path = path.join(os.getcwd(), 'multiuser/multiuser_outputs/{}/result.json'.format(current_user))
    with open(json_path, 'w') as fp:
        json.dump(result_dict, fp)

    with open(json_path, 'rb') as f:
        response = HttpResponse(f.read(), content_type='application/json')
        response['Content-Disposition'] = 'attachment; filename=' + '{}_annotations.json'.format(current_user)
        response['Content-Type'] = 'application/json; charset=utf-16'
        return response


def redo_last(request):
    current_user = request.session['current_user']
    current_idx = fetch_index(current_user)

    if current_idx == 0:
        ret = render_plots(request, current_idx)
        if not ret:
            msg = {'feedback': 'No existing annotation', 'data_total': data_total, 'current_idx': current_idx,
                   'current_user': current_user,
                   'progress':   get_progress(current_user),
                   'dash_app': 'SignalPlotDash_{}'.format(request.session['current_user'])}
        else:
            ppg, param = ret
            msg = {'feedback': 'No existing annotation', 'data_total': data_total, 'current_idx': current_idx,
                   'param': param, 'current_user': current_user,
                   'dash_app': 'SignalPlotDash_{}'.format(request.session['current_user']),
                   'progress':   get_progress(current_user)}
        return render(request, 'multiuser/anno.html', context=msg)

    last_idx = current_idx - 1
    update_index(last_idx, current_user)
    ret = render_plots(request, last_idx)

    data_db1 = sqlite3.connect(
        path.join(os.getcwd(), 'multiuser/multiuser_outputs/{}/annotation_results.db'.format(current_user)))
    c3 = data_db1.cursor()
    c3.execute("DELETE FROM annotation_results WHERE idx>{}".format(last_idx - 1))
    data_db1.commit()
    data_db1.close()

    if not ret:
        msg = {'feedback': 'This dataset is finished', 'data_total': data_total, 'current_idx': last_idx,
               'current_user': current_user,
               'progress':   get_progress(current_user), 'finished': True,
               'dash_app': 'SignalPlotDash_{}'.format(request.session['current_user'])}
    else:
        ppg, param = ret
        msg = {'param': param,
               'feedback': 'Redoing No. {}'.format(last_idx), 'current_user': current_user,
               'dash_app': 'SignalPlotDash_{}'.format(request.session['current_user']),
               'current_idx': last_idx, 'data_total': data_total, 'progress':   get_progress(current_user)}
    return render(request, 'multiuser/anno.html', context=msg)


def logout(request):
    auth.logout(request)
    return render(request, 'multiuser/login.html')


def viz_user(request, additional_feedback=None):
    current_user = request.session['current_user']
    anno_total = fetch_anno_total_user(request)
    current_idx = fetch_viz_index_user(current_user)
    anno_dict = fetch_anno_user(current_user, current_idx, anno_total)
    msg = dict()

    if anno_dict:
        plot_divs = render_result_plots_user(anno_dict)
        msg['divs'] = plot_divs
        msg['displaying'] = '{}-{}'.format(list(anno_dict.keys())[0], list(anno_dict.keys())[-1])
    else:
        msg['feedback'] = 'There are currently no more annotations for {}'.format(current_idx)
    msg['current_user'] = current_user
    if additional_feedback:
        msg['feedback'] = additional_feedback
    return render(request, 'multiuser/viz.html', context=msg)


def next_page_user(request):
    current_user = request.session['current_user']
    anno_total = fetch_anno_total_user(request)
    current_idx = fetch_viz_index_user(current_user)
    next_idx = current_idx + 5
    anno_dict = fetch_anno_user(current_user, next_idx, anno_total)
    msg = dict()

    if anno_dict:
        update_viz_index_user(current_user, next_idx)
        plot_divs = render_result_plots_user(anno_dict)
        msg['divs'] = plot_divs
        msg['displaying'] = '{}-{}'.format(list(anno_dict.keys())[0], list(anno_dict.keys())[-1])
    else:
        anno_dict = fetch_anno_user(current_user, current_idx, anno_total)
        plot_divs = render_result_plots_user(anno_dict)
        msg['divs'] = plot_divs
        msg['displaying'] = '{}-{}'.format(list(anno_dict.keys())[0], list(anno_dict.keys())[-1])
        msg['feedback'] = 'There are currently no more annotations'
    msg['current_user'] = current_user
    return render(request, 'multiuser/viz.html', context=msg)


def last_page_user(request):
    current_user = request.session['current_user']
    anno_total = fetch_anno_total_user(request)
    current_idx = fetch_viz_index_user(current_user)
    msg = dict()

    if current_idx == 0:
        msg['feedback'] = 'There are currently no more annotations'
        target_idx = 0
    else:
        target_idx = max(current_idx - 5, 0)
        update_viz_index_user(current_user, target_idx)

    anno_dict = fetch_anno_user(current_user, target_idx, anno_total)

    if anno_dict:
        plot_divs = render_result_plots_user(anno_dict)
        msg['divs'] = plot_divs
        msg['displaying'] = '{}-{}'.format(list(anno_dict.keys())[0], list(anno_dict.keys())[-1])
    else:
        msg['feedback'] = 'There are currently no more annotations'
    msg['current_user'] = current_user
    return render(request, 'multiuser/viz.html', context=msg)


def inspect_index_user(request):
    current_user = request.session['current_user']
    anno_total = fetch_anno_total_user(request)
    current_idx = fetch_viz_index_user(current_user)

    msg = dict()
    msg['current_user'] = current_user
    inspect_idx = request.POST.get('inspect_index', None)

    if len(inspect_idx) == 0:
        anno_dict = fetch_anno_user(current_user, current_idx, anno_total)
        msg['feedback'] = 'Please enter valid index'
        msg['displaying'] = '{}-{}'.format(list(anno_dict.keys())[0], list(anno_dict.keys())[-1])
        plot_divs = render_result_plots_user(anno_dict)
        msg['divs'] = plot_divs
        return render(request, 'multiuser/viz.html', context=msg)

    inspect_idx = int(inspect_idx)
    if inspect_idx <= anno_total:
        update_viz_index_user(current_user, inspect_idx)
        anno_dict = fetch_anno_user(current_user, inspect_idx, anno_total)

        if anno_dict:
            plot_divs = render_result_plots_user(anno_dict)
            msg['divs'] = plot_divs
            msg['displaying'] = '{}-{}'.format(list(anno_dict.keys())[0], list(anno_dict.keys())[-1])
        else:
            msg['feedback'] = 'There are currently no more annotations'
    else:
        anno_dict = fetch_anno_user(current_user, current_idx, anno_total)
        msg = dict()
        if anno_dict:
            plot_divs = render_result_plots_user(anno_dict)
            msg['divs'] = plot_divs
            msg['displaying'] = '{}-{}'.format(list(anno_dict.keys())[0], list(anno_dict.keys())[-1])
            msg['feedback'] = 'Index {} is not yet annotated'.format(inspect_idx)
        else:
            msg['feedback'] = 'There are currently no more annotations'

    return render(request, 'multiuser/viz.html', context=msg)


def reanno_index_user(request):
    global current_reanno_idx_user
    current_user = request.session['current_user']
    current_idx = fetch_viz_index_user(current_user)

    msg = dict()
    msg['current_user'] = current_user
    reanno_idx = request.POST.get('reanno_index', None)
    anno_total = fetch_anno_total_user(request)
    if len(reanno_idx) == 0:
        anno_dict = fetch_anno_user(current_user, current_idx, anno_total)
        if anno_dict:
            msg['feedback'] = 'Please enter valid index'
            msg['displaying'] = '{}-{}'.format(list(anno_dict.keys())[0], list(anno_dict.keys())[-1])
            plot_divs = render_result_plots_user(anno_dict)
            msg['divs'] = plot_divs
        else:
            msg['feedback'] = 'There are currently no more annotations'
        return render(request, 'multiuser/viz.html', context=msg)

    reanno_idx = int(reanno_idx)
    if reanno_idx <= anno_total:
        anno_dict = fetch_anno_user(current_user, reanno_idx, anno_total)
        current_reanno_idx_user = reanno_idx
        request.session['reanno_back_to_all'] = False
        return reanno_index_helper_user(current_user, request, reanno_idx, anno_dict[current_reanno_idx_user])
    else:
        anno_dict = fetch_anno_user(current_user, current_idx, anno_total)
        if anno_dict:
            plot_divs = render_result_plots_user(anno_dict)
            msg['divs'] = plot_divs
            msg['displaying'] = '{}-{}'.format(list(anno_dict.keys())[0], list(anno_dict.keys())[-1])
            msg['feedback'] = 'Index {} is not yet annotated'.format(reanno_idx)
        else:
            msg['feedback'] = 'There are currently no more annotations'
        msg['feedback'] = 'There are currently no existing annotations for No. {}'.format(reanno_idx)
        return render(request, 'multiuser/viz.html', context=msg)


def reanno_index_helper_user(current_user, request, index, existing_annos):
    current_idx = index
    ret = render_plots(request, current_idx)
    msg = dict()
    msg['current_user'] = current_user
    if not ret:
        msg['current_idx'] = current_idx
    else:
        ppg, param = ret
        msg = {'param': param,
               'feedback': 'Redoing No. {}'.format(current_idx),
               'current_idx': current_idx, 'current_user': current_user,
               'dash_app': 'SignalPlotDash_{}'.format(request.session['current_user'])
               }
    return render(request, 'multiuser/reanno.html', context=msg)


def reanno_clear_anno_user(request):
    return clear_anno(request, reanno=True)


def reanno_all_artifact_user(request):
    current_user = request.session['current_user']

    cwd = os.getcwd()

    current_idx = current_reanno_idx_user

    conn = sqlite3.connect(path.join(cwd, 'multiuser/multiuser_outputs/{}/annotation_results.db'.format(current_user)))

    c = conn.cursor()
    c.execute("DELETE FROM annotation_results WHERE idx={}".format(current_idx))
    c.execute("INSERT INTO annotation_results VALUES(?,?,?)", (current_idx, 0, 1920 * ((1 / 64) * 1000)))

    conn.commit()
    conn.close()
    
    reset_relayout_cache(request)
    if request.session['reanno_back_to_all']:
        get_agreeance(request)
        return viz_all(request, additional_feedback='Re-annotated No. {}'.format(current_reanno_idx_user))
    return viz_user(request, additional_feedback='Re-annotated No. {}'.format(current_reanno_idx_user))


def reanno_no_artifact_user(request):
    current_user = request.session['current_user']

    cwd = os.getcwd()

    current_idx = current_reanno_idx_user

    conn = sqlite3.connect(path.join(cwd, 'multiuser/multiuser_outputs/{}/annotation_results.db'.format(current_user)))

    c = conn.cursor()
    c.execute("DELETE FROM annotation_results WHERE idx={}".format(current_idx))
    c.execute("INSERT INTO annotation_results VALUES(?,?,?)", (current_idx, 0, 0))

    conn.commit()
    conn.close()
    
    reset_relayout_cache(request)
    if request.session['reanno_back_to_all']:
        get_agreeance(request)
        return viz_all(request, additional_feedback='Re-annotated No. {}'.format(current_reanno_idx_user))
    return viz_user(request, additional_feedback='Re-annotated No. {}'.format(current_reanno_idx_user))


def reanno_submit_anno_user(request):
    relay_cache = get_relayout_cache(request)
    current_idx = current_reanno_idx_user
    current_user = request.session['current_user']
    if len(relay_cache) == 0:
        ret = render_plots(request, current_idx)
        if not ret:
            msg = {'feedback': 'This dataset is finished', 'current_idx': current_idx,
                   'current_user': current_user, 'dash_app': 'SignalPlotDash_{}'.format(request.session['current_user'])
                   }
        else:
            ppg, param = ret
            msg = {'feedback': 'Please make annotation before submitting',
                   'param': param, 'current_user': current_user,
                   'dash_app': 'SignalPlotDash_{}'.format(request.session['current_user']),
                   'current_idx': current_idx, }
        return render(request, 'multiuser/reanno.html', context=msg)
    cwd = os.getcwd()
    conn = sqlite3.connect(path.join(cwd, 'multiuser/multiuser_outputs/{}/annotation_results.db'.format(current_user)))

    c = conn.cursor()

    c.execute("DELETE FROM annotation_results WHERE idx={}".format(current_idx))
    for anno in relay_cache:
        c.execute("INSERT INTO annotation_results VALUES(?,?,?)", (current_idx, anno[0], anno[1]))

    conn.commit()
    conn.close()

    reset_relayout_cache(request)
    if request.session['reanno_back_to_all']:
        get_agreeance(request)
        return viz_all(request, additional_feedback='Re-annotated No. {}'.format(current_reanno_idx_user))
    return viz_user(request, additional_feedback='Re-annotated No. {}'.format(current_reanno_idx_user))


def reanno_export_anno_user(request):
    return export_anno(request)


def back_to_viz_user(request):
    return viz_user(request)


def viz_export_anno_user(request):
    return export_anno(request)


def back_to_annotate_user(request):
    return anno_home(request)


def viz_all(request, additional_feedback=None):
    if 'sort' not in request.session.keys():
        request.session['sort'] = 'index'
    current_user = request.session['current_user']
    if check_finished_all():
        get_agreeance(request)
        current_idx = fetch_viz_index_all_user(current_user)
        anno_dict = fetch_anno_all(current_idx, sort=request.session['sort'])
        msg = dict()

        plot_divs = render_result_plots_all(anno_dict)
        msg['divs'] = plot_divs
        msg['displaying'] = '{}-{}'.format(list(anno_dict.keys())[0], list(anno_dict.keys())[-1])
        msg['current_user'] = current_user

        if additional_feedback:
            msg['feedback'] = additional_feedback

        user_color_dict = get_user_colors()
        user_colors = []
        for user, color in user_color_dict.items():
            user_colors.append([user, color])

        msg['user_colors'] = user_colors

        agreeance = request.session['agreeance_iou']
        agreeance_ivt = request.session['agreeance_iou_ivt']
        msg['agreeance'] = '{} | {}'.format(agreeance, agreeance_ivt)
        if request.session['sort'] == 'index':
            msg['sort_by_index'] = 'checked'
        else:
            msg['sort_by_agreeance'] = 'checked'
        return render(request, 'multiuser/viz_all.html', context=msg)
    else:
        current_idx = fetch_index(current_user)
        msg = {'data_total': data_total, 'current_idx': min(current_idx, data_total), 'current_user': current_user,
               'progress':   get_progress(current_user),
               'dash_app': 'SignalPlotDash_{}'.format(request.session['current_user']),
               'check_all_finished_feedback': 'Please wait until everyone is finished'}
        return menu(request, msg)


def next_page_all_user(request):
    current_user = request.session['current_user']
    anno_total = fetch_anno_total_all()
    current_idx = fetch_viz_index_all_user(current_user)
    next_idx = current_idx + 5
    anno_dict = fetch_anno_all(next_idx, sort=request.session['sort'])
    msg = dict()
    agreeance = request.session['agreeance_iou']
    agreeance_ivt = request.session['agreeance_iou_ivt']
    msg['agreeance'] = '{} | {}'.format(agreeance, agreeance_ivt)
    if anno_dict:
        update_viz_index_all_user(current_user, next_idx)
        plot_divs = render_result_plots_all(anno_dict)
        msg['divs'] = plot_divs
        msg['displaying'] = '{}-{}'.format(list(anno_dict.keys())[0], list(anno_dict.keys())[-1])
    else:
        anno_dict = fetch_anno_all(current_idx, sort=request.session['sort'])
        plot_divs = render_result_plots_all(anno_dict)
        msg['divs'] = plot_divs
        msg['displaying'] = '{}-{}'.format(list(anno_dict.keys())[0], list(anno_dict.keys())[-1])
        msg['feedback'] = 'There are currently no more annotations'

    msg['current_user'] = current_user
    user_color_dict = get_user_colors()
    user_colors = []
    for user, color in user_color_dict.items():
        user_colors.append([user, color])

    msg['user_colors'] = user_colors
    if request.session['sort'] == 'index':
        msg['sort_by_index'] = 'checked'
    else:
        msg['sort_by_agreeance'] = 'checked'
    return render(request, 'multiuser/viz_all.html', context=msg)


def last_page_all_user(request):
    current_user = request.session['current_user']
    anno_total = fetch_anno_total_all()
    current_idx = fetch_viz_index_all_user(current_user)
    msg = dict()
    agreeance = request.session['agreeance_iou']
    agreeance_ivt = request.session['agreeance_iou_ivt']
    msg['agreeance'] = '{} | {}'.format(agreeance, agreeance_ivt)
    if current_idx == 0:
        msg['feedback'] = 'There are currently no more annotations'
        target_idx = 0
    else:
        target_idx = max(current_idx - 5, 0)
        update_viz_index_all_user(current_user, target_idx)

    anno_dict = fetch_anno_all(target_idx, sort=request.session['sort'])

    if anno_dict:
        plot_divs = render_result_plots_all(anno_dict)
        msg['divs'] = plot_divs
        msg['displaying'] = '{}-{}'.format(list(anno_dict.keys())[0], list(anno_dict.keys())[-1])
    else:
        msg['feedback'] = 'There are currently no more annotations'
    msg['current_user'] = current_user
    user_color_dict = get_user_colors()
    user_colors = []
    for user, color in user_color_dict.items():
        user_colors.append([user, color])

    msg['user_colors'] = user_colors
    if request.session['sort'] == 'index':
        msg['sort_by_index'] = 'checked'
    else:
        msg['sort_by_agreeance'] = 'checked'
    return render(request, 'multiuser/viz_all.html', context=msg)


def viz_export_anno_all_user(request):
    return export_anno(request)


def inspect_index_all_user(request):
    current_user = request.session['current_user']
    anno_total = fetch_anno_total_all()
    current_idx = fetch_viz_index_all_user(current_user)

    msg = dict()
    msg['current_user'] = current_user
    inspect_idx = request.POST.get('inspect_index', None)
    agreeance = request.session['agreeance_iou']
    agreeance_ivt = request.session['agreeance_iou_ivt']
    msg['agreeance'] = '{} | {}'.format(agreeance, agreeance_ivt)

    if len(inspect_idx) == 0:
        anno_dict = fetch_anno_all(current_idx, sort=request.session['sort'])
        msg['feedback'] = 'Please enter valid index'
        msg['displaying'] = '{}-{}'.format(list(anno_dict.keys())[0], list(anno_dict.keys())[-1])
        plot_divs = render_result_plots_all(anno_dict)
        msg['divs'] = plot_divs
        user_color_dict = get_user_colors()
        user_colors = []
        for user, color in user_color_dict.items():
            user_colors.append([user, color])

        msg['user_colors'] = user_colors
        if request.session['sort'] == 'index':
            msg['sort_by_index'] = 'checked'
        else:
            msg['sort_by_agreeance'] = 'checked'
        return render(request, 'multiuser/viz_all.html', context=msg)

    inspect_idx = int(inspect_idx)
    if inspect_idx <= anno_total:
        update_viz_index_all_user(current_user, inspect_idx)
        anno_dict = fetch_anno_all(inspect_idx, sort=request.session['sort'], inspecting=True)

        plot_divs = render_result_plots_all(anno_dict)
        msg['divs'] = plot_divs
        msg['displaying'] = '{}-{}'.format(list(anno_dict.keys())[0], list(anno_dict.keys())[-1])
    else:
        anno_dict = fetch_anno_all(current_idx, sort=request.session['sort'], inspecting=True)
        msg = dict()
        plot_divs = render_result_plots_all(anno_dict)
        msg['divs'] = plot_divs
        msg['displaying'] = '{}-{}'.format(list(anno_dict.keys())[0], list(anno_dict.keys())[-1])
        msg['feedback'] = 'Index {} is not yet annotated'.format(inspect_idx)
    user_color_dict = get_user_colors()
    user_colors = []
    for user, color in user_color_dict.items():
        user_colors.append([user, color])

    msg['user_colors'] = user_colors
    if request.session['sort'] == 'index':
        msg['sort_by_index'] = 'checked'
    else:
        msg['sort_by_agreeance'] = 'checked'
    return render(request, 'multiuser/viz_all.html', context=msg)


def reanno_index_all_user(request):
    global current_reanno_idx_user
    current_user = request.session['current_user']
    current_idx = fetch_viz_index_all_user(current_user)

    msg = dict()
    agreeance = request.session['agreeance_iou']
    agreeance_ivt = request.session['agreeance_iou_ivt']
    msg['agreeance'] = '{} | {}'.format(agreeance, agreeance_ivt)
    msg['current_user'] = current_user
    reanno_idx = request.POST.get('reanno_index', None)
    anno_total = fetch_anno_total_all()
    if len(reanno_idx) == 0:
        anno_dict = fetch_anno_all(current_idx, sort=request.session['sort'])
        msg['feedback'] = 'Please enter valid index'
        msg['displaying'] = '{}-{}'.format(list(anno_dict.keys())[0], list(anno_dict.keys())[-1])
        plot_divs = render_result_plots_all(anno_dict)
        msg['divs'] = plot_divs
        user_color_dict = get_user_colors()
        user_colors = []
        for user, color in user_color_dict.items():
            user_colors.append([user, color])

        msg['user_colors'] = user_colors
        if request.session['sort'] == 'index':
            msg['sort_by_index'] = 'checked'
        else:
            msg['sort_by_agreeance'] = 'checked'
        return render(request, 'multiuser/viz_all.html', context=msg)

    reanno_idx = int(reanno_idx)
    if reanno_idx <= anno_total:
        anno_dict = fetch_anno_user(current_user, reanno_idx, anno_total)
        current_reanno_idx_user = reanno_idx
        request.session['reanno_back_to_all'] = True
        return reanno_index_helper_user(current_user, request, reanno_idx, anno_dict[current_reanno_idx_user])
    else:
        anno_dict = fetch_anno_all(current_idx, sort=request.session['sort'])
        plot_divs = render_result_plots_all(anno_dict)
        msg['divs'] = plot_divs
        msg['displaying'] = '{}-{}'.format(list(anno_dict.keys())[0], list(anno_dict.keys())[-1])
        msg['feedback'] = 'Index {} is not yet annotated'.format(reanno_idx)
        msg['feedback'] = 'There are currently no existing annotations for No. {}'.format(reanno_idx)
        user_color_dict = get_user_colors()
        user_colors = []
        for user, color in user_color_dict.items():
            user_colors.append([user, color])

        msg['user_colors'] = user_colors
        if request.session['sort'] == 'index':
            msg['sort_by_index'] = 'checked'
        else:
            msg['sort_by_agreeance'] = 'checked'
        return render(request, 'multiuser/viz_all.html', context=msg)


def back_to_viz_all(request):
    return viz_all(request)


def back_to_menu(request):
    username = request.session['current_user']
    current_idx = fetch_index(username)
    msg = {'data_total': data_total, 'current_idx': min(current_idx, data_total), 'current_user': username,
           'progress':   get_progress(username),
           'dash_app': 'SignalPlotDash_{}'.format(request.session['current_user'])}
    return menu(request, msg)


def menu(request, msg):
    check_user_colors()
    return render(request, 'multiuser/cards.html', context=msg)


def back_to_viz_handler(request):
    if request.session['reanno_back_to_all']:
        return viz_all(request)
    else:
        return viz_user(request)


def anno_sorting_handler_all_user(request):
    sort_method = request.POST.get('sort_method', None)
    if sort_method == 'index':
        request.session['sort'] = 'index'
    else:
        request.session['sort'] = 'agreeance'
    update_viz_index_all_user(request.session['current_user'], 0)
    return viz_all(request, additional_feedback='Sorting by {}'.format(sort_method))
