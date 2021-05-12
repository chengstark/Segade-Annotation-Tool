from django.shortcuts import render
from django.http import HttpResponse
from os import path
import pickle
import plotly.offline as opy
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
import sqlite3
import dash
import dash_core_components as dcc
import dash_html_components as html
from django_plotly_dash import DjangoDash
import plotly.express as px
from dash.dependencies import Input, Output
import json
import datetime
from scipy import signal
import os
from os import listdir
from itertools import combinations


def get_progress(current_user):
    if not path.exists(
            path.join(os.getcwd(), 'multiuser/multiuser_outputs/{}/annotation_results.db'.format(current_user))):
        return 0

    anno_db_ = sqlite3.connect(
        path.join(os.getcwd(), 'multiuser/multiuser_outputs/{}/annotation_results.db'.format(current_user)))
    c = anno_db_.cursor()
    c.execute("SELECT MAX(idx) FROM annotation_results LIMIT 1")
    anno_total = c.fetchone()[0]
    anno_db_.close()
    data_db_ = sqlite3.connect('ppg_anno_dataset/WESAD.db')
    c = data_db_.cursor()
    c.execute("SELECT MAX(_ROWID_) FROM dataset LIMIT 1")
    data_total = c.fetchone()[0] - 1
    data_db_.commit()
    data_db_.close()
    if not anno_total:
        anno_total = 0
    if not data_total:
        return 0
    return (anno_total / data_total)*100


def update_union_pool(union_pool):
    flag = True
    ret = None
    target_pool = union_pool
    while flag:
        tmp_flag = False
        filtered_union_pool = set()
        for v1 in target_pool:
            sub_flag = False
            start1, end1 = v1
            for v2 in target_pool:
                if v1 != v2:
                    start2, end2 = v2
                    if not (start1 >= end2 or end1 <= start2):
                        filtered_union_pool.add(tuple((min(start1, start2), max(end1, end2))))
                        tmp_flag = True
                        sub_flag = True

            if not sub_flag:
                filtered_union_pool.add(tuple((start1, end1)))
        target_pool = filtered_union_pool
        flag = tmp_flag
        ret = filtered_union_pool
    return ret


def calc_agreeance_iou(anno_dict):
    agreeance_rank_index_dict = dict()
    agreeance_index_agreeance_dict = dict()
    all_agreeance_ious = []
    ppg_idxs = []
    for idx, user_anno_dict in anno_dict.items():
        user_combos = list(combinations(list(user_anno_dict.keys()), 2))
        agreeance_ious = []

        for a, b in user_combos:
            a_annos = user_anno_dict[a]
            b_annos = user_anno_dict[b]

            union_pool = set()
            intersect = 0
            for a_anno in a_annos:
                flag = False
                a_start, a_end = a_anno
                for b_anno in b_annos:
                    b_start, b_end = b_anno
                    if not (a_start >= b_end or a_end <= b_start):
                        union_pool.add(tuple((min(a_start, b_start), max(a_end, b_end))))
                        flag = True
                        intersect += (min(a_end, b_end) - max(a_start, b_start))

                if not flag:
                    union_pool.add(tuple((a_start, a_end)))

            for b_anno in b_annos:
                flag = False
                b_start, b_end = b_anno
                for a_anno in a_annos:
                    a_start, a_end = a_anno
                    if not (a_start >= b_end or a_end <= b_start):
                        flag = True
                    if not flag:
                        union_pool.add(tuple((b_start, b_end)))

            union_pool = update_union_pool(union_pool)
            union = 0
            for start, end in union_pool:
                union += (end - start)
            if union == 0:
                agreeance_ious.append(1)
            else:
                agreeance_ious.append(intersect / union)
        all_agreeance_ious.append(sum(agreeance_ious) / len(agreeance_ious))
        ppg_idxs.append(idx)

    zipped_lists = zip(all_agreeance_ious, ppg_idxs)
    sorted_zipped_lists = sorted(zipped_lists)
    for i, x in enumerate(sorted_zipped_lists):
        agreeance, idx = x
        agreeance_rank_index_dict[i] = idx
        agreeance_index_agreeance_dict[idx] = agreeance

    return agreeance_rank_index_dict, agreeance_index_agreeance_dict, sum(all_agreeance_ious) / len(all_agreeance_ious)


def get_all_anno_dict():
    anno_total = fetch_anno_total_all()
    anno_dict = dict()

    user_db_dict = dict()
    cwd = os.getcwd()
    multi_user_outputs_dir = path.join(cwd, 'multiuser/multiuser_outputs/')
    for username in listdir(multi_user_outputs_dir):
        anno_db = sqlite3.connect(
            path.join(os.getcwd(), 'multiuser/multiuser_outputs/{}/annotation_results.db'.format(username)))
        user_db_dict[username] = anno_db

    for i in range(0, anno_total + 1):
        anno_dict[i] = dict()

    for i in range(0, anno_total + 1):
        for username in listdir(multi_user_outputs_dir):
            annos = []
            anno_db = user_db_dict[username]
            c = anno_db.cursor()
            c.execute("SELECT * FROM annotation_results WHERE idx={}".format(i))
            rows = c.fetchall()
            for row in rows:
                idx, start, end = row
                if start == -999:
                    anno_dict[i][username].append([-999, -999])
                else:
                    start = start / ((1 / 64) * 1000)
                    end = end / ((1 / 64) * 1000)
                    annos.append([start, end])

            annos = sorted(annos)
            union_pool = set()
            intersect = 0
            for a_anno in annos:
                flag = False
                a_start, a_end = a_anno
                for b_anno in annos:
                    if b_anno != a_anno:
                        b_start, b_end = b_anno
                        if not (a_start >= b_end or a_end <= b_start):
                            union_pool.add(tuple((min(a_start, b_start), max(a_end, b_end))))
                            flag = True
                            intersect += (min(a_end, b_end) - max(a_start, b_start))

                if not flag:
                    union_pool.add(tuple((a_start, a_end)))

            cleaned_annos = sorted(list(update_union_pool(union_pool)))

            anno_dict[i][username] = cleaned_annos

    for db in user_db_dict.values():
        db.close()

    return anno_dict


def invert_anno_dict(anno_dict):
    anno_dict_inverted = dict()
    for i in anno_dict.keys():
        anno_dict_inverted[i] = dict()
    for i in anno_dict.keys():
        for username, annos in anno_dict[i].items():
            # add head and tail
            ht_annos = []
            for x in annos:
                s, e = x
                ht_annos.append(s)
                ht_annos.append(e)
            ht_annos.insert(0, 0)
            ht_annos.append(1920)
            inverted_annos = []
            for j in np.arange(0, len(ht_annos)-1, 2):
                if ht_annos[j] == ht_annos[j+1]:
                    continue
                inverted_annos.append([ht_annos[j], ht_annos[j+1]])
            if len(inverted_annos) == 0:
                inverted_annos.append([0, 0])
            anno_dict_inverted[i][username] = inverted_annos
            
    return anno_dict_inverted


def get_agreeance(request):
    anno_dict = get_all_anno_dict()
    anno_dict_ivt = invert_anno_dict(anno_dict)
    agreeance_rank_index_dict, agreeance_index_agreeance_dict, overall_agreeance = calc_agreeance_iou(anno_dict)
    agreeance_rank_index_dict_ivt, agreeance_index_agreeance_dict_ivt, overall_agreeance_ivt = \
        calc_agreeance_iou(anno_dict_ivt)

    cwd = os.getcwd()
    with open(path.join(cwd, 'cache/agreeance_rank_index_dict.json'), 'w') as f:
        json.dump(agreeance_rank_index_dict, f)

    with open(path.join(cwd, 'cache/agreeance_index_agreeance_dict.json'), 'w') as f:
        json.dump(agreeance_index_agreeance_dict, f)
        
    with open(path.join(cwd, 'cache/agreeance_index_dict_ivt.json'), 'w') as f:
        json.dump(agreeance_rank_index_dict_ivt, f)

    with open(path.join(cwd, 'cache/agreeance_index_agreeance_dict_ivt.json'), 'w') as f:
        json.dump(agreeance_index_agreeance_dict_ivt, f)

    request.session['agreeance_iou'] = overall_agreeance
    request.session['agreeance_iou_ivt'] = overall_agreeance_ivt


def calc_time_axis(sample_rate):
    times = []
    for x in np.arange(0, sample_rate * 30, 1):
        t = datetime.timedelta(milliseconds=x * (30 * 1000 / (sample_rate * 30 - 1)))
        minutes, seconds = divmod(t.seconds, 60)
        millis = round(t.microseconds / 1000, 0)
        times.append('{}:{}'.format(seconds, millis))
    return times


def fetch_data(idx):
    data_db = sqlite3.connect('ppg_anno_dataset/WESAD.db')
    c = data_db.cursor()
    c.execute("SELECT * FROM dataset WHERE idx={}".format(idx))
    data_row = c.fetchone()
    data_db.commit()
    data_db.close()
    if data_row is None:
        return False

    ecg, acc0, acc1, acc2, ppg, act, weight, gender, age, height, skin, sport = \
        np.frombuffer(data_row[1], dtype=np.float64), \
        np.frombuffer(data_row[2], dtype=np.float64), \
        np.frombuffer(data_row[3], dtype=np.float64), \
        np.frombuffer(data_row[4], dtype=np.float64), \
        np.frombuffer(data_row[5], dtype=np.float64), \
        np.frombuffer(data_row[6], dtype=np.float64), \
        data_row[7], 'male' if data_row[8] == 0 else 'female', \
        data_row[9], data_row[10], data_row[11], data_row[12]

    param = 'Weight: {}, Gender: {}, Age: {}, Height: {}, Skin: {}, Sport: {}'. \
        format(weight, gender, age, height, skin, sport)

    return ecg, acc0, acc1, acc2, ppg, act, param


def create_annotation_results_db(current_user):
    cwd = os.getcwd()
    conn = sqlite3.connect(path.join(cwd, 'multiuser/multiuser_outputs/{}/annotation_results.db'.format(current_user)))
    c = conn.cursor()
    c.execute('''Create TABLE annotation_results("idx", "start", "end")''')
    conn.commit()
    conn.close()


def fetch_anno_total_user(request):
    current_user = request.session['current_user']
    if not path.exists(
            path.join(os.getcwd(), 'multiuser/multiuser_outputs/{}/annotation_results.db'.format(current_user))):
        create_annotation_results_db(current_user)

    anno_db_ = sqlite3.connect(
        path.join(os.getcwd(), 'multiuser/multiuser_outputs/{}/annotation_results.db'.format(current_user)))
    c = anno_db_.cursor()
    c.execute("SELECT MAX(idx) FROM annotation_results LIMIT 1")
    anno_total = c.fetchone()[0]
    if anno_total is None:
        anno_total = 0
    anno_db_.close()

    return anno_total


def fetch_viz_index_user(current_user):
    cwd = os.getcwd()
    if not path.exists(path.join(cwd, 'multiuser/multiuser_outputs/{}/cache/viz_counter.npy'.format(current_user))):
        np.save(path.join(cwd, 'multiuser/multiuser_outputs/{}/cache/viz_counter.npy'.format(current_user)),
                np.asarray([0]))
        return 0
    else:
        counter = np.load(path.join(cwd, 'multiuser/multiuser_outputs/{}/cache/viz_counter.npy'.format(current_user)))
        return int(counter)


def fetch_anno_user(current_user, current_idx, anno_total):
    anno_db = sqlite3.connect(
        path.join(os.getcwd(), 'multiuser/multiuser_outputs/{}/annotation_results.db'.format(current_user)))
    c = anno_db.cursor()

    anno_dict = dict()
    for i in range(current_idx, min(current_idx + 5, anno_total + 1)):
        anno_dict[i] = []

    flag = False
    for i in range(current_idx, min(current_idx + 5, anno_total + 1)):
        c.execute("SELECT * FROM annotation_results WHERE idx={}".format(i))
        rows = c.fetchall()
        if rows is not None:
            flag = True
        for row in rows:
            idx, start, end = row
            start = start / ((1 / 64) * 1000)
            end = end / ((1 / 64) * 1000)
            anno_dict[i].append([start, end])

    anno_db.close()

    if not flag:
        return False
    else:
        return anno_dict


def render_result_plots_user(anno_dict):
    plot_divs = []
    plot_idx = 0
    for idx, annos in anno_dict.items():
        fig = render_static(idx)
        for anno in annos:
            start, end = anno
            fig.add_vrect(
                x0=start, x1=end,
                fillcolor="#ff7f0e", opacity=0.5,
                layer="below", line_width=0
            )
        if len(annos) == 1:
            start, end = annos[0]
            if start == 0 and end == 30 * 64:
                title = 'Index {}, All Artifact'.format(idx)
            elif start == 0 and end == 0:
                title = 'Index {}, No Artifact'.format(idx)
            elif start == -999 and end == -999:
                title = 'Index {}, Skipped'.format(idx)
            else:
                title = 'Index {}'.format(idx)
        else:
            title = 'Index {}'.format(idx)

        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
        if plot_idx == 0:
            fig.update_layout(margin=dict(t=0, b=15, l=15, r=15), )
        else:
            fig.update_layout(margin=dict(t=0, b=15, l=15, r=15), )
        fig.update_yaxes(tickangle=90, visible=False)
        fig.update_xaxes(showticklabels=False)
        div = opy.plot(fig, auto_open=False, output_type='div')
        plot_divs.append([div, title])

        plot_idx += 1

    return plot_divs


def update_viz_index_user(current_user, idx):
    cwd = os.getcwd()
    np.save(path.join(cwd, 'multiuser/multiuser_outputs/{}/cache/viz_counter.npy'.format(current_user)),
            np.asarray([idx]))


def check_finished_all():
    data_db_ = sqlite3.connect('ppg_anno_dataset/WESAD.db')
    c = data_db_.cursor()
    c.execute("SELECT MAX(_ROWID_) FROM dataset LIMIT 1")
    data_total = c.fetchone()[0] - 1
    data_db_.commit()
    data_db_.close()

    flag = True

    cwd = os.getcwd()
    multi_user_outputs_dir = path.join(cwd, 'multiuser/multiuser_outputs/')
    if len(listdir(multi_user_outputs_dir)) >= 2:
        for username in listdir(multi_user_outputs_dir):
            if not path.exists(
                    path.join(os.getcwd(), 'multiuser/multiuser_outputs/{}/annotation_results.db'.format(username))):
                return False

            anno_db_ = sqlite3.connect(
                path.join(os.getcwd(), 'multiuser/multiuser_outputs/{}/annotation_results.db'.format(username)))
            c = anno_db_.cursor()
            c.execute("SELECT MAX(idx) FROM annotation_results LIMIT 1")
            anno_total = c.fetchone()[0]
            if anno_total is None:
                flag = False
            anno_db_.close()

            if anno_total != data_total:
                flag = False
    else:
        flag = False

    return flag


def fetch_anno_total_all():
    data_db_ = sqlite3.connect('ppg_anno_dataset/WESAD.db')
    c = data_db_.cursor()
    c.execute("SELECT MAX(_ROWID_) FROM dataset LIMIT 1")
    data_total = c.fetchone()[0] - 1
    data_db_.commit()
    data_db_.close()

    return data_total


def fetch_anno_all(current_idx, sort='index', inspecting=False):

    if sort == 'agreeance':
        cwd = os.getcwd()
        with open(path.join(cwd, 'cache/agreeance_rank_index_dict.json'), 'r') as f:
            agreeance_index_dict = json.load(f)

    anno_total = fetch_anno_total_all()
    anno_dict = dict()

    user_db_dict = dict()
    cwd = os.getcwd()
    multi_user_outputs_dir = path.join(cwd, 'multiuser/multiuser_outputs/')
    for username in listdir(multi_user_outputs_dir):
        anno_db = sqlite3.connect(path.join(os.getcwd(),
                                            'multiuser/multiuser_outputs/{}/annotation_results.db'.format(username)))
        user_db_dict[username] = anno_db

    if sort == 'index':
        target_idx = list(range(current_idx, min(current_idx + 5, anno_total + 1)))
    elif sort == 'agreeance' and not inspecting:
        target_idx = []
        for i in range(current_idx, min(current_idx + 5, anno_total + 1)):
            target_idx.append(int(agreeance_index_dict['{}'.format(i)]))
    else:
        target_idx = []
        start_rank = None
        for rank, idx in agreeance_index_dict.items():
            if idx == current_idx:
                start_rank = int(rank)
                break
        for i in range(start_rank, min(start_rank + 5, anno_total + 1)):
            target_idx.append(int(agreeance_index_dict['{}'.format(i)]))

    for i in target_idx:
        anno_dict[i] = dict()

    for i in target_idx:
        for username in listdir(multi_user_outputs_dir):
            anno_dict[i][username] = []

    for i in target_idx:
        for username in listdir(multi_user_outputs_dir):
            anno_db = user_db_dict[username]
            c = anno_db.cursor()
            c.execute("SELECT * FROM annotation_results WHERE idx={}".format(i))
            rows = c.fetchall()
            for row in rows:
                idx, start, end = row
                if start == -999:
                    anno_dict[i][username].append([-999, -999])
                else:
                    start = start / ((1 / 64) * 1000)
                    end = end / ((1 / 64) * 1000)
                    anno_dict[i][username].append([start, end])

    for db in user_db_dict.values():
        db.close()

    return anno_dict


def check_user_colors():
    cwd = os.getcwd()
    multi_user_outputs_dir = path.join(cwd, 'multiuser/multiuser_outputs/')

    if not path.exists(path.join(cwd, 'cache/username_color.json')):
        multi_user_outputs_dir = path.join(cwd, 'multiuser/multiuser_outputs/')
        generate_colors(listdir(multi_user_outputs_dir))

    with open(path.join(cwd, 'cache/username_color.json'), 'r') as f:
        user_color_dict = json.load(f)

    if len(user_color_dict.keys()) != len(listdir(multi_user_outputs_dir)):
        generate_colors(listdir(multi_user_outputs_dir))


def generate_colors(usernames):
    np.random.seed(1)
    cwd = os.getcwd()
    user_color_dict = dict()
    for username in usernames:
        rgb = np.random.uniform(0, 255, size=(3,))
        rgb = np.round(rgb)
        rgb_str = 'rgb({}, {}, {})'.format(rgb[0], rgb[1], rgb[2])
        user_color_dict[username] = rgb_str
    with open(path.join(cwd, 'cache/username_color.json'), 'w') as f:
        json.dump(user_color_dict, f)


def get_user_colors():
    cwd = os.getcwd()
    if not path.exists(path.join(cwd, 'cache/username_color.json')):
        multi_user_outputs_dir = path.join(cwd, 'multiuser/multiuser_outputs/')
        generate_colors(listdir(multi_user_outputs_dir))

    with open(path.join(cwd, 'cache/username_color.json'), 'r') as f:
        return json.load(f)


def update_viz_index_all_user(current_user, idx):
    cwd = os.getcwd()
    np.save(path.join(cwd, 'multiuser/multiuser_outputs/{}/cache/viz_counter_all.npy'.format(current_user)),
            np.asarray([idx]))


def fetch_viz_index_all_user(current_user):
    cwd = os.getcwd()
    if not path.exists(path.join(cwd, 'multiuser/multiuser_outputs/{}/cache/viz_counter_all.npy'.format(current_user))):
        np.save(path.join(cwd, 'multiuser/multiuser_outputs/{}/cache/viz_counter_all.npy'.format(current_user)),
                np.asarray([0]))
        return 0
    else:
        counter = np.load(path.join(cwd,
                                    'multiuser/multiuser_outputs/{}/cache/viz_counter_all.npy'.format(current_user)))
        return int(counter)


def minmax_scale_ppg(ppg):
    ppg_norm = (ppg - min(ppg)) / (max(ppg) - min(ppg))
    return ppg_norm


def scale_ppg(ppg):
    fs = 64
    low_end = 0.9 / (fs / 2)
    high_end = 5 / (fs / 2)
    filter_order = 2

    sos = signal.butter(filter_order, [low_end, high_end], btype='bandpass', output='sos')
    filtered_ppg = signal.sosfilt(sos, ppg)

    ppg_norm = (filtered_ppg - min(filtered_ppg)) / (max(filtered_ppg) - min(filtered_ppg))
    return ppg_norm


def render_static(idx):
    times_64hz = calc_time_axis(64)
    ecg, acc0, acc1, acc2, ppg, act, param = fetch_data(idx)
    ecg = signal.resample(ecg, len(times_64hz))
    acc0 = signal.resample(acc0, len(times_64hz))
    acc1 = signal.resample(acc1, len(times_64hz))
    acc2 = signal.resample(acc2, len(times_64hz))
    act_64 = signal.resample(act, len(times_64hz))
    fig = make_subplots(rows=4, cols=1, vertical_spacing=0.01, shared_xaxes=True, row_heights =[1.2, 0.75, 0.5, 0.5])
    fig.append_trace(go.Scatter(x=times_64hz, y=ppg, mode='lines', name='PPG'), row=1, col=1)
    fig.append_trace(go.Scatter(x=times_64hz, y=ecg, mode='lines', name='ECG'), row=2, col=1)
    fig.append_trace(go.Scatter(x=times_64hz, y=acc0, mode='lines', name='3-Axis_ACC'), row=3, col=1)
    fig.append_trace(go.Scatter(x=times_64hz, y=acc1, mode='lines', name='3-Axis_ACC'), row=3, col=1)
    fig.append_trace(go.Scatter(x=times_64hz, y=acc2, mode='lines', name='3-Axis_ACC'), row=3, col=1)
    fig.append_trace(go.Scatter(x=times_64hz, y=act_64, mode='lines', name='Activity'), row=4, col=1)
    fig['layout']['yaxis4'].update(tickmode='array',
                                   tickvals=list(range(0, 9)),
                                   ticktext=['Transient', 'Sitting', 'Stairs', 'Table soccer',
                                             'Cycling', 'Driving', 'Lunch Break', 'Walking', 'Working'],
                                   )
    fig.update_layout(height=360, width=1000)
    return fig


def render_result_plots_all(anno_dict):
    plot_divs = []
    plot_idx = 0

    cwd = os.getcwd()
    with open(path.join(cwd, 'cache/agreeance_index_agreeance_dict.json'), 'r') as f:
        agreeance_index_agreeance_dict = json.load(f)
        
    with open(path.join(cwd, 'cache/agreeance_index_agreeance_dict_ivt.json'), 'r') as f:
        agreeance_index_agreeance_dict_ivt = json.load(f)

    for idx, user_annos in anno_dict.items():
        fig = render_static(idx)
        user_colors = get_user_colors()
        for username, annos in user_annos.items():
            for anno in annos:
                start, end = anno
                fig.add_vrect(
                    x0=start, x1=end,
                    fillcolor=user_colors[username], opacity=0.3,
                    layer="below", line_width=0
                )

        title = 'Index {} - Agreeance {} | {}'.\
            format(idx, agreeance_index_agreeance_dict[str(idx)], agreeance_index_agreeance_dict_ivt[str(idx)])

        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
        if plot_idx == 0:
            fig.update_layout(margin=dict(t=0, b=15, l=15, r=15), )
        else:
            fig.update_layout(margin=dict(t=0, b=15, l=15, r=15), )
        fig.update_yaxes(tickangle=90, visible=False)
        fig.update_xaxes(showticklabels=False)
        div = opy.plot(fig, auto_open=False, output_type='div')
        plot_divs.append([div, title])

        plot_idx += 1

    return plot_divs





































