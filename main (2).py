# -*- coding: utf-8 -*-
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import json
import csv
from collections import defaultdict
from pymongo import MongoClient
from datetime import datetime
import os
from gunicorn.glogging import Logger
import logging

# Custom logger for Gunicorn
class CustomAccessLogger(Logger):
    def access(self, resp, req, environ, request_time):
        user_agent = environ.get('HTTP_USER_AGENT', '')
        path = environ.get('PATH_INFO', '')
        if path.startswith('/static/') or 'Go-http-client' in user_agent:
            return
        super().access(resp, req, environ, request_time)

# Console logging function
def console_log(action: str, message: str):
    print(f"[{action} at {datetime.today().strftime('%Y-%m-%d %H:%M:%S')}] {message}")

# MongoDB setup
MONGO_URI = os.getenv('MONGO_URI', "mongodb://localhost:27017/")
DB_NAME = "namaz_db"
try:
    client = MongoClient(MONGO_URI, server_selection_timeout_ms=5000)
    client.server_info()  # Test connection
    db = client[DB_NAME]
    col_grouped = db['users_grouped']
    col_raw = db['users_raw']
    console_log("DB", "Successful connection to MongoDB")
except Exception as e:
    console_log("Error", f"Failed to connect to MongoDB: {e}")
    raise

# FastAPI and templates setup
app = FastAPI(title="Dynamic Metrics Table")
templates = Jinja2Templates(directory="templates")

@app.get('/', response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})

@app.get('/metrics', response_class=HTMLResponse)
def metrics(request: Request) -> HTMLResponse:
    try:
        learners_df = load_learners('namaz_learners_anon.json')
        logs_df = load_logs('namaz_logs_anon.json')
        update_raw_collection(learners_df)
        outcome_map = build_outcome_map('namaz_outcomes.csv')
        grouped = compute_grouped_size(learners_df)
        retention = compute_retention(logs_df, learners_df)
        engagement = compute_engagement(logs_df, learners_df)
        ctr = compute_ctr(logs_df, learners_df)
        mastery = compute_mastery(logs_df, learners_df, outcome_map)
        final = (
            grouped.merge(retention, on='recommendation_method', how='left')
                   .merge(engagement, on='recommendation_method', how='left')
                   .merge(ctr, on='recommendation_method', how='left')
                   .merge(mastery, on='recommendation_method', how='left')
                   .fillna(0)
        )
        final['CTR'] = final['ctr_clicks'] / final['group_size']
        table = final[['recommendation_method', 'group_size', 'retention', 'engagement', 'CTR', 'mastery_rate']]
        agg_docs = table.to_dict(orient='records')
        col_grouped.delete_many({})
        if agg_docs:
            col_grouped.insert_many(agg_docs)
        html_table = table.to_html(classes='table table-striped', index=False)
        return templates.TemplateResponse('metrics.html', {'request': request, 'table': html_table})
    except Exception as e:
        console_log("Error", f"Error in metrics route: {e}")
        return HTMLResponse(content="Internal Server Error", status_code=500)

# Remaining functions (load_learners, load_logs, etc.) remain unchanged
def load_learners(path: str) -> pd.DataFrame:
    with open(path, encoding='utf-8') as f:
        learners = json.load(f)
    df = pd.DataFrame(learners)
    df['_id'] = df['_id'].astype(str)
    if 'selected' in df.columns:
        df['selected'] = (
            pd.to_numeric(df['selected'], errors='coerce')
              .fillna(0).astype(int)
        )
        df = df[df['selected'] == 0]
    df['launch_count'] = (
        pd.to_numeric(df.get('launch_count', 0), errors='coerce')
          .fillna(0).astype(int)
    )
    return df

def load_logs(path: str) -> pd.DataFrame:
    with open(path, encoding='utf-8-sig') as f:
        logs = json.load(f)
    df = pd.DataFrame(logs)
    df['learner_id'] = df['learner_id'].astype(str)
    return df

def update_raw_collection(df: pd.DataFrame):
    raw_docs = df.to_dict(orient='records')
    col_raw.delete_many({})
    if raw_docs:
        col_raw.insert_many(raw_docs)

def build_outcome_map(csv_path: str) -> dict:
    outcome_map = defaultdict(list)
    with open(csv_path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for assess in filter(None, map(str.strip, row['Assesses'].split(','))):
                outcome_map[row['Outcome ID']].append(assess)
    return outcome_map

def compute_grouped_size(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby('recommendation_method').agg(
        group_size=('_id', 'count')
    ).reset_index()

def compute_retention(logs_df: pd.DataFrame, learners_df: pd.DataFrame) -> pd.DataFrame:
    launch_logs = logs_df[logs_df['activity_id'] == 'launch']
    retention = (
        launch_logs.groupby('learner_id').size().reset_index(name='launch_count')
            .merge(learners_df[['_id', 'recommendation_method']],
                   left_on='learner_id', right_on='_id', how='left')
            .groupby('recommendation_method')['launch_count']
            .mean().reset_index(name='retention')
    )
    return retention

def compute_engagement(logs_df: pd.DataFrame, learners_df: pd.DataFrame) -> pd.DataFrame:
    eng = (
        logs_df.groupby('learner_id').size().reset_index(name='log_count')
            .merge(learners_df[['_id', 'recommendation_method']],
                   left_on='learner_id', right_on='_id', how='left')
            .groupby('recommendation_method')['log_count']
            .mean().reset_index(name='engagement')
    )
    return eng

def compute_ctr(logs_df: pd.DataFrame, learners_df: pd.DataFrame) -> pd.DataFrame:
    ctr_logs = logs_df[logs_df['activity_id'] == 'recommended_item_selected']
    ctr = (
        ctr_logs.groupby('learner_id').size().reset_index(name='ctr_clicks')
            .merge(learners_df[['_id', 'recommendation_method']],
                   left_on='learner_id', right_on='_id', how='left')
            .groupby('recommendation_method')['ctr_clicks']
            .sum().reset_index()
    )
    return ctr

def compute_mastery(logs_df: pd.DataFrame, learners_df: pd.DataFrame, outcome_map: dict) -> pd.DataFrame:
    if 'value' not in logs_df.columns:
        df = learners_df[['recommendation_method']].copy()
        df['mastery_rate'] = 0.0
        return df
    numeric = logs_df[logs_df['value'].apply(
        lambda x: isinstance(x, (int, float, str)) and str(x).replace('.', '', 1).isdigit()
    )].copy()
    numeric['score'] = numeric['value'].astype(float)
    item_score = (
        numeric.groupby(['learner_id', 'activity_id'])['score']
               .max().unstack(fill_value=0)
    )
    item_score['mastery_score'] = item_score.apply(
        lambda row: sum(
            1 for items in outcome_map.values()
            if any(row.get(i, 0) > 0 for i in items)
        ),
        axis=1
    )
    mastery = (
        item_score[['mastery_score']].reset_index()
            .merge(learners_df[['_id', 'recommendation_method']],
                   left_on='learner_id', right_on='_id', how='left')
            .groupby('recommendation_method')['mastery_score']
            .mean().reset_index(name='mastery_rate')
    )
    return mastery

if __name__ == "__main__":
    port = int(os.getenv('PORT', 8000))
    from gunicorn.app.base import Application

    class FastAPIApplication(Application):
        def init(self, parser, opts, args):
            return {
                'bind': f"0.0.0.0:{port}",
                'workers': 2,
                'accesslog': '-',
                'access_log_format': '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"',
                'logger_class': CustomAccessLogger,
            }

        def load(self):
            return app

    FastAPIApplication().run()
