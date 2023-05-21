import json
from datetime import date, timedelta

import pandas as pd
import plotly
import plotly.express as px

import bar_chart_race as bcr
import matplotlib as plt
plt.rcParams.update({'figure.facecolor': '#F4F3EE'})

from pathfinder.config import ATLAS_CLIENT, LOCAL_CLIENT, STATES, KEYWORDS

atlas_db = ATLAS_CLIENT.ironcomes
local_db = LOCAL_CLIENT.ironcomes

COLORS = ['#636efa',   # cyber
          '#EF553B',   # data
          '#B6E880',   # game
          '#00cc96',   # marketing
          '#FF97FF',   # mobile
          '#FF6692',   # product
          '#FFA15A',   # ux/ui
          '#19d3f3',   # web
          '#ab63fa',   # web3
          '#dcdcdc',   # gris central (pie chart)
          ]


def extract_bootcamp_from_keyword(x):
    for k, v in KEYWORDS.items():
        if x in v:
            return k


def time_plot(s_country, s_state, s_bootcamp, s_key):

    if s_key == 'All':
        query, filter_ = {'$and': [{'keywords': {'$in': KEYWORDS[s_bootcamp]}},
                                {'country': s_country.lower()},
                                {'date': {'$gte': '2022-04-01'}}  # from april
                                ]}, {'_id': 1, 'date': 1, 'location': 1}
    else:
        query, filter_ = {'$and': [{'keywords': s_key.lower().replace('_', ' ')},
                                {'country': s_country.lower()},
                                {'date': {'$gte': '2022-04-01'}}  # from april
                                ]}, {'_id': 1, 'date': 1, 'location': 1}

    df_plot = pd.DataFrame(atlas_db.jobs.find(query, filter_))

    '''
    # data parquet
    df_plot_parquet = pd.read_parquet(PATH + '/../data/jobs.parquet', engine='fastparquet',
                                      columns=['_id', 'location', 'date', 'keywords', 'country'])

    df_plot_parquet = df_plot_parquet[
        (df_plot_parquet.keywords.isin(KEYWORDS[s_bootcamp])) & (df_plot_parquet.country == s_country.lower())]

    df_plot_parquet.drop(columns=['country', 'keywords'], inplace=True)
    df_plot = pd.concat([df_plot, df_plot_parquet]).sort_values('date', ascending=False).drop_duplicates()
    '''

    # data from local mongodb
    df_plot = pd.concat(
        [df_plot, pd.DataFrame(local_db.jobs.find(query, filter_))]).sort_values('date',
                                                                                 ascending=False).drop_duplicates(subset='_id', keep='last')

    if s_country == 'USA' and s_state != 'All':
        df_plot['state'] = df_plot.location.apply(
            lambda x: x.split(',')[-1].strip())
        df_plot = df_plot[df_plot.state.isin(STATES[s_state])]
    elif s_state != 'All':
        df_plot = df_plot[df_plot.location.str.contains(s_state)]

    df_plot = df_plot.groupby('date').count()['_id'].rolling(window=1).mean()

    df_plot = df_plot.dropna().apply(lambda x: int(x)).reset_index()

    df_plot.columns = ['Date', '# Count']

    if s_state != 'All' and s_key != 'All':
        title = f'Open Positions - {s_bootcamp} - {s_key.replace("_", " ").title()} - {s_country.title()} - {s_state}'
    elif s_state != 'All' and s_key == 'All':
        title = f'Open Positions - {s_bootcamp} - {s_country.title()} - {s_state}'
    elif s_state == 'All' and s_key != 'All':
        title = f'Open Positions - {s_bootcamp} - {s_key.replace("_", " ").title()} - {s_country.title()}'
    else:
        title = f'Open Positions - {s_bootcamp} - {s_country.title()}'

    fig = px.line(df_plot, x='Date', y='# Count', title=title)

    fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                      'paper_bgcolor': 'rgba(0, 0, 0, 0)'})

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def bar_plot(s_country, s_state, s_bootcamp, s_key):

    if s_key == 'All':
        query, filter_ = {'$and': [{'keywords': {'$in': KEYWORDS[s_bootcamp]}},
                                {'country': s_country.lower()},
                                {'date': {'$gte': '2022-04-01'}}  # from april
                                ]}, {'_id': 1, 'date': 1, 'location': 1}
    else:
        query, filter_ = {'$and': [{'keywords': s_key.lower().replace('_', ' ')},
                                {'country': s_country.lower()},
                                {'date': {'$gte': '2022-04-01'}}  # from april
                                ]}, {'_id': 1, 'date': 1, 'location': 1}

    df_plot = pd.DataFrame(atlas_db.jobs.find(query, filter_))

    '''
    # data from parquet
    df_plot_parquet = pd.read_parquet(PATH + '/../data/jobs.parquet', engine='fastparquet',
                                      columns=['_id', 'location', 'date', 'keywords', 'country'])

    df_plot_parquet = df_plot_parquet[
        (df_plot_parquet.keywords.isin(KEYWORDS[s_bootcamp])) & (df_plot_parquet.country == s_country.lower())]

    df_plot_parquet.drop(columns=['country', 'keywords'], inplace=True)
    df_plot = pd.concat([df_plot, df_plot_parquet]).sort_values('date', ascending=False).drop_duplicates()
    '''

    # data from local mongodb
    df_plot = pd.concat(
        [df_plot, pd.DataFrame(local_db.jobs.find(query, filter_))]).sort_values('date',
                                                                                 ascending=False).drop_duplicates(subset='_id', keep='last')
    if s_country == 'USA' and s_state != 'All':
        df_plot['state'] = df_plot.location.apply(
            lambda x: x.split(',')[-1].strip())
        df_plot = df_plot[df_plot.state.isin(STATES[s_state])]
    elif s_state != 'All':
        df_plot = df_plot[df_plot.location.str.contains(s_state)]

    df_plot = df_plot.groupby('date').count()['_id'].rolling(window=1).mean()

    df_plot = df_plot.dropna().apply(lambda x: int(x)).reset_index()

    df_plot.columns = ['Date', '# Count']

    if s_state != 'All' and s_key != 'All':
        title = f'Open Positions - {s_bootcamp} - {s_key.replace("_", " ").title()} - {s_country.title()} - {s_state}'
    elif s_state != 'All' and s_key == 'All':
        title = f'Open Positions - {s_bootcamp} - {s_country.title()} - {s_state}'
    elif s_state == 'All' and s_key != 'All':
        title = f'Open Positions - {s_bootcamp} - {s_key.replace("_", " ").title()} - {s_country.title()}'
    else:
        title = f'Open Positions - {s_bootcamp} - {s_country.title()}'

    fig = px.bar(df_plot, x='Date', y='# Count', title=title)

    fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                      'paper_bgcolor': 'rgba(0, 0, 0, 0)'})

    fig.update_traces(marker_color='#048ab2')

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def time_series_comparative(s_country, s_state, s_bootcamp, s_key):
    today = date.today()

    query, filter_ = {'$and': [{'country': s_country.lower()},
                               {'date': {'$gt': str(
                                   today - timedelta(days=365))}}
                               ]}, {'_id': 1, 'date': 1, 'keywords': 1, 'location': 1}

    df_plot = pd.DataFrame(atlas_db.jobs.find(query, filter_))
    '''
    # data from parquet
    df_plot_parquet = pd.read_parquet(PATH + '/../data/jobs.parquet', engine='fastparquet',
                                      columns=['_id', 'date', 'keywords', 'location', 'country'])

    df_plot_parquet = df_plot_parquet[
        (df_plot_parquet.date > str(today - timedelta(days=90))) & (df_plot_parquet.country == s_country.lower())]

    df_plot_parquet.drop('country', axis=1, inplace=True)
    df_plot = pd.concat([df_plot, df_plot_parquet]).sort_values(
        'date', ascending=False).drop_duplicates()
    '''
    # data fromn local mongodb
    df_plot = pd.concat(
        [df_plot, pd.DataFrame(local_db.jobs.find(query, filter_))]).sort_values('date',
                                                                                 ascending=False).drop_duplicates(subset='_id', keep='last')

    df_plot.drop('_id', axis=1, inplace=True)

    df_plot['bootcamp'] = df_plot.keywords.apply(extract_bootcamp_from_keyword)

    df_plot.dropna(inplace=True)
    df_plot.reset_index(drop=True, inplace=True)

    if s_country == 'USA' and s_state != 'All':
        df_plot['state'] = df_plot.location.apply(
            lambda x: x.split(',')[-1].strip())
        df_plot = df_plot[df_plot.state.isin(STATES[s_state])]
    elif s_state != 'All':
        df_plot = df_plot[df_plot.location.str.contains(s_state)]

    if s_key != 'All':
        df_plot = df_plot[df_plot.bootcamp==s_bootcamp].reset_index()
        df_plot = df_plot.groupby(['date', 'keywords']).count().reset_index()
    else:
        df_plot = df_plot.groupby(['date', 'bootcamp']).count().reset_index()

    df_plot.drop('location', axis=1, inplace=True)

    if len(df_plot.columns) == 3:
        df_plot.columns = ['date', 'bootcamp', '# Count']
    elif len(df_plot.columns) == 4:
        df_plot.columns = ['date', 'bootcamp', '# Count', 'state']

    if s_state != 'All':
        title = f'Comparative by Bootcamp Open Positions Last Year - {s_country} - {s_state}'
    else:
        title = f'Comparative by Bootcamp Open Positions Last Year - {s_country}'

    colors = ['#636efa', '#EF553B', '#00cc96', '#ab63fa',
              '#FFA15A', '#19d3f3', '#FF6692', '#dcdcdc']
    
    fig = px.area(df_plot, x='date', y='# Count',
                  color='bootcamp', line_group='bootcamp', title=title)

    fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                      'paper_bgcolor': 'rgba(0, 0, 0, 0)'})

    fig.update_traces(marker_color='#048ab2', stackgroup=None,
                      fill='tozeroy', hoveron='points+fills')

    fig.data[0]['marker']['color'] = colors

    plot_data = [e for e in fig.data]
    plot_data.sort(key=lambda x: x['legendgroup'])
    
    try:
        for i, e in enumerate(plot_data):
            e['line']['color'] = COLORS[i]
    except:
        pass
    fig.data = plot_data

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def keyword_plot(s_country, s_state, s_bootcamp):
    today = date.today()

    query, filter_ = {'$and': [{'keywords': {'$in': KEYWORDS[s_bootcamp]}},
                               {'country': s_country.lower()},
                               {'date': {'$gt': str(
                                   today - timedelta(days=90))}}
                               ]}, {'_id': 1, 'date': 1, 'location': 1, 'keywords': 1}

    df_plot = pd.DataFrame(atlas_db.jobs.find(query, filter_))

    '''
    # data from parquet
    df_plot_parquet = pd.read_parquet(PATH + '/../data/jobs.parquet', engine='fastparquet',
                                      columns=['_id', 'location', 'date', 'keywords', 'country'])

    df_plot_parquet = df_plot_parquet[
        (df_plot_parquet.keywords.isin(KEYWORDS[s_bootcamp])) & (df_plot_parquet.country == s_country.lower()) & (
            df_plot_parquet.date > str(today - timedelta(days=90)))]

    df_plot_parquet.drop('country', axis=1, inplace=True)
    df_plot = pd.concat([df_plot, df_plot_parquet]).sort_values(
        'date', ascending=False).drop_duplicates()
    '''

    # data fro local mongodb
    df_plot = pd.concat(
        [df_plot, pd.DataFrame(local_db.jobs.find(query, filter_))]).sort_values('date',
                                                                                 ascending=False).drop_duplicates(subset='_id', keep='last')

    if s_country == 'USA' and s_state != 'All':
        df_plot['state'] = df_plot.location.apply(
            lambda x: x.split(',')[-1].strip())
        df_plot = df_plot[df_plot.state.isin(STATES[s_state])]
    elif s_state != 'All':
        df_plot = df_plot[df_plot.location.str.contains(s_state)]

    df_plot = df_plot.groupby('keywords').count().reset_index()[
        ['keywords', '_id']]

    df_plot.columns = ['Keywords', '# Count']

    if s_state != 'All':
        title = f'Open Positions by Keyword Last Quarter - {s_bootcamp} - {s_country.title()} - {s_state}'
    else:
        title = f'Open Positions by Keyword Last Quarter - {s_bootcamp} - {s_country.title()}'

    fig = px.bar(df_plot, x='Keywords', y='# Count', title=title, color='Keywords',
                 color_discrete_sequence=px.colors.sequential.Burgyl_r)

    fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                      'paper_bgcolor': 'rgba(0, 0, 0, 0)'})

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def keyword_race(s_country, s_state, s_bootcamp):
    today = date.today()

    query, filter_ = {'$and': [{'keywords': {'$in': KEYWORDS[s_bootcamp]}},
                               {'country': s_country.lower()},
                               {'date': {'$gt': str(
                                   today - timedelta(days=400))}}
                               ]}, {'_id': 1, 'date': 1, 'location': 1, 'keywords': 1}

    df_plot = pd.DataFrame(atlas_db.jobs.find(query, filter_))

    # data from local mongodb
    df_plot = pd.concat(
        [df_plot, pd.DataFrame(local_db.jobs.find(query, filter_))]).sort_values('date',
                                                                                 ascending=False).drop_duplicates(subset='_id', keep='last').fillna(0)

    if s_country == 'USA' and s_state != 'All':
        df_plot['state'] = df_plot.location.apply(
            lambda x: x.split(',')[-1].strip())
        df_plot = df_plot[df_plot.state.isin(STATES[s_state])]
    elif s_state != 'All':
        df_plot = df_plot[df_plot.location.str.contains(s_state)]

    df_plot.drop('_id', axis=1, inplace=True)
    df_plot.keywords = df_plot.keywords.apply(lambda x: x[:28])

    plot_data = df_plot.groupby(['date', 'keywords']).count().reset_index()
    plot_data = pd.pivot(plot_data, index='date',
                         columns='keywords', values='location')
    plot_data = plot_data.fillna(0)
    plot_data = plot_data.reset_index()
    plot_data.date = pd.to_datetime(plot_data.date)
    plot_data = plot_data.groupby(pd.Grouper(key='date', freq='w')).sum()

    for c in plot_data.columns:
        plot_data[c] = plot_data[c].cumsum()

    raceplot = bcr.bar_chart_race(df=plot_data,
                                  steps_per_period=15,
                                  period_length=700,
                                  interpolate_period=True,
                                  figsize=(8.24, 4.3),
                                  title=f'{s_bootcamp} Roles Evolution - {s_country.title()} - {s_state}',
                                  title_size=10,
                                  period_fmt='%Y-%m-%d',
                                  period_summary_func=lambda values, ranks: {'x': .97,
                                                                             'y': .05,
                                                                             's': f'Total Positions - {int(values.sum())}',
                                                                             'ha': 'right',
                                                                             'size': 8},
                                  )

    return raceplot


def enterprise_plot(s_country, s_state, s_bootcamp, s_key, n_top=15):
    today = date.today()

    if s_key == 'All':
        query, filter_ = {'$and': [{'keywords': {'$in': KEYWORDS[s_bootcamp]}},
                                {'country': s_country.lower()},
                                {'date': {'$gt': str(
                                    today - timedelta(days=90))}}
                                ]}, {'_id': 1, 'date': 1, 'location': 1, 'company_name': 1}

    else:
        query, filter_ = {'$and': [{'keywords': s_key.lower().replace('_', ' ')},
                                {'country': s_country.lower()},
                                {'date': {'$gt': str(
                                    today - timedelta(days=90))}}
                                ]}, {'_id': 1, 'date': 1, 'location': 1, 'company_name': 1} 

    df_plot = pd.DataFrame(atlas_db.jobs.find(query, filter_))

    '''
    # data from parquet
    df_plot_parquet = pd.read_parquet(PATH + '/../data/jobs.parquet', engine='fastparquet',
                                      columns=['_id', 'date', 'location', 'company_name', 'keywords', 'country'])

    df_plot_parquet = df_plot_parquet[
        (df_plot_parquet.date > str(today - timedelta(days=90))) & (df_plot_parquet.country == s_country.lower()) & (
            df_plot_parquet.keywords.isin(KEYWORDS[s_bootcamp]))]

    df_plot_parquet.drop(columns=['country', 'keywords'], inplace=True)
    df_plot = pd.concat([df_plot, df_plot_parquet]).sort_values(
        'date', ascending=False).drop_duplicates()
    '''

    # data from local mongodb
    df_plot = pd.concat(
        [df_plot, pd.DataFrame(local_db.jobs.find(query, filter_))]).sort_values('date',
                                                                                 ascending=False).drop_duplicates(subset='_id', keep='last')

    if s_country == 'USA' and s_state != 'All':
        df_plot['state'] = df_plot.location.apply(
            lambda x: x.split(',')[-1].strip())
        df_plot = df_plot[df_plot.state.isin(STATES[s_state])]
    elif s_state != 'All':
        df_plot = df_plot[df_plot.location.str.contains(s_state)]

    df_plot = df_plot.groupby('company_name').count()
    df_plot = df_plot.reset_index()[['company_name', '_id']].sort_values(
        by='_id', ascending=False).head(n_top)

    df_plot.columns = ['Company', '# Count']

    df_plot.Company = df_plot.Company.apply(lambda x: x.split('-')[0].strip())

    if s_state != 'All' and s_key != 'All':
        title = f'Top {n_top} Posting Companies Last Quarter - {s_bootcamp} - {s_key.replace("_", " ").title()} - {s_country.title()} - {s_state}'
    elif s_state != 'All' and s_key == 'All':
        title = f'Top {n_top} Posting Companies Last Quarter - {s_bootcamp} - {s_country.title()} - {s_state}'
    elif s_state == 'All' and s_key != 'All':
        title = f'Top {n_top} Posting Companies Last Quarter - {s_bootcamp} - {s_key.replace("_", " ").title()} - {s_country.title()}'
    else:
        title = f'Top {n_top} Posting Companies Last Quarter - {s_bootcamp} - {s_country.title()}'

    fig = px.bar(df_plot, x='Company', y='# Count', title=title, color='Company',
                 color_discrete_sequence=px.colors.colorbrewer.Paired)

    fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                      'paper_bgcolor': 'rgba(0, 0, 0, 0)'})

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def treemap(s_country, s_state):
    today = date.today()

    query, filter_ = {'$and': [{'country': s_country.lower()},
                               {'date': {'$gt': str(
                                   today - timedelta(days=90))}}
                               ]}, {'_id': 1, 'company_name': 1, 'date': 1, 'keywords': 1, 'location': 1}

    df_plot = pd.DataFrame(atlas_db.jobs.find(query, filter_))

    '''
    # data fro parquet
    df_plot_parquet = pd.read_parquet(PATH + '/../data/jobs.parquet', engine='fastparquet',
                                      columns=['_id', 'company_name', 'date', 'keywords', 'location', 'country'])

    df_plot_parquet = df_plot_parquet[
        (df_plot_parquet.date > str(today - timedelta(days=90))) & (df_plot_parquet.country == s_country.lower())]

    df_plot_parquet.drop('country', axis=1, inplace=True)
    df_plot = pd.concat([df_plot, df_plot_parquet]).sort_values(
        'date', ascending=False).drop_duplicates()
    '''

    # data from local mongodb
    df_plot = pd.concat(
        [df_plot, pd.DataFrame(local_db.jobs.find(query, filter_))]).drop_duplicates(subset='_id', keep='last')

    df_plot['bootcamp'] = df_plot.keywords.apply(extract_bootcamp_from_keyword)

    df_plot.dropna(inplace=True)
    df_plot.reset_index(drop=True, inplace=True)

    if s_country == 'USA' and s_state != 'All':
        df_plot['state'] = df_plot.location.apply(
            lambda x: x.split(',')[-1].strip())
        df_plot = df_plot[df_plot.state.isin(STATES[s_state])]
    elif s_state != 'All':
        df_plot = df_plot[df_plot.location.str.contains(s_state)]

    df_plot.keywords = df_plot.keywords.apply(lambda x: 'frontend developer' if x=='front-end developer' else x)
    df_plot.keywords = df_plot.keywords.apply(lambda x: 'backend developer' if x=='back-end developer' else x)

    df_plot = df_plot.groupby(['bootcamp', 'keywords']).count().reset_index()

    if s_state != 'All':
        title = f'Bootcamp Treemap Open Positions Last Quarter - {s_country} - {s_state}'
    else:
        title = f'Bootcamp Treemap Open Positions Last Quarter - {s_country}'

    df_plot.rename(columns={'location': 'count'}, inplace=True)

    fig = px.treemap(df_plot, path=[px.Constant('All'), 'bootcamp', 'keywords'], values='count', color='count',
                     color_continuous_scale='pinkyl', width=1200, height=700, title=title)

    fig.update(layout_coloraxis_showscale=True)
    fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                      'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
    fig.update_traces(
        root_color='blue', hovertemplate='bootcamp/role=%{label}<br>count=%{value}<extra></extra>')

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def sunburst(s_country, s_state):
    today = date.today()

    query, filter_ = {'$and': [{'country': s_country.lower()},
                               {'date': {'$gt': str(
                                   today - timedelta(days=90))}}
                               ]}, {'_id': 1, 'company_name': 1, 'date': 1, 'keywords': 1, 'location': 1}

    df_plot = pd.DataFrame(atlas_db.jobs.find(query, filter_))

    '''
    # data from parquet
    df_plot_parquet = pd.read_parquet(PATH + '/../data/jobs.parquet', engine='fastparquet',
                                      columns=['_id', 'company_name', 'date', 'keywords', 'location', 'country'])

    df_plot_parquet = df_plot_parquet[
        (df_plot_parquet.date > str(today - timedelta(days=90))) & (df_plot_parquet.country == s_country.lower())]

    df_plot_parquet.drop('country', axis=1, inplace=True)
    df_plot = pd.concat([df_plot, df_plot_parquet]).sort_values(
        'date', ascending=False).drop_duplicates()
    '''
    # data from local mongodb

    df_plot = pd.concat(
        [df_plot, pd.DataFrame(local_db.jobs.find(query, filter_))]).drop_duplicates(subset='_id', keep='last')

    df_plot['bootcamp'] = df_plot.keywords.apply(extract_bootcamp_from_keyword)

    df_plot.dropna(inplace=True)
    df_plot.reset_index(drop=True, inplace=True)

    if s_country == 'USA' and s_state != 'All':
        df_plot['state'] = df_plot.location.apply(
            lambda x: x.split(',')[-1].strip())
        df_plot = df_plot[df_plot.state.isin(STATES[s_state])]
    elif s_state != 'All':
        df_plot = df_plot[df_plot.location.str.contains(s_state)]

    df_plot.keywords = df_plot.keywords.apply(lambda x: 'frontend developer' if x=='front-end developer' else x)
    df_plot.keywords = df_plot.keywords.apply(lambda x: 'backend developer' if x=='back-end developer' else x)

    df_plot = df_plot.groupby(['bootcamp', 'keywords']).count().reset_index()

    if s_state != 'All':
        title = f'Bootcamp Treemap Open Positions Last Quarter - {s_country} - {s_state}'
    else:
        title = f'Bootcamp Treemap Open Positions Last Quarter - {s_country}'

    df_plot.rename(columns={'location': 'count'}, inplace=True)

    fig = px.sunburst(df_plot, path=[px.Constant('All'), 'bootcamp', 'keywords'], values='count', color='count',
                      color_continuous_scale='pinkyl', width=1200, height=700, title=title)

    fig.update(layout_coloraxis_showscale=True)
    fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                      'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
    fig.update_traces(
        root_color='blue', hovertemplate='bootcamp/role=%{label}<br>count=%{value}<extra></extra>')

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def skills_plot(s_country, s_state, s_bootcamp, s_key='All'):
    today = date.today()

    if s_key != 'All':
        s_key = s_key.lower().replace('_', ' ').strip()

        if s_key == 'uxui designer':
            s_key = 'ux/ui designer'

        query, filter_ = {'$and': [{'keywords': s_key},
                                   {'country': s_country.lower()},
                                   {'date': {'$gt': str(
                                       today - timedelta(days=90))}}
                                   ]}, {'_id': 1, 'date': 1, 'location': 1}

    else:
        query, filter_ = {'$and': [{'keywords': {'$in': KEYWORDS[s_bootcamp]}},
                                   {'country': s_country.lower()},
                                   {'date': {'$gt': str(
                                       today - timedelta(days=90))}}
                                   ]}, {'_id': 1, 'date': 1, 'location': 1}

    df_jobs = pd.DataFrame(atlas_db.jobs.find(query, filter_))

    '''
    # data from parquet
    df_jobs_parquet = pd.read_parquet(PATH + '/../data/jobs.parquet', engine='fastparquet',
                                      columns=['_id', 'location', 'date', 'keywords', 'country'])

    if s_key == 'All':
        df_jobs_parquet = df_jobs_parquet[
            (df_jobs_parquet.keywords.isin(KEYWORDS[s_bootcamp])) & (df_jobs_parquet.country == s_country.lower()) & (
                df_jobs_parquet.date > str(today - timedelta(days=90)))]
    else:
        df_jobs_parquet = df_jobs_parquet[
            (df_jobs_parquet.keywords == s_key) & (df_jobs_parquet.country == s_country.lower()) & (
                df_jobs_parquet.date > str(today - timedelta(days=90)))]

    df_jobs_parquet.drop(columns=['country', 'keywords'], inplace=True)

    df_jobs = pd.concat([df_jobs, df_jobs_parquet]).sort_values(
        'date', ascending=False).drop_duplicates()
    '''

    # data from local mongodb
    df_jobs = pd.concat(
        [df_jobs, pd.DataFrame(local_db.jobs.find(query, filter_))]).sort_values('date',
                                                                                 ascending=False).drop_duplicates(subset='_id', keep='last')

    if s_country == 'USA' and s_state != 'All':
        df_jobs['state'] = df_jobs.location.apply(
            lambda x: x.split(',')[-1].strip())
        df_jobs = df_jobs[df_jobs.state.isin(STATES[s_state])]
    elif s_state != 'All':
        df_jobs = df_jobs[df_jobs.location.str.contains(s_state)]

    atlas_skills = atlas_db.skills.find(
        {'_id': {'$in': df_jobs['_id'].tolist()}})
    df_plot_atlas = pd.DataFrame(atlas_skills)
    df_plot_atlas = df_plot_atlas.explode('skills').groupby('skills').count()
    df_plot_atlas = df_plot_atlas.reset_index(
    )[['skills', '_id']].sort_values(by='_id', ascending=False)

    '''
    # data from parquet
    df_plot_local = pd.read_parquet(
        PATH + '/../data/skills.parquet', engine='fastparquet')
    df_plot_local = df_plot_local[df_plot_local['_id'].isin(
        df_jobs['_id'].tolist())]
    df_plot_local.skills = df_plot_local.skills.apply(
        lambda x: x[1:-1].replace('"', '').split(','))
    df_plot_local = df_plot_local.explode('skills').groupby('skills').count()
    df_plot_local = df_plot_local.reset_index(
    )[['skills', '_id']].sort_values(by='_id', ascending=False)
    '''

    # data from local mongodb
    local_skills = local_db.skills.find(
        {'_id': {'$in': df_jobs['_id'].tolist()}})
    df_plot_local = pd.DataFrame(local_skills)
    df_plot_local = df_plot_local.explode('skills').groupby('skills').count()
    df_plot_local = df_plot_local.reset_index(
    )[['skills', '_id']].sort_values(by='_id', ascending=False)

    df_plot = pd.concat([df_plot_atlas, df_plot_local])
    df_plot.skills = df_plot.skills.apply(
        lambda x: 'excel' if 'excel' in x else x)
    df_plot.skills = df_plot.skills.apply(
        lambda x: 'power bi' if 'power bi' in x else x)
    df_plot.skills = df_plot.skills.apply(
        lambda x: 'postgres' if 'postgres' in x else x)
    df_plot.skills = df_plot.skills.apply(lambda x: 'sql' if 'sql' in x else x)
    df_plot.skills = df_plot.skills.apply(
        lambda x: 'aws' if 'amazon web services' in x else x)
    df_plot.skills = df_plot.skills.apply(lambda x: 'aws' if 's3' in x else x)
    df_plot.skills = df_plot.skills.apply(
        lambda x: 'gcp' if 'google cloud' in x else x)

    df_plot.skills = df_plot.skills.apply(
        lambda x: 'node.js' if 'node' in x else x)
    df_plot.skills = df_plot.skills.apply(
        lambda x: 'go' if 'golang' in x else x)
    df_plot.skills = df_plot.skills.apply(
        lambda x: 'html/css' if 'html' in x else x)
    df_plot.skills = df_plot.skills.apply(
        lambda x: 'html/css' if 'css' in x else x)

    df_plot = df_plot.groupby('skills').sum(
    ).reset_index().sort_values(by='_id', ascending=False)

    df_plot = df_plot[(df_plot['_id'] > 20) & (df_plot['skills'] != 'seo')]

    df_plot.columns = ['Skills', '# Count']

    if s_state == 'All' and s_key == 'All':
        title = f'Top Skills Last Quarter - {s_bootcamp} - {s_country.title()}'
    elif s_state != 'All' and s_key == 'All':
        title = f'Top Skills Last Quarter - {s_bootcamp} - {s_country.title()} - {s_state}'
    elif s_state == 'All' and s_key != 'All':
        title = f'Top Skills Last Quarter - {s_bootcamp} - {s_country.title()} - {s_key.title()}'
    elif s_state != 'All' and s_key != 'All':
        title = f'Top Skills Last Quarter - {s_bootcamp} - {s_key.title()} - {s_country.title()} - {s_state}'

    fig = px.bar(df_plot, x='Skills', y='# Count', title=title, color='Skills',
                 color_discrete_sequence=px.colors.colorbrewer.Paired)

    fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                      'paper_bgcolor': 'rgba(0, 0, 0, 0)'})

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def skills_race(s_country, s_state, s_bootcamp, s_key='All'):
    today = date.today()

    if s_key != 'All':
        s_key = s_key.lower().replace('_', ' ').strip()

        if s_key == 'uxui designer':
            s_key = 'ux/ui designer'

        query, filter_ = {'$and': [{'keywords': s_key},
                                {'country': s_country.lower()},
                                {'date': {'$gt': str(
                                    today - timedelta(days=400))}}
                                ]}, {'_id': 1, 'date': 1, 'location': 1, 'keywords': 1}

    else:
        query, filter_ = {'$and': [{'keywords': {'$in': KEYWORDS[s_bootcamp]}},
                                {'country': s_country.lower()},
                                {'date': {'$gt': str(
                                    today - timedelta(days=400))}}
                                ]}, {'_id': 1, 'date': 1, 'location': 1, 'keywords': 1}

        
    df_jobs = pd.DataFrame(atlas_db.jobs.find(query, filter_))

    # data from local mongodb
    df_jobs = pd.concat(
        [df_jobs, pd.DataFrame(local_db.jobs.find(query, filter_))]).sort_values('date',
                                                                                ascending=False).drop_duplicates(subset='_id', keep='last')

    if s_country == 'USA' and s_state != 'All':
        df_jobs['state'] = df_jobs.location.apply(
            lambda x: x.split(',')[-1].strip())
        df_jobs = df_jobs[df_jobs.state.isin(STATES[s_state])]
    elif s_state != 'All':
        df_jobs = df_jobs[df_jobs.location.str.contains(s_state)]


    atlas_skills = atlas_db.skills.find({'_id': {'$in': df_jobs['_id'].tolist()}})
    df_plot_atlas = pd.DataFrame(atlas_skills)

    # data from local mongodb
    local_skills = local_db.skills.find({'_id': {'$in': df_jobs['_id'].tolist()}})
    df_plot_local = pd.DataFrame(local_skills)

    df_plot = pd.concat([df_plot_atlas, df_plot_local])
    df_plot = df_plot.explode('skills')

    df_plot.skills = df_plot.skills.apply(
        lambda x: 'excel' if 'excel' in x else x)
    df_plot.skills = df_plot.skills.apply(
        lambda x: 'power bi' if 'power bi' in x else x)
    df_plot.skills = df_plot.skills.apply(
        lambda x: 'postgres' if 'postgres' in x else x)
    df_plot.skills = df_plot.skills.apply(lambda x: 'sql' if 'sql' in x else x)
    df_plot.skills = df_plot.skills.apply(
        lambda x: 'aws' if 'amazon web services' in x else x)
    df_plot.skills = df_plot.skills.apply(lambda x: 'aws' if 's3' in x else x)
    df_plot.skills = df_plot.skills.apply(
        lambda x: 'gcp' if 'google cloud' in x else x)

    df_plot.skills = df_plot.skills.apply(
        lambda x: 'node.js' if 'node' in x else x)
    df_plot.skills = df_plot.skills.apply(
        lambda x: 'go' if 'golang' in x else x)
    df_plot.skills = df_plot.skills.apply(
        lambda x: 'html/css' if 'html' in x else x)
    df_plot.skills = df_plot.skills.apply(
        lambda x: 'html/css' if 'css' in x else x)
    

    df_plot = df_jobs.merge(df_plot).drop_duplicates()
    df_plot.skills = df_plot.skills.apply(lambda x: x[:28])

    plot_data = df_plot.groupby(['date', 'skills']).count().reset_index()
    plot_data=pd.pivot(plot_data, index='date', columns='skills', values='location')
    plot_data = plot_data.fillna(0)
    plot_data = plot_data.reset_index()
    plot_data.date=pd.to_datetime(plot_data.date)
    plot_data = plot_data.groupby(pd.Grouper(key='date', freq='w')).sum()

    for c in plot_data.columns:
        plot_data[c]=plot_data[c].cumsum()

    raceplot = bcr.bar_chart_race(df=plot_data,
                                  steps_per_period=15,
                                  period_length=700,
                                  interpolate_period=True,
                                  n_bars=40,
                                  figsize=(8.24, 4.3),
                                  title=f'{s_key.title()} - {s_bootcamp} Skills Evolution - {s_country.title()} - {s_state}',
                                  title_size=10,
                                  bar_label_size=5.2, 
                                  tick_label_size=5.6,
                                  period_fmt='%Y-%m-%d',
                                  )

    return raceplot


def dashboard_comparative(data):

    data = data.groupby(['date', 'bootcamp']).count().reset_index()

    fig = px.area(data, x='date', y='count', color='bootcamp',
                  line_group='bootcamp', width=950, height=550)

    fig.update_yaxes(showticklabels=True, visible=False)
    fig.update_xaxes(showticklabels=True, visible=False)
    fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                      'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
    fig.update_traces(marker_color='#048ab2', stackgroup=None,
                      fill='tozeroy', hoveron='points+fills')

    plot_data = [e for e in fig.data]
    plot_data.sort(key=lambda x: x['legendgroup'])
    for i, e in enumerate(plot_data):
        e['line']['color'] = COLORS[i]

    fig.data = plot_data

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def dashboard_plot(data):

    data = data.groupby('bootcamp').count().reset_index()

    data['avg'] = data['count']/90

    fig = px.sunburst(data, path=[px.Constant('All'), 'bootcamp'], values='count', color='bootcamp',
                      width=550, height=550)

    fig.update(layout_coloraxis_showscale=False)
    fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                      'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
    fig.update_traces(
        hovertemplate='bootcamp=%{customdata[0]}<br>count=%{value}<br>avg=%{hovertext:.2f}<extra></extra>', hovertext=data['avg'])

    fig.data[0]['marker']['colors'] = COLORS

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


PLOT_FUNCTIONS = {'Time_series_line': time_plot,
                  'Time_series_bar': bar_plot,
                  'Comparative': time_series_comparative,
                  'Keywords': keyword_plot,
                  'Keyword_race': keyword_race,
                  'Enterprises': enterprise_plot,
                  'Skills': skills_plot,
                  'Skill_race': skills_race,
                  'Dashboard_comparative': dashboard_comparative,
                  'Treemap': treemap,
                  'Sunburst': sunburst,
                  'Dashboard_plot': dashboard_plot}
