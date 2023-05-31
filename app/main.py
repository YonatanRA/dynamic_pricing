
import pandas as pd
from IPython.display import HTML
from flask import Flask, render_template, request, url_for, redirect, session

from tools.html_selector import HTML_SELECTOR
from tools.plots import PLOT_FUNCTIONS

import os

PATH = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.secret_key = 'database_key'


USER_PICTURE = 'img/yo.jpg'
LOGO = 'img/logo_lion.svg'


# purchase cash y debt


@app.route('/', methods=['POST', 'GET'])
def login():

    if session.get('email') and session.get('password'):
        return redirect('/dashboard', code=302)

    if request.method == 'POST':

        name = request.form['email']
        password = request.form['password']
        # login

        if name != '' and password != '':
            session['email'] = name
            session['password'] = password
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html')

    return render_template('login.html')


@app.route('/logout/', methods=['POST', 'GET'])
def logout():

    session.pop('email', None)
    session.pop('password', None)

    session.clear()
    return redirect(url_for('login'))


@app.route('/overview/', methods=['POST', 'GET'])
def overview():

    if session.get('email') is None or session.get('password') is None:
        return redirect('/', code=302)

    average_daily_jobs = 423352
    n_companies = 390350
    salaries = 186324.61
    average_lifetime_jobs = 5.23

    graphJSON = PLOT_FUNCTIONS['line_overview']()
    plotgraphJSON = PLOT_FUNCTIONS['tree_overview']()
    
    return render_template('overview.html',
                           user_picture=USER_PICTURE,
                           logo=LOGO,
                           n_companies=n_companies,
                           average_daily_jobs=average_daily_jobs,
                           average_lifetime_jobs=average_lifetime_jobs,
                           salaries=salaries,
                           graphJSON=graphJSON,
                           plotgraphJSON=plotgraphJSON)


@app.route('/data/', methods=['POST', 'GET'])
def data():

    global USER_PICTURE, LOGO

    if session.get('email') is None or session.get('password') is None:
        return redirect('/', code=302)

    df = pd.read_parquet(PATH + '/data/sample_data.parquet')

    return render_template('data.html',
                           user_picture=USER_PICTURE,
                           logo=LOGO,
                           tables=[HTML(df.to_html(classes='dataframe',
                                                   header=True,
                                                   index=False,
                                                   escape=False,
                                                   render_links=True
                                                   ))])


@app.route('/demand/', methods=['POST', 'GET'])
def demand():

    global USER_PICTURE, LOGO

    if session.get('email') is None or session.get('password') is None:
        return redirect('/', code=302)

    if request.method == 'POST':
        s_metric = request.form['metric']
        s_plot = request.form['plot']

    else:
        s_metric = 'cash'
        s_plot = 'd/p'

    graph = PLOT_FUNCTIONS[s_plot](s_metric)

    return render_template('demand.html',
                           user_picture=USER_PICTURE,
                           logo=LOGO,
                           graphJSON=graph,
                           metrics=HTML_SELECTOR['METRIC'],
                           metrics_len=len(HTML_SELECTOR['METRIC']),
                           front_metrics=[e.replace('_', ' ')
                                          for e in HTML_SELECTOR['METRIC']],
                           s_metric=s_metric,
                           front_s_metric=s_metric.replace('_', ' '),
                           plots=HTML_SELECTOR['PLOTS'],
                           plots_len=len(HTML_SELECTOR['PLOTS']),
                           front_plots=[e.replace('_', ' ')
                                        for e in HTML_SELECTOR['PLOTS']],
                           s_plot=s_plot,
                           front_s_plot=s_plot.replace('_', ' '),
                           )


@app.route('/about/', methods=['POST', 'GET'])
def about():

    global USER_PICTURE, LOGO

    if session.get('email') is None or session.get('password') is None:
        return redirect('/', code=302)

    return render_template('about.html',
                           user_picture=USER_PICTURE,
                           logo=LOGO,
                           )


@app.route('/clients/', methods=['POST', 'GET'])
def clients():
    global USER_PICTURE, LOGO

    if session.get('email') is None or session.get('password') is None:
        return redirect('/', code=302)

    return render_template('clients.html',
                           user_picture=USER_PICTURE,
                           logo=LOGO,
                           )


if __name__ == '__main__':
    app.run(debug=True, port=5006)
