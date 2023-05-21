
import pandas as pd
from IPython.display import HTML
from flask import Flask, render_template, request

from tools.html_selector import HTML_SELECTOR
from tools.plots import PLOT_FUNCTIONS

import os

PATH = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.secret_key = 'database_key'


USER_PICTURE = 'img/yo.jpg'
LOGO = 'img/logo_lion.svg'


@app.route('/data/', methods=['POST', 'GET'])
def data():

    global USER_PICTURE, LOGO, PATH

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

    if request.method == 'POST':
        s_metric = request.form['metric']
        s_plot = request.form['plot']

    else:
        s_metric = 'extracash'
        s_plot = 'demand'

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


@app.route('/', methods=['POST', 'GET'])
def about():
    global USER_PICTURE, LOGO

    return render_template('about.html',
                           user_picture=USER_PICTURE,
                           logo=LOGO,
                           )


if __name__ == '__main__':
    app.run(debug=True, port=5006)
