
import pandas as pd
from IPython.display import HTML
from flask import Flask, render_template, request

from tools.html_selector import HTML_SELECTOR
from tools.plots import PLOT_FUNCTIONS


app = Flask(__name__)
app.secret_key = 'database_key'

@app.route('/data/', methods=['POST', 'GET'])
def data():

    df = pd.read_parquet('data/sample_data.parquet')

    return render_template('data.html',
                           user_picture='USER_PICTURE',
                           logo='LOGO',
                           tables=[HTML(df.to_html(classes='dataframe',
                                                    header=True,
                                                    index=False,
                                                    escape=False,
                                                    render_links=True
                                                              ))])


@app.route('/demand/', methods=['POST', 'GET'])
def demand():

    if request.method == 'POST':
        s_metric = request.form['metric']

    else:
        s_metric = 'Time_series_bar'

    graph = ''
        
    return render_template('demand.html',
                           user_picture='USER_PICTURE',
                           logo='LOGO',
                           links='LINKS',
                           graphJSON=graph,
                           metrics=HTML_SELECTOR['METRIC'],
                           metrics_len=len(HTML_SELECTOR['METRIC']),
                           front_metrics=[e.replace('_', ' ')
                                          for e in HTML_SELECTOR['METRIC']],
                           s_metric=s_metric,
                           front_s_metric=s_metric.replace('_', ' '),
                          )

@app.route('/profit/', methods=['POST', 'GET'])
def profit():

    if request.method == 'POST':
        s_metric = request.form['metric']

    else:
        s_metric = 'Time_series_bar'

    graph = ''
        
    return render_template('profit.html',
                           user_picture='USER_PICTURE',
                           logo='LOGO',
                           links='LINKS',
                           graphJSON=graph,
                           metrics=HTML_SELECTOR['METRIC'],
                           metrics_len=len(HTML_SELECTOR['METRIC']),
                           front_metrics=[e.replace('_', ' ')
                                          for e in HTML_SELECTOR['METRIC']],
                           s_metric=s_metric,
                           front_s_metric=s_metric.replace('_', ' '),
                          )

@app.route('/', methods=['POST', 'GET'])
def about():
    global USER_PICTURE, LOGO, LINKS

    return render_template('about.html',
                           user_picture='USER_PICTURE',
                           logo='LOGO',
                           )


if __name__ == '__main__':
    app.run(debug=True, port=5006)
