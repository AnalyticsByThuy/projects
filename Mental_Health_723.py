# Import Library

import requests
import re
import os
import csv
import time
import datetime
import pandas as pd
import numpy as np
import seaborn as sns
from bs4 import BeautifulSoup
from urllib.request import urlopen
import matplotlib.pyplot as plt
import spacy
from collections import Counter
import ipywidgets as widgets
from IPython.display import display, HTML
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import folium
import warnings
warnings.filterwarnings("ignore")
from geopy.geocoders import Nominatim
get_ipython().system('pip install pandasql')
get_ipython().system('pip install --upgrade sqlalchemy==1.4.46')
from pandasql import sqldf
get_ipython().system('pip install vaderSentiment')
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'
from dateutil.relativedelta import relativedelta


# Data Collection

def created_modify(pst):
    today = datetime.date.today()
    if 'month' in pst:
        diff = int(pst.split(' ')[0])
        modified_date = today - relativedelta(months=diff)
        return modified_date
    elif 'year' in pst:
        diff = int(pst.split(' ')[0])
        modified_date = today - relativedelta(years=diff)
        return modified_date
    elif 'day' in pst:
        diff = int(pst.split(' ')[0])
        modified_date = today - datetime.timedelta(days=diff)
        return modified_date
    else:
        return today

def data_clean(f):
    f.drop_duplicates().fillna('Not Specified').reset_index(drop=True)
    return f


# Google News


headers = { }
url = 'https://www.google.com/search'
data = []
for page in range(1, 51):
    params = {'q': 'mental health', 'tbm': 'nws', 'tbs': 'qdr:y', 'start': f'{(page-1)*10}'}
    response = requests.get(url, headers=headers, params=params).content
    soup = BeautifulSoup(response, 'lxml')
    for i in soup.find_all('div', class_='SoaBEf'):
        title = i.find('div', class_='n0jPhd ynAwRc MBeuO nDgy9d').text
        site = i.find('div', class_='MgUUmf NUnG9d').text
        content = i.find('div', class_='GI74Re nDgy9d').text
        date = created_modify(i.find('div', class_='OSrXXb rbYSKb LfVVr').text)
        link = i.a['href']
        data.append({'title': title, 'site': site, 'content': content, 'date': date, 'link': link})
        
df = pd.DataFrame(data)
data_clean(df)


# Reddit posts

url = 'https://www.reddit.com/search/'
data = []
for page in range(1, 21):
    params = {'q': 'mental health', 't': 'year',  'after': f'after:t3_{(page-1)*25}'}
    response = requests.get(url, headers=headers, params=params).content
    soup = BeautifulSoup(response, 'lxml')
    subreddit = [i.text for i in soup.select('faceplate-tracker a.text-18')]
    title = [i.text.strip() for i in soup.select('faceplate-tracker a.text-16')]
    date = [i['ts'].split('T')[0] for i in soup.select('faceplate-timeago')]
    comments = [int(i.find_previous_sibling('faceplate-number').get('number')) for i in soup.find_all('span', string='comments')]
    votes = [int(j.find_previous('faceplate-number')['number']) for span_tag in soup.find_all('span', string='comments') for j in span_tag.find_previous_siblings('faceplate-number', limit=1)]
    link = [i['href'] for i in soup.select('a.text-16')]
    for i in range(len(title)):
        data.append({'title': title[i], 'subreddit': subreddit[i], 'link': link[i], 'comments': comments[i], 'votes': votes[i], 'date': date[i]})

rd = pd.DataFrame(data)
data_clean(rd)


# Trend and Sentiment Analysis

## Dataset

analyzer = SentimentIntensityAnalyzer()
df['sentiment'] = df['title'].apply(lambda x: analyzer.polarity_scores(x.lower())['compound'])
rd['sentiment'] = rd['title'].apply(lambda x: analyzer.polarity_scores(x.lower())['compound'])

A = sqldf('''
WITH a AS (
    SELECT strftime('%Y-%m', date) as date, sentiment FROM df
    UNION ALL
    SELECT strftime('%Y-%m', date) as date, sentiment FROM rd
)
select date,
         CASE
           WHEN sentiment > 0 THEN 'Positive'
           WHEN sentiment < 0 THEN 'Negative'
           ELSE 'Neutral'
       END AS sentiment_state,
       COUNT(*) AS count,
       AVG(sentiment) over (partition by date) AS avg_sentiment
from a
group by date, sentiment_state

''', globals())


## Visualization

color_dict = {'Positive': '#3AA6B9', 'Negative': '#FF9EAA', 'Neutral': '#64CCC5'}
def V1():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(A['date'], A['count'], color=[color_dict[state] for state in A['sentiment_state']], label='count')
    ax2 = ax.twinx()
    ax2.plot(A['date'], A['avg_sentiment'], color='#2D4356', marker='o', label='avg_sentiment')

    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color_dict[state]) for state in color_dict.keys()]
    ax.legend(handles=legend_handles, labels=color_dict.keys(), loc='upper left')
    ax2.legend(loc='upper right')

    ax.set_ylabel('Number of Posts')
    ax2.set_ylabel('Average Sentiment')
    ax.set_xlabel('Date of Posts')
    
    ax.tick_params(axis='x', rotation=45)
    fig.suptitle('Trend Analysis of Mental Health-Related Posts on Reddit and Google News (July 2022 - July 2023)', fontsize=11)
    fig.subplots_adjust(bottom=0.2, left=0.05, right=0.95)

    return plt.show()

V1()


# Media Bias Analysis

## Dataset
### Sentiment Bias by Media
A0 = sqldf('''
SELECT source, round(AVG(sentiment), 2) AS avg_polarity, round(bias, 2) as bias
FROM (
    SELECT 'Reddit' AS source, sentiment, (SELECT COUNT(*) FROM rd WHERE sentiment > 0) - (SELECT COUNT(*) FROM rd WHERE sentiment < 0) AS bias
    FROM rd
    UNION ALL
    SELECT 'Google News' AS source, sentiment, (SELECT COUNT(*) FROM df WHERE sentiment > 0) - (SELECT COUNT(*) FROM df WHERE sentiment < 0) AS bias
    FROM df
) AS combined_sentiments
GROUP BY source
''', globals())

### Top User Engagement
A1 = sqldf('''
WITH a AS (
    SELECT subreddit, title, link, (comments+votes) AS engagement_rd, date
    FROM rd
    GROUP BY subreddit, title
    ORDER BY engagement_rd DESC
    LIMIT 5
),
b AS (
    SELECT site, title, link, COUNT(*) AS cnt, date
    FROM df
    GROUP BY site, title
    ORDER BY cnt DESC
    LIMIT 5
)
SELECT a.date, a.title, a.link, a.subreddit as user
FROM a
UNION ALL
SELECT b.date, b.title, b.link, b.site AS user
FROM b
ORDER BY date desc
''', globals())


## Visualization

def V2():
    theta = np.linspace(0, 2*np.pi, len(A0['source']), endpoint=False)
    radii = A0['avg_polarity']
    width = np.pi/3
    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1 = fig.add_subplot(111, polar=True)
    bars = ax1.bar(theta, radii, width=width, alpha=0.8)
    colors = [color_dict['Positive'] if polarity >= 0 else color_dict['Negative'] for polarity in radii]
    for bar, color in zip(bars, colors):
        bar.set_facecolor(color)

    ax1.set_xticks(theta)
    ax1.set_xticklabels(A0['source'], fontsize=9)
    ax1.set_yticklabels([])
    ax1.set_title('Average Sentiment by Media Source', fontsize=9)

    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color_dict['Negative']), plt.Rectangle((0, 0), 1, 1, color=color_dict['Positive'])]
    legend_labels = ['Negative', 'Positive']
    ax1.legend(legend_handles, legend_labels, loc='upper left', fontsize=8)

    ax1.text(0.95, 0.95, f"Reddit Bias: {A0.loc[A0['source']=='Reddit', 'bias'].item()}", transform=ax1.transAxes, ha='right', va='top', fontsize=8, color='black')
    ax1.text(0.95, 0.9, f"Google News Bias: {A0.loc[A0['source']=='Google News', 'bias'].item()}", transform=ax1.transAxes, ha='right', va='top', fontsize=8, color='black')

    plt.tight_layout()
    return plt.show()
V2()


def A1S():
    A1['title'] = A1.apply(lambda row: f'<a href="{row["link"]}">{row["title"]}</a>', axis=1)
    A1.style.set_properties(**{
        'text-align': 'left',
        'font-size': '9px',
        'border-collapse': 'collapse',
        'padding': '2px'
    })
    html_table = A1[['date', 'title', 'user']].to_html(escape=False, index=False, classes=['professional-table'], header=True)
    html_table = html_table.replace('<table', '<table style="border-collapse: collapse; border: 2px solid black; width: 100%;"')
    html_table = html_table.replace('<th', '<th style="text-align: left; padding: 0px; border: 1px solid black;"')
    html_table = html_table.replace('<td', '<td style="border: 1px solid gray; padding: 3px;"')
    html_table = html_table.replace('<td>NaN</td>', '<td style="border: 1px solid gray; padding: 3px;"></td>')  # Handle NaN values

    with open('M_table.html', 'w') as file:
        file.write(html_table)
    return display(HTML(html_table))
A1S()


# Geographic Analysis

## Dataset

B = sqldf('''
select date, lower(title) as title, sentiment from df
union all
select date, lower(title) as title, sentiment from rd
''', globals())

nlp = spacy.load('en_core_web_sm')
concerns = []
kw = []
for content in B['title']:
    doc = nlp(content)
    for entity in doc.ents:
        if entity.label_ not in ['CARDINAL'] and not re.search(r'[^a-zA-Z\s]', entity.text):
            kw.append(entity.text)
        if entity.label_ in ['GPE']:
            concerns.append(entity.text)

#Regional Concerns
concerns_cnt = Counter(concerns)
filtered_concerns = {concern: freq for concern, freq in concerns_cnt.items() if freq > 1}
cns = pd.DataFrame(filtered_concerns.items(), columns=['Regional Concerns', 'Frequency'])

#Top Keywords
kw_cnt = Counter(kw)
filtered_kw = {kw: freq for kw, freq in sorted(kw_cnt.items(), key=lambda x: x[1], reverse=True)[:10]}
kws = pd.DataFrame(filtered_kw.items(), columns=['top_kw', 'Frequency'])

## Visualization

def V3():
    map_center = [0, 0]
    mp = folium.Map(location=map_center, zoom_start=2)
    geolocator = Nominatim(user_agent='my_app')

    for index, row in cns.iterrows():
        location = geolocator.geocode(row['Regional Concerns'])
        if location is not None:
            # Determine the marker color based on frequency
            if row['Frequency'] <= 2:
                marker_color = '#64CCC5'  # Set color for low frequency
            elif row['Frequency'] <= 5:
                marker_color = '#FFD700'  # Set color for medium frequency
            else:
                marker_color = '#FF0000'  # Set color for high frequency

            # Create the label with the frequency and regional concern
            label = f"{row['Frequency']} - {row['Regional Concerns']}"

            folium.Marker([location.latitude, location.longitude], popup=label,
                          icon=folium.Icon(color=marker_color)).add_to(mp)
    return display(mp)

V3()


# Topic Modeling

## Dataset

keywords = sqldf('''
select * from kws
''', globals())


stopwords = ['a', 'the', 'and', 'of', 'in', 'to', 'for', 'on', 'is', 'are', 'that', 'this', 'with', 'as', 'by', 'from', 'at', 'be']
B['title'] = B['title'].apply(lambda x: ' '.join([word for word in x.split() if word not in stopwords]))

vectorizer = CountVectorizer()
dtm = vectorizer.fit_transform(B['title'])
feature_names = np.array(vectorizer.get_feature_names_out())
n_topics = 5
lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
lda.fit(dtm)
def print_topics(model, feature_names, n_top_words):
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        topic_words = ' '.join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        topics.append(topic_words)
    return topics
t = print_topics(lda, feature_names, 10)

## Visualization

def V4():
    plt.axis([0, 10, 0, 10])
    for i in range(len(keywords)):
        x = np.random.randint(0, len(keywords))
        y = np.random.randint(0, len(keywords))
        plt.text(x, y, keywords['top_kw'][i], ha='center', rotation=0, wrap=True, color='black', fontsize=11)
    plt.text(4, 1, t[0], ha='left', rotation=15, wrap=True, color='red', fontsize=10)
    plt.text(6, 5, t[1], ha='left', rotation=15, wrap=True, color='blue', fontsize=10)
    plt.text(5, 5, t[2], ha='right', rotation=-15, wrap=True, color='green', fontsize=10)
    plt.text(5, 10, t[3], fontsize=10, style='oblique', ha='center', va='top', wrap=True, color='orange')
    plt.text(3, 4, t[4], family='serif', style='italic', ha='right', wrap=True, color='purple', fontsize=10)
    
    return plt.show()
V4()


# Deeper Analysis

reddit_quartile = sqldf('''
    with a as (
        SELECT NTILE(4) OVER (ORDER BY comments asc) AS reddit_quartile, comments
        FROM rd
    ),
    b as (
        SELECT reddit_quartile, max(comments) as threshold
        FROM a
        GROUP BY reddit_quartile
    ),
    c as (
        SELECT reddit_quartile, avg(comments) as average
        FROM a
        GROUP BY reddit_quartile
    ),
    d as (
        SELECT a.reddit_quartile, round(100.0 * sum(case when a.comments > c.average then 1 else 0 end) / count(*), 2) as percentage
        FROM a
        JOIN c ON a.reddit_quartile = c.reddit_quartile
        GROUP BY a.reddit_quartile
    )
    SELECT b.reddit_quartile, b.threshold, c.average, d.percentage
    FROM b
    JOIN c ON b.reddit_quartile = c.reddit_quartile
    JOIN d ON b.reddit_quartile = d.reddit_quartile
''', globals())

reddit_quartile_filtered = sqldf('''
    with a as (SELECT NTILE(4) OVER (ORDER BY comments asc) AS reddit_quartile, subreddit from rd)
    select reddit_quartile, subreddit, count(*) as frequency
    from a group by reddit_quartile, subreddit
    having frequency > 15
    order by reddit_quartile, frequency desc;
''', globals())

def DA(tb):
    fig, ax = plt.subplots()
    ax.axis('off')
    table = ax.table(cellText=tb.values, colLabels=tb.columns, cellLoc='left', loc='left')
    table.auto_set_font_size(True)
    for key, cell in table.get_celld().items():
        cell.set_text_props(ha='left')
    return plt.show()

DA(reddit_quartile)
DA(reddit_quartile_filtered)
