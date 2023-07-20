# Web scraping

import requests
import re
import csv
import time
import datetime
import pandas as pd
import numpy as np
import seaborn as sns
from bs4 import BeautifulSoup
from urllib.request import urlopen
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

headers = {}


skills = ['python', 'power bi', 'jupyter', 'sql', 'tableau', 'excel']
base_url = 'https://www.linkedin.com/jobs/search/?keywords={}&location={}&start={}'
keywords = ['data analyst', 'data scientist']
desired_post = 200
def check_job(location):
    data = []
    for keyword in keywords:
        post_count = 0
        start = 0
        while post_count < desired_post:
            url = base_url.format(keyword, location, start)
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, 'lxml')
            job_cards = soup.select('div.base-card')
            for card in job_cards:
                position = card.find('span', class_='sr-only').text.strip()
                job_url = card.find('a')['href']
                # skills
                job_r = requests.get(job_url, headers=headers).text
                job_soup = BeautifulSoup(job_r, 'lxml')
                ct = [i.text.strip() for i in job_soup.find_all('span', class_='description__job-criteria-text description__job-criteria-text--criteria')]
                level = ct[0] if ct else None
                job_function = ct[2] if ct and len(ct) > 2 else None
                industry = ct[3] if ct and len(ct) > 3 else None
                description_section = job_soup.find('section', class_='show-more-less-html')
                skill_check = ', '.join([skill for skill in skills if description_section and skill in description_section.text.lower() or skill in position.lower()])
                logo = card.find('img')['data-delayed-url']
                company = card.find('a', class_='hidden-nested-link').text.strip()
                city = card.find('span', class_='job-search-card__location').text.strip()
                area = location
                status = card.find('span', class_='result-benefits__text').text.strip() if card.find('span', class_='result-benefits__text') else 'None'
                ld = card.find('time')['datetime']
                dd = card.find('time').text.strip()
                role = keyword
                data.append({'role': role, 'position': position,'job url': job_url, 'Req. Skills': skill_check, 'company': company,'logo': logo,'city': city, 'area': area,'Level': level, 'Job Function':job_function, 'Industry':industry,'list_date': ld,'diff_date': dd,'status': status})
                post_count += 1
                if post_count >= desired_post:
                    break
            start += 25
            time.sleep(2)
    return pd.DataFrame(data)

dfs = pd.concat([check_job(location) for location in ['Vietnam', 'India']], ignore_index=True)

# Data Cleaning

dfs.drop_duplicates(subset=['job url'], keep='first')
dfs.fillna('Not Specified')
dfs = dfs.replace('None', 'Not Specified')
dfs.reset_index(drop=True)


# Save CSV file

#today = datetime.date.today()
#filename = f'analytics_linkedin_{today}.csv'
#dfs.to_csv(filename, index=False)
dfs = pd.read_csv('analytics_linkedin_2023-07-02.csv')


# Data Visualization<
# ## Prep.

get_ipython().system('pip install pandasql')
get_ipython().system('pip install --upgrade sqlalchemy==1.4.46')
from pandasql import sqldf
from IPython.display import display, HTML
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.core.interactiveshell import InteractiveShell
import sqlite3


# ## Dataset

df_skill = pd.DataFrame()
for skill in skills:
    filtered_df_skill = sqldf(f"SELECT [area], [role], [Level], [position],'{skill}' AS [skill] FROM dfs WHERE [Req. Skills] LIKE '%{skill}%'", globals())
    df_skill = pd.concat([df_skill, filtered_df_skill], ignore_index=True)
    
display_df = pd.DataFrame()
display_df['position'] = dfs.apply(lambda row: f'<a href="{row["job url"]}" target="_blank">{row["position"]}</a>', axis=1)
display_df['company'] = dfs.apply(lambda row: f'<div><img src="{row["logo"]}" style="max-height:30px;"></div>{row["company"]}', axis=1)
display_df[['Job Function', 'Industry', 'city', 'area', 'Level', 'list_date', 'status','role']] = dfs[['Job Function', 'Industry', 'city', 'area', 'Level', 'list_date', 'status','role']]


# ## Setting: Dropdown Widgets

roles = list(set(dfs['role']))
areas = list(set(dfs['area']))

# Dropdown Widgets
role_dropdown = widgets.Dropdown(options=['All'] + roles, value='All', description='Roles:')
area_dropdown = widgets.Dropdown(options=['All'] + areas, value='All', description='Areas:')

# Output Widgets
table_output = widgets.Output()
graph_output = widgets.Output()
pie_output = widgets.Output()


# ## Table (HTML)

def update_output(selected_roles, selected_areas):
    table_output.clear_output()
    filtered_df = display_df[display_df['role'].isin(selected_roles) & display_df['area'].isin(selected_areas)]
    styled_table = filtered_df.style.hide(axis="index").set_table_attributes('class="dataframe table"')
    styled_table_html = styled_table.to_html()
    table_with_scroll = f'<div style="overflow: auto; max-height: 400px; white-space: nowrap">{styled_table_html}</div>'
    with table_output:
        display(HTML(table_with_scroll))


# ## Bar Graph: Identify Valuable Skills for Each Role

def update_graph(selected_roles, selected_areas):
    graph_output.clear_output()
    filtered_df = df_skill[df_skill['role'].isin(selected_roles) & df_skill['area'].isin(selected_areas)]
    skill_role_counts = filtered_df.groupby(['skill', 'role']).size().reset_index(name='count')
    data = []
    for idx, role in enumerate(selected_roles):
        role_counts = skill_role_counts[skill_role_counts['role'] == role]
        trace = go.Bar(
            x=role_counts['skill'],
            y=role_counts['count'],
            name=role,
        )
        data.append(trace)
    layout = go.Layout(
        barmode='stack',
        title=dict(text='Skill-Role Relationship'),
        xaxis=dict(title='Skill'),
        yaxis=dict(title='Count'),
    )
    fig = go.Figure(data=data, layout=layout)
    fig.update_layout(height=300)
    with graph_output:
        fig.show()


# ## Pie Graph: Analyze In-Demand Skills

def update_pie(selected_roles, selected_areas):
    pie_output.clear_output()
    filtered_df = df_skill[df_skill['role'].isin(selected_roles) & df_skill['area'].isin(selected_areas)]
    skill_counts = filtered_df.groupby('skill')['position'].count()
    labels = [f"{skill} ({count})" for skill, count in zip(skill_counts.index, skill_counts)]
    explode = [0.1 if skill == skill_counts.index[0] else 0 for skill in skill_counts.index]
    fig = go.Figure(data=[go.Pie(labels=labels, values=skill_counts, hole=0.5)])
    fig.update_layout(title=dict(text='In-Demand Skills'),  height=300)
    with pie_output:
        fig.show()


# ## Initial values

selected_roles = roles
selected_areas = areas


# ## Dropdown Update

def dropdowns_observer(*args):
    selected_roles = [role_dropdown.value] if role_dropdown.value != 'All' else roles
    selected_areas = [area_dropdown.value] if area_dropdown.value != 'All' else areas
    update_output(selected_roles, selected_areas)
    update_graph(selected_roles, selected_areas)
    update_pie(selected_roles, selected_areas)

# Observe the dropdowns' values
role_dropdown.observe(dropdowns_observer, 'value')
area_dropdown.observe(dropdowns_observer, 'value')

# Initial update of the output and graph
update_output(selected_roles, selected_areas)
update_graph(selected_roles, selected_areas)
update_pie(selected_roles, selected_areas)


# ## Creating Layout

grid = widgets.GridspecLayout(2, 2, height='780px', width='1005', layout=widgets.Layout(align_items="center"))
grid[0, 0] = pie_output
grid[0, 1] = graph_output
grid[1, 0:2] = table_output


# ## Creating Dashboard

dashboard_container = widgets.VBox([grid], layout=widgets.Layout(align_items="center"))
widgets.VBox([widgets.HBox([role_dropdown, area_dropdown]), dashboard_container])
