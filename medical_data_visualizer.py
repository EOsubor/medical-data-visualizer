import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv('medical_examination.csv')

# Add 'overweight' column
bmi = (df.weight / ((df.height / 100) ** 2)).astype(int)
bmi_filter = lambda x: 1 if x > 25 else 0
df['overweight'] = bmi.apply(bmi_filter)

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.
filt = lambda x: 0 if x == 1 else 1
df['cholesterol'] = df['cholesterol'].apply(filt)
df['gluc'] = df['gluc'].apply(filt)
#df['cholesterol'] = np.where(df['cholesterol'].values > 1, 1, 0)
#df['gluc'] = np.where(df['gluc'].values > 1, 1, 0)

# Draw Categorical Plot
def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    df_cat = pd.melt(frame=df, value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'], id_vars=['cardio'])


    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the columns for the catplot to work correctly.
    #df_cat = df_cat.groupby(['cardio', 'variable', 'value'], as_index=False).count().rename(columns={'value':'total'}).reset_index()

    # Draw the catplot with 'sns.catplot()'
    g = sns.catplot(x='variable', data=df_cat, hue='value', col='cardio', kind='count').set_axis_labels('variable', 'total')
    fig = g.fig
    
    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig


# Draw Heat Map
def draw_heat_map():
    # Clean the data
    df_heat = df.loc[(df['ap_lo'] <= df['ap_hi']) & 
    (df['height'] >= df['height'].quantile(0.025)) & 
    (df["height"] <= df["height"].quantile(0.975)) & 
    (df["weight"] >= df["weight"].quantile(0.025)) & 
    (df["weight"] <= df["weight"].quantile(0.975))]

    # Calculate the correlation matrix
    corr = df_heat.corr(method='pearson')

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr))



    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 7))

    # Draw the heatmap with 'sns.heatmap()'
    sns.heatmap(corr, mask=mask, annot=True, fmt='0.1f', vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={'shrink':.5})


    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig
