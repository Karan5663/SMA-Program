import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import missingno as msno
import plotly.express as px
import pandas as pd
import os
import warnings

warnings.filterwarnings("ignore")

# --- Load Dataset ---
def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        print("File not found. Please check the file path.")
        return None
    except pd.errors.EmptyDataError:
        print("File is empty. Please check the file.")
        return None
    except pd.errors.ParserError:
        print("Error parsing the file. Please check the file format.")
        return None

# --- Initial Data Exploration ---
def explore_data(df):
    if df is not None and not df.empty:
        print("First 5 rows of the dataset:")
        print(df.head())
        print("\nMissing values per column:")
        print(df.isnull().sum())
        msno.matrix(df, color=(0.55, 0.55, 0.5))
        plt.show()
    else:
        print("DataFrame is empty or None.")

# --- Clean Missing Data ---
def clean_missing_values(df):
    df['Location'].fillna('Unknown', inplace=True)
    df['Cuisines'].fillna('Unknown', inplace=True)
    df['Rating'].fillna(df['Rating'].mean(), inplace=True)  # Fill with the mean rating
    df['Cost For Two'].fillna(df['Cost For Two'].mean(), inplace=True)  # Fill with the mean cost
    df['Online Order'].fillna('No', inplace=True)  # Assuming 'No' as default
    df['Book Table'].fillna('No', inplace=True)  # Assuming 'No' as default
    return df

# --- Data Cleaning ---
def clean_data(df):
    if df is not None and not df.empty:
        try:
            # Drop unnecessary columns if they exist
            unnecessary_columns = ['url', 'address', 'phone', 'menu_item', 'reviews_list', 'dish_liked']
            df = df.drop([col for col in unnecessary_columns if col in df.columns], axis=1)

            # Clean the 'rate' column
            df['Ratings'] = df['Ratings'].apply(rate_clean)
            df.dropna(inplace=True)

            # Rename columns for better readability
            df.rename(columns={
                'name': 'Restaurant Name',
                'online_order': 'Online Order',
                'book_table': 'Book Table',
                'rate': 'Ratings',
                'votes': 'Votes',
                'location': 'Location',
                'rest_type': 'Type Of Restaurant',
                'cuisines': 'Cuisines',
                'approx_cost(for two people)': 'Cost For Two',
                'listed_in(type)': 'Category',
                'listed_in(city)': 'Listed in City'
            }, inplace=True)

            # Clean special characters in Restaurant Name
            df['Restaurant Name'] = df['Restaurant Name'].str.replace(r'[^\x00-\x7F]+', '', regex=True)

            # Transform 'Cost For Two' to 'Cost Per Head'
            df['Cost For Two'] = df['Cost For Two'].str.replace(',', '').astype(float)
            df['Cost Per Head'] = df['Cost For Two'] / 2
            df.drop(columns=['Cost For Two'], inplace=True)

            return df
        except Exception as e:
            print(f"Error during data cleaning: {e}")
            return df
    else:
        print("DataFrame is empty or None.")
        return df

# --- Rate Column Cleaning ---
def rate_clean(value):
    if value in ['NEW', '-']:
        return np.nan
    else:
        try:
            value = str(value).split('/')[0]
            return float(value)
        except ValueError:
            return np.nan

# --- Aggregate Rare Categories ---
def aggregate_rare_categories(df, column, threshold):
    if df is not None and not df.empty:
        value_counts = df[column].value_counts()
        rare_values = value_counts[value_counts < threshold].index
        df[column] = df[column].apply(lambda x: 'others' if x in rare_values else x)
        return df
    else:
        return df

# --- Visualizations ---
def plot_wordcloud(df, column, title):
    if df is not None and not df.empty:
        text = " ".join(value for value in df[column])
        word_cloud = WordCloud(width=2300, height=800, colormap='jet', background_color="white").generate(text)
        plt.figure(figsize=(50, 8))
        plt.imshow(word_cloud, interpolation="gaussian")
        plt.axis("off")
        plt.title(title)
        plt.show()

def plot_pie_chart(df, column, title):
    if df is not None and not df.empty:
        df[column].value_counts()[:10].plot.pie(figsize=(10, 10), autopct='%1.0f%%')
        plt.title(title)
        plt.show()

def plot_bar_chart(df, x_column, y_column, title):
    if df is not None and not df.empty:
        plt.figure(figsize=(20, 6))
        p = sns.barplot(x=df[x_column], y=df[y_column], palette='tab10', saturation=1, edgecolor="#1c1c1c", linewidth=2)
        p.axes.set_title(title, fontsize=25)
        plt.ylabel("Cost Per Head", fontsize=20)
        plt.xlabel("\n" + x_column, fontsize=20)
        plt.xticks(rotation=90)
        for container in p.containers:
            p.bar_label(container, label_type="center", padding=6, size=15, color="black", rotation=90,
                        bbox={"boxstyle": "round", "pad": 0.6, "facecolor": "orange", "edgecolor": "Brown", "alpha": 1})
        sns.despine(left=True, bottom=True)
        plt.show()

def plot_top_voted_restaurants(df):
    if df is not None and not df.empty:
        vote_dataset = df.copy(deep=True).sort_values(by=['Votes'], ascending=False)
        vote_dataset = vote_dataset.drop_duplicates(subset=['Restaurant Name'], keep='first').reset_index().head(20)

        plt.figure(figsize=(16, 12))
        vote_plt = sns.barplot(x='Votes', y='Restaurant Name', palette='viridis', data=vote_dataset)

        vote_plt.bar_label(vote_plt.containers[0], label=vote_dataset['Votes'])
        plt.title('Top Voted Restaurants in Bangalore')
        plt.show()

def plot_3d_scatter(df):
    if df is not None and not df.empty:
        fig = px.scatter_3d(df, x='Cost Per Head', y='Ratings', z='Votes', color="Cost Per Head", opacity=1,
                            width=700, height=650, color_continuous_scale='dense', title="<b>Cost Per Head</b>")
        fig.update_traces(marker_size=4)
        fig.show()

def plot_heatmaps(df):
    if df is not None and not df.empty:
        yes = df[df['Online Order'] == 'Yes'].describe().T
        no = df[df['Online Order'] == 'No'].describe().T
        colors = ['#2BAE66', 'pink']
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(17, 5))

        plt.subplot(1, 2, 1)
        sns.heatmap(yes[['mean']], annot=True, cmap=colors, linewidths=0.8, linecolor='Red', cbar=False, fmt='.2f')
        plt.title('Mean Values: Online Order')

        plt.subplot(1, 2, 2)
        sns.heatmap(no[['mean']], annot=True, cmap=colors, linewidths=0.8, linecolor='Red', cbar=False, fmt='.2f')
        plt.title('Mean Values: Offline Order')

        fig.tight_layout(pad=2)
        plt.show()

# --- Main Execution ---
def main():
    # Define file path
    filepath = 'C://Users//Karan//OneDrive//Desktop//sma//SMA_PROGRAMS//Exp 3//zomato.csv'

    # Check if file exists
    if os.path.exists(filepath):
        print("File found!")
        df = load_data(filepath)
        if df is not None:
            df = clean_missing_values(df)  # Handle missing values before cleaning
            explore_data(df)
            df = clean_data(df)

            # Aggregate rare categories
            df = aggregate_rare_categories(df, 'Cuisines', 10)
            df = aggregate_rare_categories(df, 'Location', 10)

            # Visualizations
            plot_wordcloud(df, "Cuisines", "Cuisines Word Cloud")
            plot_pie_chart(df, "Cuisines", "Top 10 Cuisines")
            plot_bar_chart(df, "Category", "Cost Per Head", "Top Categories by Cost Per Head")
            plot_top_voted_restaurants(df)
            plot_3d_scatter(df)
            plot_heatmaps(df)
        else:
            print("Data loading failed. Please check the file path and format.")
    else:
        print(f"File not found at {filepath}. Please check the file path.")

# --- Run the Main Function ---
if __name__ == "__main__":
    main()
