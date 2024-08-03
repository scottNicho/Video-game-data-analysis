#!/usr/bin/env python
# coding: utf-8

# In[16]:


import requests
import pandas as pd
from bs4 import BeautifulSoup
import re


# In[17]:


def scrape_main_table(url, table_class):
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to retrieve page with status code {response.status_code}")
        return None

    soup = BeautifulSoup(response.content, 'html.parser')
    tables = soup.find_all('table', {'class': table_class})

    main_table = None
    for table in tables:
        # Check if the first header cell contains 'Title'
        headers = table.find_all('th')
        if headers and 'Title' in headers[0].text:
            main_table = table
            break

    if main_table is None:
        print("No main table found with 'Title' in the first column header")
        return None

    df = pd.read_html(str(main_table))[0]
    return df


# In[ ]:





# In[18]:


def remove_region_initials(date_str):
    if isinstance(date_str, str):
        return re.sub(r'[A-Z]{2,3}$', '', date_str).strip()  # Remove trailing 2 or 3 uppercase letters
    return date_str


# In[ ]:





# In[ ]:





# In[19]:


def convert_dates(date_str):
    if date_str in ['Unreleased', '']:
        return pd.NaT
    try:
        return pd.to_datetime(date_str, errors='coerce')
    except ValueError:
        return pd.NaT


# In[20]:


def clean_dataframe(df, platform_name):
    # Convert column names to strings
    df.columns = df.columns.map(str)
    
    # Identify date columns dynamically
    date_columns = {
        'PAL Release date': None,
        'JP Release date': None,
        'NA Release date': None
    }
    
    for col in df.columns:
        if 'First released' in col:
            df['earliest date'] = df[col].apply(convert_dates)
            df['Platform'] = platform_name
            return df
    
    for col in df.columns:
        if 'PAL' in col or 'EU' in col:
            date_columns['PAL Release date'] = col
        elif 'JP' in col:
            date_columns['JP Release date'] = col
        elif 'NA' in col:
            date_columns['NA Release date'] = col
    
    # Filter and rename columns to standard names
    desired_columns = ['Title', 'PAL Release date', 'JP Release date', 'NA Release date']
    df = df[[col for col in df.columns if col in date_columns.values() or col == 'Title']]
    
    # Apply date conversion
    for key, date_col in date_columns.items():
        if date_col and date_col in df.columns:
            df[date_col] = df[date_col].apply(convert_dates)
    
    # Calculate the earliest date
    date_cols = [date_columns['PAL Release date'], date_columns['JP Release date'], date_columns['NA Release date']]
    date_cols = [col for col in date_cols if col is not None and col in df.columns]
    if date_cols:
        df['earliest date'] = df[date_cols].min(axis=1, skipna=True)
    else:
        df['earliest date'] = None
    
    df['Platform'] = platform_name
    
    # Rename columns to standard names
    new_columns = []
    for col in df.columns:
        if 'PAL' in col or 'EU' in col:
            new_columns.append('PAL Release date')
        elif 'JP' in col:
            new_columns.append('JP Release date')
        elif 'NA' in col:
            new_columns.append('NA Release date')
        elif col == 'earliest date':
            new_columns.append('earliest date')
        elif col == 'Platform':
            new_columns.append('Platform')
        else:
            new_columns.append(col)
    df.columns = new_columns
    
    return df
   


# In[ ]:





# In[21]:


def clean_PS_dataframe(df, platform_name):
    # Convert column names to strings
    df.columns = df.columns.map(str)
    print("Original columns:", df.columns)

    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [' '.join(col).strip() for col in df.columns.values]
        print("Flattened columns:", df.columns)

    # Normalize column names by stripping extra whitespace and lowercasing
    df.columns = df.columns.map(lambda x: ' '.join(x.split()).lower())
    print("Normalized columns:", df.columns)
    
    # Identify date columns dynamically
    date_columns = {
        'PAL Release date': None,
        'JP Release date': None,
        'NA Release date': None
    }
    
    # Check for 'First released' column
    for col in df.columns:
        if 'first released' in col:
            print(f"Found 'First released' column: {col}")
            df['earliest date'] = df[col].apply(convert_dates)
            df['Platform'] = platform_name
            df = df[[c for c in df.columns if 'title' in c or 'earliest date' in c or 'platform' in c]]
            print("Resulting DataFrame with 'First released' column:")
            print(df.head())
            return df

    # Identify other date columns if 'First released' is not found
    for col in df.columns:
        if 'pal' in col or 'eu' in col:
            date_columns['PAL Release date'] = col
        elif 'jp' in col:
            date_columns['JP Release date'] = col
        elif 'na' in col or 'north america' in col:
            date_columns['NA Release date'] = col

    # Print identified date columns
    print("Identified date columns:", date_columns)
    
    # Filter and rename columns to standard names
    print("Columns before filtering:", df.columns)
    desired_columns = ['title'] + list(date_columns.values())
    df = df[[col for col in df.columns if col in desired_columns]]
    print("Columns after filtering:", df.columns)

    # Apply date conversion
    for key, date_col in date_columns.items():
        if date_col and date_col in df.columns:
            print(f"Converting dates for column: {date_col}")
            df[date_col] = df[date_col].apply(convert_dates)

    # Calculate the earliest date
    date_cols = [date_columns['PAL Release date'], date_columns['JP Release date'], date_columns['NA Release date']]
    date_cols = [col for col in date_cols if col is not None and col in df.columns]
    print("Date columns for earliest date calculation:", date_cols)
    if date_cols:
        df['earliest date'] = df[date_cols].min(axis=1, skipna=True)
    else:
        df['earliest date'] = None
    
    df['Platform'] = platform_name
    
    # Rename columns to standard names
    new_columns = []
    for col in df.columns:
        if 'pal' in col or 'eu' in col:
            new_columns.append('PAL Release date')
        elif 'jp' in col:
            new_columns.append('JP Release date')
        elif 'na' in col or 'north america' in col:
            new_columns.append('NA Release date')
        elif col == 'earliest date':
            new_columns.append('earliest date')
        elif col == 'platform':
            new_columns.append('Platform')
        elif 'title' in col:
            new_columns.append('Title')
        else:
            new_columns.append(col)
    df.columns = new_columns

    print("Final dataframe columns:", df.columns)
    print("Final dataframe head:")
    print(df.head())

    return df


# In[ ]:





# In[22]:


Console_Sales_PY = pd.read_excel(r"C:\Users\scott\OneDrive\Excel files\Game consols sales per year.xlsx")


# In[23]:


Console_Sales_PY.head()


# In[ ]:





# In[24]:


Top_50_Games = pd.read_excel(r"C:\Users\scott\OneDrive\Excel files\game analysis.xlsx")


# In[25]:


Top_50_Games.head()


# In[ ]:





# In[195]:


url_xbox = 'https://en.wikipedia.org/wiki/List_of_Xbox_games#0%E2%80%939'
table_class = 'wikitable'  # Change this if the class of the table is different

data_frame_Xbox = scrape_main_table(url_xbox, table_class)

if data_frame_Xbox is not None:
    data_frame_Xbox.to_csv('wikipedia_Xbox.csv', index=False)
    print("Data scraped and saved")
else:
    print("Failed to scrape the table.")


# In[196]:


url_Xbox_collection = {'Xbox':['https://en.wikipedia.org/wiki/List_of_Xbox_games#0%E2%80%939'],
                  'Xbox 360': ['https://en.wikipedia.org/wiki/List_of_Xbox_360_games_(A%E2%80%93L)','https://en.wikipedia.org/wiki/List_of_Xbox_360_games_(M%E2%80%93Z)'],
                 'Xbox One':['https://en.wikipedia.org/wiki/List_of_Xbox_One_games_(A%E2%80%93L)','https://en.wikipedia.org/wiki/List_of_Xbox_One_games_(M%E2%80%93Z)'],
                       'Xbox series X&S':['https://en.wikipedia.org/wiki/List_of_Xbox_Series_X_and_Series_S_games']}
Xbox_dfs = {}
for platform, url_list in url_Xbox_collection.items():
    combined_Xbox_df = pd.DataFrame()
    for url in url_list:
        print(f"Scraping data for {platform} from {url}...")
        df = scrape_main_table(url, 'wikitable')
        if df is not None:
            combined_Xbox_df = pd.concat([combined_Xbox_df, df], ignore_index=True)
        else:
            print(f"Failed to scrape the table for {platform} from {url}.")
    
    if not combined_Xbox_df.empty:
        cleaned_df = clean_dataframe(combined_Xbox_df, platform)
        Xbox_dfs[platform] = cleaned_df
        cleaned_df.to_csv(f'wikipedia_{platform.replace(" ", "_")}.csv', index=False)
        print(f"Data for {platform} scraped and saved.")
    else:
        print(f"No data scraped for {platform}.")


# In[197]:


print(Xbox_dfs['Xbox 360'])


# In[198]:


url_Playstation_collection = {'Playstation 2':['https://en.wikipedia.org/wiki/List_of_PlayStation_2_games_(A%E2%80%93K)','https://en.wikipedia.org/wiki/List_of_PlayStation_2_games_(L%E2%80%93Z)#Games_list_(L%E2%80%93Z)'],
                             'Playstation 3':['https://en.wikipedia.org/wiki/List_of_PlayStation_3_games_(A%E2%80%93C)','https://en.wikipedia.org/wiki/List_of_PlayStation_3_games_(D%E2%80%93I)','https://en.wikipedia.org/wiki/List_of_PlayStation_3_games_(J%E2%80%93P)','https://en.wikipedia.org/wiki/List_of_PlayStation_3_games_(Q%E2%80%93Z)'],
                              'Playstation 4':['https://en.wikipedia.org/wiki/List_of_PlayStation_4_games_(A%E2%80%93L)','https://en.wikipedia.org/wiki/List_of_PlayStation_4_games_(M%E2%80%93Z)'],
                              'Playstation 5':['https://en.wikipedia.org/wiki/List_of_PlayStation_5_games']}
Playstation_dfs = {}
for platform, url_list in url_Playstation_collection.items():
    combined_Playstation_df = pd.DataFrame()
    for url in url_list:
        print(f"Scraping data for {platform} from {url}...")
        df = scrape_main_table(url, 'wikitable')
        if df is not None:
            combined_Playstation_df = pd.concat([combined_Playstation_df, df], ignore_index=True)
        else:
            print(f"Failed to scrape the table for {platform} from {url}.")
    
    if not combined_Playstation_df.empty:
        cleaned_df = clean_PS_dataframe(combined_Playstation_df, platform)
        Playstation_dfs[platform] = cleaned_df
        cleaned_df.to_csv(f'wikipedia_{platform.replace(" ", "_")}.csv', index=False)
        print(f"Data for {platform} scraped and saved.")
    else:
        print(f"No data scraped for {platform}.")


# In[199]:


Playstation_dfs['Playstation 2'] = Playstation_dfs['Playstation 2'][['Title','First released','Platform']]


# In[ ]:


Playstation_dfs['Playstation 2']['First released'] = Playstation_dfs['Playstation 2']['First released'].apply(remove_region_initials)


# In[ ]:


print(Playstation_dfs['Playstation 3'])


# In[ ]:





# In[188]:


url_Nintendo_collection = {'GameCube':['https://en.wikipedia.org/wiki/List_of_GameCube_games'],
                           'Wii':['https://en.wikipedia.org/wiki/List_of_Wii_games'],
                           'Wii U':['https://en.wikipedia.org/wiki/List_of_Wii_U_games'],
                           'Switch':['https://en.wikipedia.org/wiki/List_of_Nintendo_Switch_games_(0%E2%80%93A)','https://en.wikipedia.org/wiki/List_of_Nintendo_Switch_games_(B)',
                                    'https://en.wikipedia.org/wiki/List_of_Nintendo_Switch_games_(C%E2%80%93G)','https://en.wikipedia.org/wiki/List_of_Nintendo_Switch_games_(H%E2%80%93P)',
                                    'https://en.wikipedia.org/wiki/List_of_Nintendo_Switch_games_(Q%E2%80%93Z)']}
Nintendo_dfs = {}
for platform, url_list in url_Nintendo_collection.items():
    combined_Nintendo_df = pd.DataFrame()
    for url in url_list:
        print(f"Scraping data for {platform} from {url}...")
        df = scrape_main_table(url, 'wikitable')
        if df is not None:
            combined_Nintendo_df = pd.concat([combined_Nintendo_df, df], ignore_index=True)
        else:
            print(f"Failed to scrape the table for {platform} from {url}.")
    
    if not combined_Nintendo_df.empty:
        cleaned_df = clean_dataframe(combined_Nintendo_df, platform)
        Nintendo_dfs[platform] = cleaned_df
        cleaned_df.to_csv(f'wikipedia_{platform.replace(" ", "_")}.csv', index=False)
        print(f"Data for {platform} scraped and saved.")
    else:
        print(f"No data scraped for {platform}.")


# In[ ]:





# In[66]:


print(Nintendo_dfs['GameCube'])


# In[ ]:





# In[ ]:





# In[ ]:




