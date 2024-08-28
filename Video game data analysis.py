#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
import pandas as pd
from io import StringIO
from bs4 import BeautifulSoup
import seaborn as sns
import matplotlib.pyplot as plt
import re
import matplotlib.dates as mdates
from scipy.stats import gaussian_kde


# In[2]:


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

    # Wrap main_table with StringIO before passing to pd.read_html
    df = pd.read_html(StringIO(str(main_table)))[0]
    return df


# In[ ]:





# In[3]:


def remove_region_initials(date_str):
    if isinstance(date_str, str):
        return re.sub(r'[A-Z]{2,3}$', '', date_str).strip()  # Remove trailing 2 or 3 uppercase letters
    return date_str


# In[ ]:





# In[ ]:





# In[4]:


def convert_dates(date_str):
    if date_str in ['Unreleased', '']:
        return pd.NaT
    try:
        return pd.to_datetime(date_str, errors='coerce')
    except ValueError:
        return pd.NaT


# In[5]:


def add_first_released_column(df):
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
    
    # Identify date columns
    for col in df.columns:
        if 'pal' in col or 'eu' in col:
            date_columns['PAL Release date'] = col
        elif 'jp' in col:
            date_columns['JP Release date'] = col
        elif 'na' in col or 'north america' in col:
            date_columns['NA Release date'] = col

    # Print identified date columns
    print("Identified date columns:", date_columns)
    
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
        df['First released'] = df[date_cols].min(axis=1, skipna=True)
    
    # Display the dataframe with the new 'First released' column
    print("DataFrame with 'First released' column added:")
    print(df.head())

    return df


# In[6]:


def add_first_released_column_WiiU(df):
    # Convert column names to strings
    df.columns = df.columns.map(str)
    print("Original columns:", df.columns)

    # Normalize column names by stripping extra whitespace and lowercasing
    df.columns = df.columns.map(lambda x: ' '.join(x.split()).lower())
    print("Normalized columns:", df.columns)
    
    # Identify date columns dynamically
    date_columns = {
        'PAL Release date': None,
        'JP Release date': None,
        'NA Release date': None
    }
    
    # Identify date columns
    for col in df.columns:
        if 'pal' in col or 'eu' in col:
            date_columns['PAL Release date'] = col
        elif 'jp' in col:
            date_columns['JP Release date'] = col
        elif 'na' in col or 'north america' in col:
            date_columns['NA Release date'] = col

    # Print identified date columns
    print("Identified date columns:", date_columns)
    
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
        df['First released'] = df[date_cols].min(axis=1, skipna=True)
    
    # Display the dataframe with the new 'First released' column
    print("DataFrame with 'First released' column added:")
    print(df.head())

    return df


# In[7]:


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
        df['First released'] = df[date_cols].min(axis=1, skipna=True)
    else:
        df['First released'] = None
    
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
        elif col == 'First released':
            new_columns.append('First released')
        elif col == 'Platform':
            new_columns.append('Platform')
        else:
            new_columns.append(col)
    df.columns = new_columns
    
    return df
   


# In[ ]:





# In[8]:


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





# In[9]:


Console_Sales_PY = pd.read_excel(r"C:\Users\scott\OneDrive\Excel files\Game consols sales per year.xlsx")


# In[10]:


Console_Sales_PY.head()


# In[ ]:





# In[11]:


Top_50_Games = pd.read_excel(r"C:\Users\scott\OneDrive\Excel files\game analysis.xlsx")


# In[12]:


Top_50_Games.head()


# In[ ]:





# In[13]:


url_xbox = 'https://en.wikipedia.org/wiki/List_of_Xbox_games#0%E2%80%939'
table_class = 'wikitable'  # Change this if the class of the table is different

data_frame_Xbox = scrape_main_table(url_xbox, table_class)

if data_frame_Xbox is not None:
    data_frame_Xbox.to_csv('wikipedia_Xbox.csv', index=False)
    print("Data scraped and saved")
else:
    print("Failed to scrape the table.")


# In[14]:


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
        Xbox_dfs[platform]=  Xbox_dfs[platform][['First released','Platform']]
        cleaned_df.to_csv(f'wikipedia_{platform.replace(" ", "_")}.csv', index=False)
        print(f"Data for {platform} scraped and saved.")
    else:
        print(f"No data scraped for {platform}.")


# In[15]:


print(Xbox_dfs['Xbox 360'])


# In[16]:


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
    Playstation_dfs[platform] = combined_Playstation_df
    
    


# In[17]:


Playstation_dfs['Playstation 2']['First released'] = Playstation_dfs['Playstation 2']['First released'].apply(remove_region_initials)


# In[18]:


Playstation_dfs['Playstation 2'].columns = Playstation_dfs['Playstation 2'].columns.str.strip()


# In[19]:


Playstation_dfs['Playstation 2'].columns


# In[20]:


Playstation_dfs['Playstation 2']['Platform'] = 'Playstation 2'


# In[21]:


Playstation_dfs['Playstation 2'] = Playstation_dfs['Playstation 2'][['First released','Platform']]


# In[22]:


Playstation_dfs['Playstation 2']


# In[23]:


print(Playstation_dfs['Playstation 3'])


# In[24]:


add_first_released_column(Playstation_dfs['Playstation 3'])


# In[25]:


print(Playstation_dfs['Playstation 3'].columns)


# In[26]:


Playstation_dfs['Playstation 3']['Platform'] = 'Playstation 3'


# In[27]:


Playstation_dfs['Playstation 3'] = Playstation_dfs['Playstation 3'][['First released','Platform']]


# In[28]:


Playstation_dfs['Playstation 3'].head()


# In[29]:


add_first_released_column(Playstation_dfs['Playstation 4'])


# In[30]:


Playstation_dfs['Playstation 4']['Platform'] = 'Playstation 4'


# In[31]:


Playstation_dfs['Playstation 4'] = Playstation_dfs['Playstation 4'][['First released','Platform']]


# In[32]:


Playstation_dfs['Playstation 4'].head()


# In[33]:


print(Playstation_dfs['Playstation 5'])


# In[34]:


add_first_released_column(Playstation_dfs['Playstation 5'])


# In[35]:


Playstation_dfs['Playstation 5']['Platform'] = 'Playstation 5'


# In[36]:


Playstation_dfs['Playstation 5'] = Playstation_dfs['Playstation 5'][['First released','Platform']]


# In[37]:


Playstation_dfs['Playstation 5'].head()


# In[ ]:





# In[38]:


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
        Nintendo_dfs[platform] = combined_Nintendo_df
    # if not combined_Nintendo_df.empty:
    #     cleaned_df = clean_dataframe(combined_Nintendo_df, platform)
    #     Nintendo_dfs[platform] = cleaned_df
    #     cleaned_df.to_csv(f'wikipedia_{platform.replace(" ", "_")}.csv', index=False)
    #     print(f"Data for {platform} scraped and saved.")
    # else:
    #     print(f"No data scraped for {platform}.")


# In[39]:


if isinstance(Nintendo_dfs['GameCube'].columns, pd.MultiIndex):
    Nintendo_dfs['GameCube'].columns = [' '.join(col).strip() for col in Nintendo_dfs['GameCube'].columns.values]
    print("Flattened columns:", Nintendo_dfs['GameCube'].columns)


# In[40]:


Nintendo_dfs['GameCube']['First released First released'] = Nintendo_dfs['GameCube']['First released First released'].apply(remove_region_initials)


# In[41]:


Nintendo_dfs['GameCube']['First released'] = Nintendo_dfs['GameCube']['First released First released']


# In[42]:


Nintendo_dfs['GameCube']['Platform'] = 'GameCube'


# In[43]:


Nintendo_dfs['GameCube'] = Nintendo_dfs['GameCube'][['First released','Platform']]


# In[ ]:





# In[44]:


print(Nintendo_dfs['GameCube'])


# In[45]:


print(Nintendo_dfs['Wii'])


# In[46]:


if isinstance(Nintendo_dfs['Wii'].columns, pd.MultiIndex):
    Nintendo_dfs['Wii'].columns = [' '.join(col).strip() for col in Nintendo_dfs['Wii'].columns.values]
    print("Flattened columns:", Nintendo_dfs['Wii'].columns)


# In[47]:


Nintendo_dfs['Wii']['First released First released'] = Nintendo_dfs['Wii']['First released First released'].apply(remove_region_initials)


# In[48]:


Nintendo_dfs['Wii']['First released'] = Nintendo_dfs['Wii']['First released First released']


# In[49]:


Nintendo_dfs['Wii']['Platform'] = 'Wii'


# In[50]:


Nintendo_dfs['Wii'] = Nintendo_dfs['Wii'][['First released','Platform']]


# In[51]:


print(Nintendo_dfs['Wii'])


# In[ ]:





# In[52]:


print(Nintendo_dfs['Wii U'])


# In[53]:


if isinstance(Nintendo_dfs['Wii U'].columns, pd.MultiIndex):
    Nintendo_dfs['Wii U'].columns = [' '.join(col).strip() for col in Nintendo_dfs['Wii U'].columns.values]
    print("Flattened columns:", Nintendo_dfs['Wii U'].columns)


# In[54]:


Nintendo_dfs['Wii U'] = add_first_released_column_WiiU(Nintendo_dfs['Wii U'])


# In[55]:


Nintendo_dfs['Wii U']['Platform'] = 'Wii U'


# In[56]:


Nintendo_dfs['Wii U'] = Nintendo_dfs['Wii U'][['First released','Platform']]


# In[57]:


print(Nintendo_dfs['Wii U'])


# In[58]:


print(Nintendo_dfs['Switch']['Release date'])


# In[59]:


Nintendo_dfs['Switch']['First released'] = Nintendo_dfs['Switch']['Release date'].apply(convert_dates)


# In[60]:


print(Nintendo_dfs['Switch'].columns)


# In[61]:


Nintendo_dfs['Switch']['Platform'] = 'Switch'


# In[62]:


Nintendo_dfs['Switch'] = Nintendo_dfs['Switch'][['First released','Platform']]


# In[63]:


Nintendo_dfs['Switch'].head(7)


# In[64]:


for platform, df in Playstation_dfs.items():
    print(f"Columns for {platform}:")
    print(df.columns)
    print()  # Add an empty line for better readability between different platforms


# In[65]:


for platform, df in Xbox_dfs.items():
    print(f"Columns for {platform}:")
    print(df.columns)
    print()  # Add an empty line for better readability between different platforms


# In[66]:


for platform, df in Nintendo_dfs.items():
    print(f"Columns for {platform}:")
    print(df.columns)
    print()  # Add an empty line for better readability between different platforms


# In[67]:


# create a single combined data frame for analysis 


# In[68]:


all_Consoles_df = []

# Add PlayStation DataFrames
for platform, df in Playstation_dfs.items():
    all_Consoles_df.append(df)  # Correctly append to the list

# Add Xbox DataFrames
for platform, df in Xbox_dfs.items():
    all_Consoles_df.append(df)  # Correctly append to the list

# Add Nintendo DataFrames
for platform, df in Nintendo_dfs.items():
    all_Consoles_df.append(df)  # Correctly append to the list

# Concatenate all DataFrames into a single DataFrame
All_df = pd.concat(all_Consoles_df, ignore_index=True)

All_df ['First released'] = pd.to_datetime(All_df ['First released'], errors='coerce')

All_df = All_df.dropna(subset=['First released'])
All_df['First released'] = pd.to_datetime(All_df['First released'], errors='coerce')
All_df['First released'] = All_df['First released'].apply(lambda x: x if pd.isna(x) else x.normalize())


# In[69]:


ps2_games = All_df[All_df['Platform'] == 'Playstation 2']
print(ps2_games)


# In[ ]:





# In[70]:


ps2_games = All_df[All_df['Platform'] == 'Playstation 2']

# Convert 'First released' to datetime if it's not already
ps2_games['First released'] = pd.to_datetime(ps2_games['First released'], errors='coerce')

# Plot the distribution of the release dates using a histogram with KDE
plt.figure(figsize=(12, 8))
sns.histplot(ps2_games['First released'], kde=True, stat='count', bins=30)

# Set plot title and labels
plt.title('Distribution of PlayStation 2 Game Releases Over Time')
plt.xlabel('Release Date')
plt.ylabel('Number of Games Released')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Show the plot
plt.show()


# In[71]:


# Filter data for PlayStation consoles
ps_data = All_df[All_df['Platform'].str.contains('Playstation')]

plt.figure(figsize=(12, 8))

# Create the violin plot
sns.violinplot(data=ps_data, y='Platform', x='First released', inner=None, color=".8")

# Overlay a boxplot to show quartiles and whiskers
sns.boxplot(data=ps_data, y='Platform', x='First released', whis=1.5, width=0.1, fliersize=0)

# Set plot title and labels
plt.title('Distribution of Game Releases Over Time by PlayStation Console')
plt.ylabel('PlayStation Console')
plt.xlabel('Release Date')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Show the plot
plt.show()


# In[73]:


ps_data = All_df[All_df['Platform'].str.contains('Playstation')]

# Create a list to store the summary statistics as dictionaries
summary_list = []

# Calculate the statistics for each platform
platforms = ps_data['Platform'].unique()
for platform in platforms:
    platform_data = ps_data[ps_data['Platform'] == platform]['First released'].dropna()
    q1 = platform_data.quantile(0.25)
    median = platform_data.quantile(0.5)
    q3 = platform_data.quantile(0.75)
    min_date = platform_data.min()
    max_date = platform_data.max()
    
    # Append the statistics to the list as a dictionary
    summary_list.append({
        'Platform': platform,
        'Q1': q1.date(),
        'Median': median.date(),
        'Q3': q3.date(),
        'Min': min_date.date(),
        'Max': max_date.date()
    })

# Convert the list of dictionaries to a DataFrame
summary_stats = pd.DataFrame(summary_list)

# Display the summary statistics table
print(summary_stats)


# In[74]:


ps_data = All_df[All_df['Platform'].str.contains('Playstation')]

# Get a list of unique PlayStation platforms
platforms = ps_data['Platform'].unique()

# Loop through each platform and create a plot
for platform in platforms:
    # Filter the data for the specific platform
    platform_games = ps_data[ps_data['Platform'] == platform]
    
    # Convert 'First released' to datetime if it's not already
    platform_games['First released'] = pd.to_datetime(platform_games['First released'], errors='coerce')

    # Plot the distribution of the release dates using a histogram with KDE
    plt.figure(figsize=(12, 8))
    sns.histplot(platform_games['First released'], kde=True, stat='count', bins=30)

    # Set plot title and labels
    plt.title(f'Distribution of {platform} Game Releases Over Time')
    plt.xlabel('Release Date')
    plt.ylabel('Number of Games Released')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Show the plot
    plt.show()


# In[76]:


ps_data = All_df[All_df['Platform'].str.contains('Playstation')]

# Convert 'First released' to datetime if it's not already
ps_data['First released'] = pd.to_datetime(ps_data['First released'], errors='coerce')

# Create a figure
plt.figure(figsize=(12, 8))

# Loop through each platform and plot on the same graph
platforms = ps_data['Platform'].unique()
for platform in platforms:
    platform_games = ps_data[ps_data['Platform'] == platform]
    
    # Plot the distribution of the release dates using KDE on the same graph
    sns.kdeplot(platform_games['First released'], label=platform, bw_adjust=0.5)

# Set plot title and labels
plt.title('Distribution of PlayStation Game Releases Over Time by Console')
plt.xlabel('Release Date')
plt.ylabel('Density')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Show legend
plt.legend(title='Platform')

# Show the plot
plt.show()


# In[77]:


plt.figure(figsize=(12, 8))

# Loop through each platform and plot on the same graph
for platform in platforms:
    platform_games = ps_data[ps_data['Platform'] == platform]
    
    # Plot the distribution of the release dates using a histogram on the same graph
    sns.histplot(platform_games['First released'], label=platform, bins=30, kde=False, stat='count', element='step', alpha=0.5)

# Set plot title and labels
plt.title('Distribution of PlayStation Game Releases Over Time by Console')
plt.xlabel('Release Date')
plt.ylabel('Number of Games Released')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Show legend
plt.legend(title='Platform')

# Show the plot
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[78]:


# Create the data as a list of dictionaries
data = [
    {'Console': 'PlayStation 2', 'Release Date': '2000-03-04'},
    {'Console': 'PlayStation 3', 'Release Date': '2006-11-11'},
    {'Console': 'PlayStation 4', 'Release Date': '2013-11-15'},
    {'Console': 'PlayStation 5', 'Release Date': '2020-11-12'},
    {'Console': 'Xbox', 'Release Date': '2001-11-15'},
    {'Console': 'Xbox 360', 'Release Date': '2005-11-22'},
    {'Console': 'Xbox One', 'Release Date': '2013-11-22'},
    {'Console': 'Xbox Series X/S', 'Release Date': '2020-11-10'},
    {'Console': 'GameCube', 'Release Date': '2001-09-14'},
    {'Console': 'Wii', 'Release Date': '2006-11-19'},
    {'Console': 'Wii U', 'Release Date': '2012-11-18'},
    {'Console': 'Nintendo Switch', 'Release Date': '2017-03-03'}
]

# Create a DataFrame
df = pd.DataFrame(data)

# Convert the 'Release Date' column to datetime format
df['Release Date'] = pd.to_datetime(df['Release Date'])

# Display the DataFrame
print(df)


# In[ ]:




