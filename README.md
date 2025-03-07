# Electric Twins Hack - Social Media Analysis Dashboard

This dashboard analyzes social media data to detect suspicious activities related to wildlife trafficking.

## Features

- Network visualization of people and group memberships
- Suspicion score calculation and ranking
- Detailed view of person activities and content analysis
- Location and species detection from posts

## Setup

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

## Data

The application uses a SQLite database containing anonymized social media data, located in the `data/` directory.

## Structure

- `app.py` - Main Streamlit application
- `explore.py` - Original exploration code with Marimo
- `data/` - Contains the SQLite database and other data resources

Billy, Joonas & Luis


