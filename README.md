  # Restaurant Order Analytics & Recommendation System
  
  A data-driven project to analyze restaurant orders and provide cuisine-level recommendations based on customer behavior, ratings, and operational metrics.
  
  ## Project Structure
  
  - `data/`: Raw and processed data files in different sub-folders
  - `notebooks/`: Jupyter notebooks for Cleaning, EDA, Forecasting and Recommendation System
  - `dashboard/`: Streamlit dashboard code
  - `reports/`: Presentations and business recommendations
  - `requirements.txt`: Requirements to run the project 
  
  ## Approach
  
  ### 1. Data Cleaning
  - Identified and handled missing and duplicate values.
  - Extracted useful features from `order_date` such as day of the week, weekend/weekday flag, and month.
  - Standardized data formats across all columns.
  - Detected and mitigated the impact of outliers.
  
  ### 2. Exploratory Data Analysis (EDA)
  - Used `seaborn` and `matplotlib` to analyze:
    - Popular dishes and cuisines
    - Sales patterns by hour, customer rating, cuisines and other parameters
    - Inventory Usage estimation through total order and demand.
  
  ### 3. Modeling
  - Due to the dataset being limited to a single day:
    - Trained a **Random Forest Regressor** (RFR) to forecast demand during dinner hours (6 PM–11 PM) using lunch-hour data (11 AM–5 PM).
    - Visualized feature importance to understand key demand drivers.
  
  ### 4. Recommendation Engine
  - Designed a **cuisine-based recommendation system** using:
    - Order frequency
    - Revenue, Customer food and delivery ratings
  - Normalized all metrics to balance their influence and avoid domination by any single factor.
  - Created a correlation heatmap to analyze relationships among ratings and revenue.
  
  ### 5. Dashboard
  - Built a **Streamlit dashboard** to interactively explore data insights and support business decisions.
  - To run the dashboard:
    ```bash
    cd dashboard
    streamlit run dashboard.py
  
  ## Setup
  ```bash
  pip install -r requirements.txt
