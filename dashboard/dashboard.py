import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load your data
df = pd.read_pickle("../data/cleaned/combined.pkl")

# Heading
st.set_page_config(page_title="Restaurant Demand Dashboard", layout="wide")
st.title("Restaurant Detailed Analysis Dashboard")

# --- Sidebar Filters ---
st.sidebar.header("🔍 Filters")
select_all = st.sidebar.checkbox("Select All", value=True)

# Get unique options safely
all_payment_modes = df["payment_mode"].dropna().unique().tolist()
all_cuisines = df["cuisine"].dropna().unique().tolist()
all_restaurants = df["restaurant_name"].dropna().unique().tolist()

if select_all:
    selected_payment = all_payment_modes
    selected_cuisines = all_cuisines
    selected_restaurants = all_restaurants
    hour_range = (0, 23)
else:
    selected_payment = st.sidebar.multiselect(
        "Select Payment Mode(s)", 
        options=all_payment_modes,
        default=all_payment_modes[:1]  # At least one selected
    )
    selected_cuisines = st.sidebar.multiselect(
        "Select Cuisine(s)", 
        options=all_cuisines,
        default=all_cuisines[:1]  # At least one selected
    )
    selected_restaurants = st.sidebar.multiselect(
        "Select Restaurant(s)", 
        options=all_restaurants,
        default=all_restaurants[:1]  # At least one selected
    )
    hour_range = st.sidebar.slider("Select Hour Range", 0, 23, (0, 23))

# --- Apply Filters ---
filtered_df = df[
    (df["payment_mode"].isin(selected_payment)) &
    (df["cuisine"].isin(selected_cuisines)) &
    (df["restaurant_name"].isin(selected_restaurants)) &
    (df["hour"].between(hour_range[0], hour_range[1]))
]

# KPIs
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("📦 Filtered Orders", len(filtered_df))
col2.metric("💰 Filtered Revenue", f"₹{filtered_df['order_amount'].sum():,.0f}")
col3.metric("⏱️ Avg Delivery Time", f"{filtered_df['delivery_time_taken_(mins)'].mean():.1f} mins")
col4.metric("⭐ Avg Food Rating", f"{filtered_df['customer_rating_food'].mean():.2f}")
col5.metric("🚚 Avg Delivery Rating", f"{filtered_df['customer_rating_delivery'].mean():.2f}")

# Grouped Data for model
hourly_agg = df.groupby('hour').agg(
    total_orders=('order_id', 'count'),
    total_items=('quantity_of_items', 'sum'),
    total_revenue=('order_amount', 'sum'),
    avg_delivery_time=('delivery_time_taken_(mins)', 'mean')
).reset_index()

# Split by time: use early hours (0-17) to predict later hours (18-23)
train_data = hourly_agg[hourly_agg['hour'] < 18]
test_data = hourly_agg[hourly_agg['hour'] >= 18]

# Features and target variable for training
X_train = train_data[['total_items', 'total_revenue', 'avg_delivery_time']]
y_train = train_data['total_orders']
X_test = test_data[['total_items', 'total_revenue', 'avg_delivery_time']]
y_test = test_data['total_orders']

# Model Training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predicting for test data
y_pred = model.predict(X_test)

# Revenue by Hour
st.markdown("---")
st.subheader("💰 Revenue by Hour")
revenue_hourly = filtered_df.groupby("hour")["order_amount"].sum().reset_index()
chart_type = st.selectbox("Chart Type for Revenue by Hour", ["Bar", "Line"])
if chart_type == "Bar":
    fig6 = px.bar(revenue_hourly, x="hour", y="order_amount", color="order_amount", color_continuous_scale='Viridis')
else:
    fig6 = px.line(revenue_hourly, x="hour", y="order_amount", markers=True)
st.plotly_chart(fig6, use_container_width=True)

# Hourly Demand
st.markdown("---")
st.subheader("📈 Hourly Demand")
hourly_orders = filtered_df.groupby("hour").size().reset_index(name="total_orders")
fig1 = px.line(hourly_orders, x="hour", y="total_orders", markers=True, labels={"hour":"Hour", "total_orders":"Total Orders Placed"})
st.plotly_chart(fig1, use_container_width=True)

# Payment Mode Distribution
st.markdown("---")
st.subheader("💳 Payment Mode Distribution")
fig_payment = px.pie(filtered_df, names="payment_mode", hole=0.4)
st.plotly_chart(fig_payment, use_container_width=True)

# Popular Dishes
st.markdown("---")
st.subheader("🍽️ Popular Dishes")
col1, col2 = st.columns(2)

with col1:
    top_dishes = filtered_df["category"].value_counts().head(10).reset_index()
    top_dishes.columns = ["category", "count"]
    fig_dishes = px.bar(top_dishes, x="category", y="count", color="count", title="🍽️ Most Ordered Categories")
    st.plotly_chart(fig_dishes, use_container_width=True)

with col2:
    cuisine_counts = filtered_df['cuisine'].value_counts().reset_index()
    cuisine_counts.columns = ['cuisine', 'order_count']
    fig_cuisine = px.bar(cuisine_counts, x='cuisine', y='order_count', color='cuisine', title="🍽️ Most Frequent Ordered Dishes")
    st.plotly_chart(fig_cuisine, use_container_width=True)

# Top Customers
with st.expander("👥 Top Customers"):
    top_customers_df = filtered_df.groupby("customer_name")["order_amount"].sum().sort_values(ascending=False).head(10).reset_index()
    fig_top_customers = px.bar(top_customers_df, x="customer_name", y="order_amount",
                               labels={"customer_name": "Customer", "order_amount": "Revenue"},
                               color="order_amount", color_continuous_scale='Turbo')
    st.plotly_chart(fig_top_customers, use_container_width=True)

# Top Restaurants
with st.expander("🏆 Top Restaurants by Revenue"):
    top_restaurants_df = filtered_df.groupby("restaurant_name")["order_amount"].sum().sort_values(ascending=False).head(10).reset_index()
    fig3 = px.bar(top_restaurants_df, x="restaurant_name", y="order_amount",
                  labels={"restaurant_name": "Restaurant", "order_amount": "Revenue"},
                  color="order_amount", color_continuous_scale='Plasma')
    st.plotly_chart(fig3, use_container_width=True)

# Cuisine-wise Sales
st.markdown("---")
st.subheader("🍽️ Cuisine-wise Sales Analysis")
col1, col2 = st.columns(2)

with col1:
    avg_sales = filtered_df.groupby("cuisine")["order_amount"].mean().reset_index().sort_values("order_amount", ascending=False)
    fig_avg = px.bar(avg_sales, x="cuisine", y="order_amount", title="Average Sales by Cuisine", labels={"order_amount": "Avg Order Amount (₹)", "cuisine": "Cuisine"}, color="cuisine")
    fig_avg.update_layout(showlegend=False, xaxis_tickangle=45)
    st.plotly_chart(fig_avg, use_container_width=True)

with col2:
    total_sales = filtered_df.groupby("cuisine")["order_amount"].sum().reset_index().sort_values("order_amount", ascending=False)
    fig_total = px.bar(total_sales, x="cuisine", y="order_amount", title="Total Sales by Cuisine", labels={"order_amount": "Total Order Amount (₹)", "cuisine": "Cuisine"}, color="cuisine")
    fig_total.update_layout(showlegend=False, xaxis_tickangle=45)
    st.plotly_chart(fig_total, use_container_width=True)

# Ratings Distribution
st.markdown("---")
st.subheader("💬 Relation of Customer Rating with Orders")
col1, col2 = st.columns(2)

with col1:
    fig5 = px.histogram(filtered_df, x="customer_rating_food", nbins=10, title="Food Ratings Distribution", labels={"count":"Total Orders Placed", "customer_rating_food":"Customer Rating"})
    st.plotly_chart(fig5, use_container_width=True)

with col2:
    rating_sales = filtered_df.groupby('customer_rating_food', as_index=False)['order_amount'].sum()
    fig_rating = px.bar(rating_sales, x='customer_rating_food', y='order_amount', color='customer_rating_food', color_continuous_scale='Plasma', title='Total Sales by Customer Food Rating', labels={"customer_rating_food":"Customer Rating", "order_amount":"Total Revenue"})
    st.plotly_chart(fig_rating, use_container_width=True)


# Predicted vs Actual Orders
st.markdown("---")
st.subheader("📊 Predicted vs Actual Orders")
fig_pred = go.Figure()
fig_pred.add_trace(go.Scatter(x=test_data["hour"], y=y_test, mode='lines+markers', name='Actual Orders'))
fig_pred.add_trace(go.Scatter(x=test_data["hour"], y=y_pred, mode='lines+markers', name='Predicted Orders'))
fig_pred.update_layout(title="Predicted vs Actual Hourly Orders", xaxis_title="Hour", yaxis_title="Orders")
st.plotly_chart(fig_pred, use_container_width=True)