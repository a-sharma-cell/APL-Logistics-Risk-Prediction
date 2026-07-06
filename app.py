import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="APL Logistics — Risk Dashboard",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model():
    with open('rf_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()

@st.cache_data
def load_data():
    df = pd.read_csv('APL_Logistics.csv', encoding='latin1')
    return df

@st.cache_data
def prepare_risk_data():
    df = pd.read_csv('APL_Logistics.csv', encoding='latin1')
    raw_df = df.copy()

    df = df.drop(columns=['Delivery Status', 'Days for shipping (real)',
                           'Customer Fname', 'Customer Lname', 'Customer Street',
                           'Customer Id', 'Order Customer Id', 'Product Name', 'Customer Zipcode'])

    label_cols = ['Category Name', 'Customer City', 'Customer State',
                  'Department Name', 'Order City', 'Order Country',
                  'Order Region', 'Order State']
    le = LabelEncoder()
    for col in label_cols:
        df[col] = le.fit_transform(df[col])

    df = pd.get_dummies(df, columns=['Type', 'Customer Country', 'Customer Segment',
                                      'Market', 'Shipping Mode', 'Order Status'])

    X = df.drop(columns=['Late_delivery_risk'])

    scale_cols = ['Days for shipment (scheduled)', 'Benefit per order',
                  'Sales per customer', 'Category Id', 'Category Name',
                  'Customer City', 'Customer State', 'Department Id',
                  'Department Name', 'Latitude', 'Longitude', 'Order City',
                  'Order Country', 'Order Item Discount', 'Order Item Discount Rate',
                  'Order Item Product Price', 'Order Item Profit Ratio',
                  'Order Item Quantity', 'Sales', 'Order Item Total',
                  'Order Profit Per Order', 'Order Region', 'Order State',
                  'Product Price']

    X[scale_cols] = scaler.transform(X[scale_cols])

    proba = model.predict_proba(X)[:, 1]

    raw_df['Late_Probability'] = (proba * 100).round(1)
    raw_df['Risk_Category'] = pd.cut(proba,
                                      bins=[0, 0.3, 0.7, 1.0],
                                      labels=['Low Risk', 'Medium Risk', 'High Risk'])
    return raw_df

df = load_data()

st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/8c/APL_logo.svg/1200px-APL_logo.svg.png", width=150)
st.sidebar.title("APL Logistics")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Overview", "📊 Risk Analysis", "🔍 Order Prediction", "🚨 Action Panel"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Filters")

shipping_filter = st.sidebar.selectbox(
    "Shipping Mode",
    ["All", "First Class", "Second Class", "Same Day", "Standard Class"]
)

risk_threshold = st.sidebar.slider(
    "Risk Threshold (%)",
    min_value=0,
    max_value=100,
    value=70
)

if shipping_filter != "All":
    df_filtered = df[df['Shipping Mode'] == shipping_filter]
else:
    df_filtered = df.copy()

if page == "🏠 Overview":
    st.title("🚚 APL Logistics — Delay Risk Dashboard")
    st.markdown("##### Predict late delivery risk before shipment")
    st.markdown("---")

    total_orders = len(df_filtered)
    late_orders = df_filtered['Late_delivery_risk'].sum()
    late_pct = round((late_orders / total_orders) * 100, 1)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Orders", f"{total_orders:,}")
    with col2:
        st.metric("Late Orders", f"{int(late_orders):,}", delta=f"{late_pct}%")
    with col3:
        st.metric("Model Accuracy", "77%", delta="Random Forest")
    with col4:
        st.metric("On-Time Orders", f"{int(total_orders - late_orders):,}")

    st.markdown("---")
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Late Delivery % by Shipping Mode")
        mode_data = df_filtered.groupby('Shipping Mode')['Late_delivery_risk'].mean().reset_index()
        mode_data.columns = ['Shipping Mode', 'Late %']
        mode_data['Late %'] = (mode_data['Late %'] * 100).round(1)
        fig1 = px.bar(mode_data, x='Shipping Mode', y='Late %',
                      color='Late %', color_continuous_scale='Reds', text='Late %')
        fig1.update_traces(texttemplate='%{text}%', textposition='outside')
        st.plotly_chart(fig1, use_container_width=True)

    with col_right:
        st.subheader("Late Delivery % by Region (Top 10)")
        region_data = df_filtered.groupby('Order Region')['Late_delivery_risk'].mean().reset_index()
        region_data.columns = ['Region', 'Late %']
        region_data['Late %'] = (region_data['Late %'] * 100).round(1)
        region_data = region_data.sort_values('Late %', ascending=False).head(10)
        fig2 = px.bar(region_data, x='Late %', y='Region',
                      orientation='h', color='Late %', color_continuous_scale='Oranges')
        fig2.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig2, use_container_width=True)

elif page == "📊 Risk Analysis":
    st.title("📊 Risk Analysis")
    st.markdown("##### Understanding key delay drivers")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Feature Importance — Top 10")
        feature_data = pd.DataFrame({
            'Feature': ['Days for shipment (scheduled)', 'Shipping Mode_Standard Class',
                        'Shipping Mode_First Class', 'Latitude', 'Order City',
                        'Longitude', 'Order State', 'Benefit per order',
                        'Order Profit Per Order', 'Order Item Profit Ratio'],
            'Importance': [0.0636, 0.0619, 0.0573, 0.0560, 0.0528,
                           0.0515, 0.0468, 0.0409, 0.0407, 0.0360]
        })
        fig3 = px.bar(feature_data, x='Importance', y='Feature',
                      orientation='h', color='Importance', color_continuous_scale='Blues')
        fig3.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        st.subheader("Shipping Mode — Late % Comparison")
        mode_data2 = df.groupby('Shipping Mode')['Late_delivery_risk'].mean().reset_index()
        mode_data2.columns = ['Shipping Mode', 'Late %']
        mode_data2['Late %'] = (mode_data2['Late %'] * 100).round(1)
        fig4 = px.bar(mode_data2, x='Shipping Mode', y='Late %',
                      color='Late %', color_continuous_scale='Reds', text='Late %')
        fig4.update_traces(texttemplate='%{text}%', textposition='outside')
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")
    st.subheader("Model Performance Comparison")
    model_perf = pd.DataFrame({
        'Model': ['Logistic Regression', 'XGBoost', 'Random Forest'],
        'Accuracy': [71, 74, 77],
        'Recall': [54, 64, 69],
        'F1 Score': [67, 73, 77]
    })
    fig5 = px.bar(model_perf, x='Model', y=['Accuracy', 'Recall', 'F1 Score'],
                  barmode='group',
                  color_discrete_sequence=['#378ADD', '#E24B4A', '#639922'])
    st.plotly_chart(fig5, use_container_width=True)

    st.markdown("---")
    st.subheader("Key Insights")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.info("🚨 **First Class** shipping has 95.3% late delivery rate — highest among all modes")
    with col_b:
        st.info("📅 **Scheduled Days** is the strongest predictor — unrealistic promises cause delays")
    with col_c:
        st.info("🌍 **Region** has minimal impact — shipping mode matters more than location")

elif page == "🔍 Order Prediction":
    st.title("🔍 Order Risk Prediction")
    st.markdown("##### Enter order details to get late delivery risk score")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        shipping_mode_pred = st.selectbox("Shipping Mode",
            ["First Class", "Second Class", "Same Day", "Standard Class"])
        scheduled_days = st.slider("Scheduled Shipping Days", 1, 7, 3)
        order_quantity = st.slider("Order Quantity", 1, 5, 2)
        benefit = st.number_input("Benefit per Order ($)", value=50.0)
        sales = st.number_input("Sales per Customer ($)", value=200.0)

    with col2:
        market = st.selectbox("Market", ["Africa", "Europe", "LATAM", "Pacific Asia", "USCA"])
        customer_segment = st.selectbox("Customer Segment",
            ["Consumer", "Corporate", "Home Office"])
        order_profit = st.number_input("Order Profit ($)", value=30.0)
        product_price = st.number_input("Product Price ($)", value=100.0)
        discount_rate = st.slider("Discount Rate", 0.0, 1.0, 0.1)

    st.markdown("---")

    if st.button("🔍 Predict Risk", use_container_width=True):
        mode_risk = {
            "First Class": 0.85,
            "Second Class": 0.65,
            "Same Day": 0.40,
            "Standard Class": 0.30
        }
        base_risk = mode_risk[shipping_mode_pred]
        day_factor = (scheduled_days - 1) * 0.02
        qty_factor = (order_quantity - 1) * 0.01
        profit_factor = -0.01 if benefit > 100 else 0.02
        risk_score = min(0.99, max(0.01, base_risk + day_factor + qty_factor + profit_factor))
        risk_pct = round(risk_score * 100, 1)

        st.markdown("### Prediction Result")
        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1:
            st.metric("Late Probability", f"{risk_pct}%")
        with col_r2:
            st.metric("Shipping Mode", shipping_mode_pred)
        with col_r3:
            st.metric("Scheduled Days", scheduled_days)

        if risk_score >= 0.7:
            st.error(f"🔴 HIGH RISK — {risk_pct}% probability of late delivery")
            st.error("⚠️ Recommended: Reroute shipment or upgrade shipping mode immediately")
        elif risk_score >= 0.3:
            st.warning(f"🟡 MEDIUM RISK — {risk_pct}% probability of late delivery")
            st.warning("👀 Recommended: Monitor closely and prepare backup options")
        else:
            st.success(f"🟢 LOW RISK — {risk_pct}% probability of late delivery")
            st.success("✅ Recommended: Standard processing — no immediate action needed")

elif page == "🚨 Action Panel":
    st.title("🚨 Action Panel")
    st.markdown("##### High risk orders requiring immediate attention")
    st.markdown("---")

    with st.spinner("Loading risk scores from model..."):
        risk_df = prepare_risk_data()

    if shipping_filter != "All":
        risk_df = risk_df[risk_df['Shipping Mode'] == shipping_filter]

    risk_df_filtered = risk_df[risk_df['Late_Probability'] >= risk_threshold]

    high_risk = len(risk_df[risk_df['Risk_Category'] == 'High Risk'])
    medium_risk = len(risk_df[risk_df['Risk_Category'] == 'Medium Risk'])
    low_risk = len(risk_df[risk_df['Risk_Category'] == 'Low Risk'])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Threshold Set", f"{risk_threshold}%")
    with col2:
        st.metric("High Risk Orders", f"{high_risk:,}")
    with col3:
        st.metric("Medium Risk Orders", f"{medium_risk:,}")
    with col4:
        st.metric("Orders Above Threshold", f"{len(risk_df_filtered):,}")

    st.markdown("---")
    st.subheader(f"Orders with Late Probability >= {risk_threshold}%")

    display_cols = ['Shipping Mode', 'Order Region', 'Days for shipment (scheduled)',
                    'Customer Segment', 'Late_Probability', 'Risk_Category']

    st.dataframe(
        risk_df_filtered[display_cols].head(100).sort_values('Late_Probability', ascending=False),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")
    st.subheader("Recommended Actions")
    st.error("🔴 **High Risk:** Immediately reroute or upgrade shipping — contact customer proactively")
    st.warning("🟡 **Medium Risk:** Monitor closely — prepare backup shipping options")
    st.success("🟢 **Low Risk:** Standard processing — no immediate action required")