import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai

# ------------------ CONFIG ------------------ #
st.set_page_config(page_title="AI Data Analyzer", layout="wide")
st.title("📊 Sarvesh's Data Analysis Platform")

# Gemini API Setup
API_KEY = st.secrets["API_KEY"]  # <-- we will set this in Streamlit Cloud Secrets
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

# ------------------ UPLOAD ------------------ #
uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.write("### Preview of Data", df.head())

        # ------------------ USER QUERY ------------------ #
        query = st.text_input("Ask your question in plain English and hit enter button:")

        if query:
            # Pass query + dataframe info to Gemini
            prompt = f"""
            You are a data analyst. Analyze the dataframe below based on the user's query.
            Data (first 200 rows max): {df.head(50000).to_dict()}
            Query: {query}
            Return:
            - A clear explanation
            - Any calculated table in markdown format if needed
            - Suggested graph type (bar, line, pie, histogram, box)
            """

            response = model.generate_content(prompt)
            answer = response.text

            st.markdown("### 📝 Analysis")
            st.write(answer)

            # ------------------ CHARTS ------------------ #
            chart = None
            try:
                if "average" in query.lower() or "mean" in query.lower():
                    chart = px.bar(df.mean().reset_index(), x="index", y=0, title="Average of Columns")
                elif "group" in query.lower() or "count" in query.lower():
                    col = df.columns[0]
                    chart = px.histogram(df, x=col, title=f"Grouping by {col}")
                elif "percentage" in query.lower():
                    col = df.columns[0]
                    chart = px.pie(df, names=col, title=f"Percentage Distribution of {col}")
            except Exception as e:
                st.warning("Couldn't auto-generate chart: " + str(e))

            if chart:
                st.plotly_chart(chart, use_container_width=True)

    except Exception as e:
        st.error("Error reading file: " + str(e))
