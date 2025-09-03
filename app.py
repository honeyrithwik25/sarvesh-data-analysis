import streamlit as st
import pandas as pd
import plotly.express as px
from openai import OpenAI
from difflib import get_close_matches

# ------------------ CONFIG ------------------ #
st.set_page_config(page_title="AI Data Analyzer", layout="wide")
st.title("📊 Sarvesh's Data Analysis Platform")

# OpenRouter API Setup
API_KEY = st.secrets["API_KEY"]  # <-- Set this in Streamlit Secrets
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=API_KEY)

# ------------------ Helper Function ------------------ #
def match_column(user_word, df_columns):
    """Find the closest matching column name for a user word."""
    matches = get_close_matches(user_word.lower(), [c.lower() for c in df_columns], n=1, cutoff=0.6)
    if matches:
        for col in df_columns:
            if col.lower() == matches[0]:
                return col
    return None

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
            computed_answer = None
            chart = None
            q = query.lower()

            # ✅ Direct computations with pandas
            try:
                # Totals
                if "total" in q or "sum" in q:
                    numeric_cols = df.select_dtypes(include="number").columns
                    if len(numeric_cols) > 0:
                        computed_answer = df[numeric_cols].sum().to_frame("Total").reset_index()
                        st.markdown("### 📝 Computed Answer from Full Data")
                        st.write(computed_answer)

                # Averages
                elif "average" in q or "mean" in q:
                    numeric_cols = df.select_dtypes(include="number").columns
                    if len(numeric_cols) > 0:
                        computed_answer = df[numeric_cols].mean().to_frame("Average").reset_index()
                        st.markdown("### 📝 Computed Answer from Full Data")
                        st.write(computed_answer)

                # Group by queries
                elif "group" in q or " by " in q or "wise" in q:
                    words = q.replace("wise", "by").split()
                    if "by" in words or "group" in words:
                        group_candidate = None
                        value_candidate = None
                        for w in words:
                            col_match = match_column(w, df.columns)
                            if col_match and group_candidate is None:
                                group_candidate = col_match
                            elif col_match:
                                value_candidate = col_match

                        # Default numeric column if none matched
                        if value_candidate is None:
                            numeric_cols = df.select_dtypes(include="number").columns
                            if len(numeric_cols) > 0:
                                value_candidate = numeric_cols[0]

                        if group_candidate and value_candidate:
                            computed_answer = df.groupby(group_candidate)[value_candidate].sum().reset_index()
                            st.markdown(f"### 📝 {value_candidate} by {group_candidate}")
                            st.write(computed_answer)

                            # Auto bar chart
                            chart = px.bar(computed_answer, x=group_candidate, y=value_candidate,
                                           title=f"{value_candidate} by {group_candidate}")

            except Exception as e:
                st.warning("Couldn't compute directly: " + str(e))

            # ✅ Build prompt for AI explanation
            prompt = f"""
            You are a data analyst. The dataframe has {len(df)} rows and {len(df.columns)} columns.
            Columns: {list(df.columns)}
            Query: {query}
            If numeric calculations or group by are needed, they are shown above.
            Otherwise, provide logical analysis and explanation.
            """

            response = client.chat.completions.create(
                model="openai/gpt-4o-mini",  # you can swap this with any OpenRouter model
                messages=[{"role": "user", "content": prompt}]
            )

            answer = response.choices[0].message.content

            st.markdown("### 🤖 AI Analysis")
            st.write(answer)

            # ------------------ CHARTS ------------------ #
            try:
                if not chart:  # if not already plotted in group by
                    if "average" in q or "mean" in q:
                        chart = px.bar(df.mean().reset_index(), x="index", y=0, title="Average of Columns")
                    elif "group" in q or "count" in q:
                        col = df.columns[0]
                        chart = px.histogram(df, x=col, title=f"Grouping by {col}")
                    elif "percentage" in q:
                        col = df.columns[0]
                        chart = px.pie(df, names=col, title=f"Percentage Distribution of {col}")
            except Exception as e:
                st.warning("Couldn't auto-generate chart: " + str(e))

            if chart:
                st.plotly_chart(chart, use_container_width=True)

    except Exception as e:
        st.error("Error reading file: " + str(e))
