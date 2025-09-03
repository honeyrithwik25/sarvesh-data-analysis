import streamlit as st
import pandas as pd
import plotly.express as px
from openai import OpenAI
from difflib import get_close_matches

# ------------------ CONFIG ------------------ #
st.set_page_config(page_title="AI Data Analyzer", layout="wide")
st.title("📊 Sarvesh's Data Analysis Platform")

# OpenRouter API Setup
API_KEY = st.secrets["API_KEY"]  # <-- Put your OpenRouter key in Streamlit Secrets
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

        # ------------------ Switch ------------------ #
        use_ai = st.checkbox("Show AI Explanation", value=True)

        if query:
            computed_answer = None
            chart = None
            q = query.lower()

            try:
                # ------------------ TOTALS ------------------ #
                if "total" in q or "sum" in q:
                    words = q.split()
                    target_col = None
                    for w in words:
                        col_match = match_column(w, df.columns)
                        if col_match:
                            target_col = col_match
                            break

                    if target_col:
                        computed_answer = pd.DataFrame({
                            "Column": [target_col],
                            "Total": [df[target_col].sum()]
                        })
                        st.markdown(f"### 📝 Total of {target_col}")
                        st.write(computed_answer)
                    else:
                        numeric_cols = df.select_dtypes(include="number").columns
                        if len(numeric_cols) > 0:
                            computed_answer = df[numeric_cols].sum().to_frame("Total").reset_index()
                            st.markdown("### 📝 Totals from All Numeric Columns")
                            st.write(computed_answer)

                # ------------------ AVERAGES ------------------ #
                elif "average" in q or "mean" in q:
                    words = q.split()
                    target_col = None
                    for w in words:
                        col_match = match_column(w, df.columns)
                        if col_match:
                            target_col = col_match
                            break

                    if target_col:
                        computed_answer = pd.DataFrame({
                            "Column": [target_col],
                            "Average": [df[target_col].mean()]
                        })
                        st.markdown(f"### 📝 Average of {target_col}")
                        st.write(computed_answer)
                    else:
                        numeric_cols = df.select_dtypes(include="number").columns
                        if len(numeric_cols) > 0:
                            computed_answer = df[numeric_cols].mean().to_frame("Average").reset_index()
                            st.markdown("### 📝 Averages from All Numeric Columns")
                            st.write(computed_answer)

                # ------------------ GROUP BY ------------------ #
                elif "group" in q or " by " in q or "wise" in q:
                    words = q.replace("wise", "by").split()
                    group_candidate = None
                    value_candidate = None
                    for w in words:
                        col_match = match_column(w, df.columns)
                        if col_match and group_candidate is None:
                            group_candidate = col_match
                        elif col_match:
                            value_candidate = col_match

                    if value_candidate is None:
                        numeric_cols = df.select_dtypes(include="number").columns
                        if len(numeric_cols) > 0:
                            value_candidate = numeric_cols[0]

                    if group_candidate and value_candidate:
                        computed_answer = df.groupby(group_candidate)[value_candidate].sum().reset_index()
                        st.markdown(f"### 📝 {value_candidate} by {group_candidate}")
                        st.write(computed_answer)

                        chart = px.bar(computed_answer, x=group_candidate, y=value_candidate,
                                       title=f"{value_candidate} by {group_candidate}")

            except Exception as e:
                st.warning("Couldn't compute directly: " + str(e))

            # ------------------ AI EXPLANATION ------------------ #
            if use_ai:  # ✅ Only if checkbox is checked
                prompt = f"""
                You are a data analyst. The dataframe has {len(df)} rows and {len(df.columns)} columns.
                Columns: {list(df.columns)}
                Query: {query}
                The computed results (if any) are already shown in table format above.
                Please provide a short explanation or insights only.
                """

                response = client.chat.completions.create(
                    model="openai/gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}]
                )

                answer = response.choices[0].message.content
                st.markdown("### 🤖 AI Explanation")
                st.write(answer)

            # ------------------ CHARTS ------------------ #
            if chart:
                st.plotly_chart(chart, use_container_width=True)

    except Exception as e:
        st.error("Error reading file: " + str(e))
