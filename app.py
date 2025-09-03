import streamlit as st
import pandas as pd
import plotly.express as px
from openai import OpenAI

# ------------------ CONFIG ------------------ #
st.set_page_config(page_title="AI Data Analyzer", layout="wide")
st.title("📊 Sarvesh's Data Analysis Platform")

# OpenRouter API Setup
API_KEY = st.secrets["API_KEY"]  # <-- Set this in Streamlit Secrets
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=API_KEY)

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

            # ✅ Direct computations with pandas
            try:
                if "total" in query.lower() or "sum" in query.lower():
                    numeric_cols = df.select_dtypes(include="number").columns
                    if len(numeric_cols) > 0:
                        computed_answer = df[numeric_cols].sum().to_frame("Total").reset_index()
                        st.markdown("### 📝 Computed Answer from Full Data")
                        st.write(computed_answer)

                elif "average" in query.lower() or "mean" in query.lower():
                    numeric_cols = df.select_dtypes(include="number").columns
                    if len(numeric_cols) > 0:
                        computed_answer = df[numeric_cols].mean().to_frame("Average").reset_index()
                        st.markdown("### 📝 Computed Answer from Full Data")
                        st.write(computed_answer)

                elif " by " in query.lower():  # detect group by query
                    words = query.lower().split()
                    if "by" in words:
                        by_index = words.index("by")
                        if by_index > 0 and by_index < len(words) - 1:
                            group_col = words[by_index + 1].capitalize()
                            numeric_cols = df.select_dtypes(include="number").columns
                            if group_col in df.columns and len(numeric_cols) > 0:
                                computed_answer = df.groupby(group_col)[numeric_cols].sum().reset_index()
                                st.markdown(f"### 📝 Grouped by {group_col}")
                                st.write(computed_answer)
                                # auto-bar chart
                                chart = px.bar(computed_answer, x=group_col, y=numeric_cols[0],
                                               title=f"{numeric_cols[0]} by {group_col}")

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
