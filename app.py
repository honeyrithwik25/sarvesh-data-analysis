import streamlit as st
import pandas as pd
import plotly.express as px
from openai import OpenAI
from difflib import get_close_matches
import io

# ------------------ CONFIG ------------------ #
st.set_page_config(page_title="AI Data Analyzer", layout="wide")
st.title("📊 Universal AI Data Analysis Platform")

# ------------------ API SETUP ------------------ #
st.sidebar.header("🔑 API Configuration")

# Provider dropdown
provider = st.sidebar.selectbox(
    "Choose AI Provider",
    ["OpenRouter", "DeepSeek", "Gemini", "Qwen"]
)

# API key (from Streamlit Secrets or manual entry)
API_KEY = st.secrets.get("API_KEY", None)
if not API_KEY:
    API_KEY = st.sidebar.text_input("Enter your API Key", type="password")

# Map provider to base_url and default model
provider_map = {
    "OpenRouter": {"base_url": "https://openrouter.ai/api/v1", "model": "openai/gpt-4o-mini"},
    "DeepSeek": {"base_url": "https://api.deepseek.com", "model": "deepseek-chat"},
    "Gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta", "model": "gemini-2.0-flash"},
    "Qwen": {"base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "model": "qwen-max"}
}

base_url = provider_map[provider]["base_url"]
default_model = provider_map[provider]["model"]

# Initialize client
client = OpenAI(base_url=base_url, api_key=API_KEY)

# ------------------ Helper Functions ------------------ #
def match_column(user_word, df_columns):
    """Find the closest matching column name for a user word."""
    from difflib import get_close_matches
    matches = get_close_matches(user_word.lower(), [c.lower() for c in df_columns], n=1, cutoff=0.6)
    if matches:
        for col in df_columns:
            if col.lower() == matches[0]:
                return col
    return None

def download_button(df, filename="results.csv"):
    """Generate a CSV download button for a DataFrame."""
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="⬇️ Download Results as CSV",
        data=csv_buffer.getvalue(),
        file_name=filename,
        mime="text/csv"
    )

# ------------------ UPLOAD ------------------ #
uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if uploaded_file and API_KEY:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.write("### Preview of Data", df.head())

        # ------------------ USER QUERY ------------------ #
        query = st.text_input("Ask your question in plain English and hit enter:")

        # Toggle for AI explanation
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
                        download_button(computed_answer, "total_results.csv")
                    else:
                        numeric_cols = df.select_dtypes(include="number").columns
                        if len(numeric_cols) > 0:
                            computed_answer = df[numeric_cols].sum().to_frame("Total").reset_index()
                            st.markdown("### 📝 Totals from All Numeric Columns")
                            st.write(computed_answer)
                            download_button(computed_answer, "total_results.csv")

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
                        download_button(computed_answer, "average_results.csv")
                    else:
                        numeric_cols = df.select_dtypes(include="number").columns
                        if len(numeric_cols) > 0:
                            computed_answer = df[numeric_cols].mean().to_frame("Average").reset_index()
                            st.markdown("### 📝 Averages from All Numeric Columns")
                            st.write(computed_answer)
                            download_button(computed_answer, "average_results.csv")

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
                        download_button(computed_answer, "groupby_results.csv")

                        chart = px.bar(computed_answer, x=group_candidate, y=value_candidate,
                                       title=f"{value_candidate} by {group_candidate}")

                # ------------------ GRAPHS ON DEMAND ------------------ #
                elif "chart" in q or "graph" in q or "bar" in q or "pie" in q or "line" in q:
                    numeric_cols = df.select_dtypes(include="number").columns
                    if len(numeric_cols) > 0:
                        col = numeric_cols[0]
                        if "bar" in q:
                            chart = px.bar(df, x=df.columns[0], y=col, title=f"Bar Chart of {col}")
                        elif "pie" in q:
                            chart = px.pie(df, names=df.columns[0], values=col, title=f"Pie Chart of {col}")
                        elif "line" in q:
                            chart = px.line(df, x=df.columns[0], y=col, title=f"Line Chart of {col}")
                        else:
                            chart = px.histogram(df, x=col, title=f"Histogram of {col}")

            except Exception as e:
                st.warning("Couldn't compute directly: " + str(e))

            # ------------------ AI EXPLANATION ------------------ #
            if use_ai and API_KEY:
                prompt = f"""
                You are a data analyst. The dataframe has {len(df)} rows and {len(df.columns)} columns.
                Columns: {list(df.columns)}
                Query: {query}
                The computed results (if any) are already shown in table format above.
                Please provide a short explanation or insights only.
                """
                try:
                    response = client.chat.completions.create(
                        model=default_model,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    answer = response.choices[0].message.content
                    st.markdown("### 🤖 AI Explanation")
                    st.write(answer)
                except Exception as e:
                    st.error(f"AI call failed: {e}")

            # ------------------ CHARTS ------------------ #
            if chart:
                st.plotly_chart(chart, use_container_width=True)

    except Exception as e:
        st.error("Error reading file: " + str(e))
elif not API_KEY:
    st.warning("⚠️ Please provide an API key to use AI features.")
