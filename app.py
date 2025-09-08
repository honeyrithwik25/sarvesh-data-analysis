import streamlit as st
import pandas as pd
import plotly.express as px
from difflib import get_close_matches
import io

# ------------------ CONFIG ------------------ #
st.set_page_config(page_title="AI Data Analyzer", layout="wide")
st.title("📊 Sarvesh's Data Analysis Platform (Multi-Provider)")

# ------------------ SIDEBAR: API / PROVIDER ------------------ #
st.sidebar.header("🔧 AI Provider & Key")

provider = st.sidebar.selectbox(
    "Choose AI Provider",
    ["OpenRouter", "DeepSeek", "OpenAI", "Gemini"]
)

# Prefer secret stored key; fallback to manual input
API_KEY = st.secrets.get("API_KEY", None)
if not API_KEY:
    API_KEY = st.sidebar.text_input("Enter API Key", type="password")

# Optional: allow custom model name
model_input = st.sidebar.text_input("Model (optional)", value="")  # if empty, we use defaults below

# Provider defaults for base_url + model (for OpenAI-compatible providers)
provider_map = {
    "OpenRouter": {"base_url": "https://openrouter.ai/api/v1", "model": "openai/gpt-4o-mini"},
    "DeepSeek": {"base_url": "https://api.deepseek.com", "model": "deepseek-chat"},
    "OpenAI": {"base_url": None, "model": "gpt-4o-mini"},  # typical OpenAI usage (no custom base_url)
    # Gemini handled separately
}

default_model = model_input.strip() or provider_map.get(provider, {}).get("model", "")

# ------------------ Helper Functions ------------------ #
def match_column(user_word, df_columns):
    """Find the closest matching column name for a user word (case-insensitive)."""
    matches = get_close_matches(user_word.lower(), [c.lower() for c in df_columns], n=1, cutoff=0.6)
    if matches:
        for col in df_columns:
            if col.lower() == matches[0]:
                return col
    return None

def download_button(df, filename="results.csv"):
    """Create CSV download button for a DataFrame."""
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

# AI explanation toggle
use_ai = st.checkbox("Show AI Explanation", value=True)

# Small notice if key missing and AI requested
if use_ai and not API_KEY:
    st.warning("To see AI explanations please provide an API key in the sidebar or in Streamlit Secrets (API_KEY).")

# ------------------ MAIN LOGIC ------------------ #
if uploaded_file:
    try:
        # Load file
        if uploaded_file.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.write("### Preview of Data", df.head())

        # Query input
        query = st.text_input("Ask your question in plain English and hit enter:")

        if query:
            q = query.lower()
            computed_answer = None
            chart = None

            # ---------- Attempt direct pandas computations first ---------- #
            try:
                # TOTAL / SUM
                if "total" in q or "sum" in q:
                    words = q.split()
                    target_col = None
                    for w in words:
                        col_match = match_column(w, df.columns)
                        if col_match:
                            target_col = col_match
                            break

                    if target_col:
                        # coerce to numeric safely
                        total_val = pd.to_numeric(df[target_col], errors="coerce").sum()
                        computed_answer = pd.DataFrame({"Column": [target_col], "Total": [total_val]})
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

                # AVERAGE / MEAN
                elif "average" in q or "mean" in q:
                    words = q.split()
                    target_col = None
                    for w in words:
                        col_match = match_column(w, df.columns)
                        if col_match:
                            target_col = col_match
                            break

                    if target_col:
                        avg_val = pd.to_numeric(df[target_col], errors="coerce").mean()
                        computed_answer = pd.DataFrame({"Column": [target_col], "Average": [avg_val]})
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

                # GROUP BY (improved prioritization)
                elif "group" in q or " by " in q or "wise" in q:
                    words = q.replace("wise", "by").split()
                    group_candidate = None
                    value_candidate = None

                    for w in words:
                        col_match = match_column(w, df.columns)
                        if col_match:
                            low = col_match.lower()
                            # Prioritize grouping columns if name-like
                            if any(k in low for k in ["name", "dept", "category", "type", "group", "region", "center", "cost center", "cost_center", "account"]):
                                group_candidate = col_match
                            # Prioritize numeric/value columns
                            elif any(k in low for k in ["amount", "value", "price", "cost", "lc", "total", "amt"]):
                                value_candidate = col_match
                            else:
                                if group_candidate is None:
                                    group_candidate = col_match
                                else:
                                    value_candidate = col_match

                    # fallback numeric column
                    if value_candidate is None:
                        numeric_cols = df.select_dtypes(include="number").columns
                        if len(numeric_cols) > 0:
                            value_candidate = numeric_cols[0]

                    if group_candidate and value_candidate:
                        # coerce numeric column before aggregation
                        df[value_candidate] = pd.to_numeric(df[value_candidate], errors="coerce")
                        computed_answer = df.groupby(group_candidate)[value_candidate].sum().reset_index()
                        st.markdown(f"### 📝 {value_candidate} by {group_candidate}")
                        st.write(computed_answer)
                        download_button(computed_answer, "groupby_results.csv")
                        chart = px.bar(computed_answer, x=group_candidate, y=value_candidate,
                                       title=f"{value_candidate} by {group_candidate}")

                # GRAPH/CHART REQUESTS
                elif any(k in q for k in ["chart", "graph", "bar", "pie", "line", "hist", "histogram"]):
                    numeric_cols = df.select_dtypes(include="number").columns
                    if len(numeric_cols) == 0:
                        st.warning("No numeric columns available for charting.")
                    else:
                        # Choose columns intelligently: attempt to detect grouping column in query, else use first column as x
                        x_col = None
                        for w in q.split():
                            m = match_column(w, df.columns)
                            if m and m not in numeric_cols:
                                x_col = m
                                break
                        if x_col is None:
                            x_col = df.columns[0]

                        y_col = None
                        for w in q.split():
                            m = match_column(w, df.columns)
                            if m and m in numeric_cols:
                                y_col = m
                                break
                        if y_col is None:
                            y_col = numeric_cols[0]

                        if "bar" in q:
                            chart = px.bar(df, x=x_col, y=y_col, title=f"Bar: {y_col} by {x_col}")
                        elif "pie" in q:
                            chart = px.pie(df, names=x_col, values=y_col, title=f"Pie: {y_col} by {x_col}")
                        elif "line" in q:
                            chart = px.line(df, x=x_col, y=y_col, title=f"Line: {y_col} by {x_col}")
                        else:
                            chart = px.histogram(df, x=y_col, title=f"Histogram of {y_col}")

                        st.plotly_chart(chart, use_container_width=True)

            except Exception as e:
                st.warning(f"Couldn't compute directly: {e}")

            # ---------- AI Explanation (optional) ---------- #
            if use_ai and API_KEY:
                explanation = None
                prompt = f"""You are a data analyst. The dataframe has {len(df)} rows and {len(df.columns)} columns.
Columns: {list(df.columns)}
Query: {query}
Show a short explanation/insight only. Computed results (if any) are already shown above."""

                if provider == "Gemini":
                    # Robust Gemini handling - try multiple SDK invocation styles
                    try:
                        import google.generativeai as genai
                    except Exception as e:
                        st.error(f"Missing google-generativeai library or import failed: {e}")
                        explanation = None
                    else:
                        try:
                            genai.configure(api_key=API_KEY)
                        except Exception as e:
                            st.error(f"Failed to configure google.generativeai with provided key: {e}")
                            explanation = None
                        else:
                            model_name = model_input.strip() or "gemini-2.5-flash"
                            explanation_text = None
                            last_err = None

                            # Try invocation 1: generate_text
                            try:
                                resp = genai.generate_text(model=model_name, prompt=prompt)
                                if hasattr(resp, "text"):
                                    explanation_text = resp.text
                                elif hasattr(resp, "candidates") and len(resp.candidates) > 0 and hasattr(resp.candidates[0], "content"):
                                    explanation_text = resp.candidates[0].content
                                else:
                                    explanation_text = str(resp)
                            except Exception as e:
                                last_err = e

                            # Try invocation 2: generate_content
                            if not explanation_text:
                                try:
                                    resp = genai.generate_content(model=model_name, input=prompt)
                                    # try to extract text field
                                    if hasattr(resp, "text"):
                                        explanation_text = resp.text
                                    elif hasattr(resp, "result") and hasattr(resp.result, "output"):
                                        outputs = getattr(resp.result, "output")
                                        if outputs and len(outputs) > 0:
                                            # output content may be list/dict/str - attempt safe extraction
                                            first = outputs[0]
                                            if isinstance(first, str):
                                                explanation_text = first
                                            elif isinstance(first, dict):
                                                # common key names: "content", "text"
                                                if "content" in first and isinstance(first["content"], str):
                                                    explanation_text = first["content"]
                                                else:
                                                    explanation_text = str(first)
                                            else:
                                                explanation_text = str(first)
                                    else:
                                        explanation_text = str(resp)
                                except Exception as e:
                                    last_err = e

                            # Try invocation 3: genai.models.generate(...)
                            if not explanation_text:
                                try:
                                    if hasattr(genai, "models") and hasattr(genai.models, "generate"):
                                        resp = genai.models.generate(model=model_name, prompt=prompt)
                                        # attempt to extract text
                                        if hasattr(resp, "output"):
                                            explanation_text = str(resp.output)
                                        else:
                                            explanation_text = str(resp)
                                except Exception as e:
                                    last_err = e

                            if explanation_text:
                                explanation = explanation_text
                            else:
                                st.error("Could not extract text from Google SDK. Last SDK error: " + str(last_err))
                                explanation = None

                else:
                    # OpenAI-compatible providers: OpenRouter, DeepSeek, OpenAI
                    try:
                        from openai import OpenAI
                        base_url = provider_map.get(provider, {}).get("base_url")
                        client = OpenAI(base_url=base_url, api_key=API_KEY) if base_url else OpenAI(api_key=API_KEY)
                        model_name = default_model or "gpt-4o-mini"
                        resp = client.chat.completions.create(
                            model=model_name,
                            messages=[{"role": "user", "content": prompt}]
                        )
                        explanation = resp.choices[0].message.content
                    except Exception as e:
                        st.error(f"AI call failed: {e}")
                        explanation = None

                if explanation:
                    st.markdown("### 🤖 AI Explanation")
                    st.write(explanation)

            # ---------- Show chart if computed earlier ---------- #
            if chart:
                st.plotly_chart(chart, use_container_width=True)

    except Exception as e:
        st.error(f"Error reading file: {e}")

else:
    if not uploaded_file:
        st.info("Upload a CSV or Excel file to get started.")
    if use_ai and not API_KEY:
        st.warning("Provide an API key in the sidebar or in Streamlit Secrets (API_KEY) to enable AI explanations.")
