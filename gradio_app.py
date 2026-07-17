import json
import os
import gradio as gr
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google import genai
from sklearn.cluster import KMeans

from a_data_api import fetcharticles
from b_sentiment import get_sentiment_score, get_sentiment_label
from c_embeddings import getembedding
from d_cluster import summarizeclusters
from e_genai import summarize_all


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def run_pipeline(company: str, begin_date: str, section: str, envfile: str):
    """
    Generator that yields (log_message, csv_path, json_path, txt_path) tuples
    so Gradio can stream progress to the log box and populate file outputs at
    the very end.
    """

    def step(msg: str):
        """Yield a progress line with no output files yet."""
        yield msg, None, None, None

    # ── Validate inputs ────────────────────────────────────────────────────
    if not company.strip():
        yield "❌ Please enter a company name.", None, None, None
        return
    if len(begin_date.strip()) != 8 or not begin_date.strip().isdigit():
        yield "❌ Date must be in YYYYMMDD format (e.g. 20240101).", None, None, None
        
    if envfile and not os.path.isfile(envfile):
        yield f"❌ Uploaded env file not found at '{envfile}'.\n", None, None, None
        return

    yield "✅ Inputs validated.\n", None, None, None

    # ── Step 1: Fetch articles ─────────────────────────────────────────────
    yield "⏳ Step 1/5 — Fetching articles…\n", None, None, None
    try:
        df = fetcharticles(
            company=company.strip(),
            begin_date=begin_date.strip(),
            section=section,
            envfile=envfile,
        )
        
    except Exception as e:
        yield f"❌ Failed to fetch articles: {e}\n", None, None, None
        return

    if df.empty:
        yield "⚠️ No articles returned. Stopping.\n", None, None, None
        return

    yield f"✅ Retrieved {len(df)} articles.\n", None, None, None

    # ── Step 2: Sentiment analysis ─────────────────────────────────────────
    yield "⏳ Step 2/5 — Running sentiment analysis…\n", None, None, None
    try:
        df["sentiment"] = df["text"].apply(get_sentiment_score)
        df["sentiment_label"] = df["sentiment"].apply(get_sentiment_label)
    except Exception as e:
        yield f"❌ Sentiment analysis failed: {e}\n", None, None, None
        return
    yield "✅ Sentiment analysis complete.\n", None, None, None

    # ── Step 3: Embeddings ─────────────────────────────────────────────────
    yield "⏳ Step 3/5 — Generating embeddings…\n", None, None, None
    try:
        if envfile and os.path.isfile(envfile):
            load_dotenv(envfile)
            
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise EnvironmentError("GEMINI_API_KEY not found in env file.")
        client = genai.Client(api_key=gemini_api_key)
        df["embedding"] = df["text"].apply(
            lambda text: json.dumps(getembedding(client, text))
        )
    except Exception as e:
        yield f"❌ Embedding generation failed: {e}\n", None, None, None
        return
    yield "✅ Embedding generation complete.\n", None, None, None

    # ── Step 4: Clustering ─────────────────────────────────────────────────
    yield "⏳ Step 4/5 — Running clustering…\n", None, None, None
    try:
        df_embedding = df["embedding"].apply(json.loads)
        valid_mask = df_embedding.apply(lambda x: isinstance(x, list) and len(x) > 0)
        df = df[valid_mask].copy()
        df_embedding = df_embedding[valid_mask]

        if len(df) == 0:
            raise ValueError("No valid embeddings found after filtering.")

        emb_matrix = np.vstack(df_embedding.values)
        num_clusters = min(3, len(df))
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init="auto")
        df["cluster"] = kmeans.fit_predict(emb_matrix)
        cluster_summaries = summarizeclusters(df)
    except Exception as e:
        yield f"❌ Clustering failed: {e}\n", None, None, None
        return
    yield "✅ Cluster analysis complete.\n", None, None, None

    # ── Step 5: GenAI summary ──────────────────────────────────────────────
    yield "⏳ Step 5/5 — Generating GenAI summary…\n", None, None, None
    try:
        sentiment_summary = {
            "numarticles": len(df),
            "numpositive": int((df["sentiment"] > 0).sum()),
            "numnegative": int((df["sentiment"] < 0).sum()),
            "numneutral": int((df["sentiment"] == 0).sum()),
            "avgsentiment": float(df["sentiment"].mean()),
        }
        date_range_str = f"{begin_date.strip()} to today"

        cluster_insights_text = summarize_all(
            client=client,
            company=company.strip(),
            section=section,
            daterangestr=date_range_str,
            sentimentsummary=sentiment_summary,
            clustersummaries=cluster_summaries,
        )
    except Exception as e:
        yield f"❌ GenAI summary failed: {e}\n", None, None, None
        return
    yield "✅ GenAI summary complete.\n", None, None, None

    # ── Save outputs ───────────────────────────────────────────────────────
    yield "💾 Saving output files…\n", None, None, None
    try:
        csv_path = "final_output.csv"
        json_path = "cluster_summaries.json"
        txt_path = "analysis.txt"

        df.to_csv(csv_path, index=False)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(cluster_summaries, f, indent=2)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(cluster_insights_text)
    except Exception as e:
        yield f"❌ Failed to save files: {e}\n", None, None, None
        return

    yield (
        "🎉 All steps completed successfully!\n\nDownload your results below.",
        csv_path,
        json_path,
        txt_path,
    )


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_ui():
    with gr.Blocks(
        title="Article Intelligence Pipeline",
        theme=gr.themes.Soft(),
    ) as demo:

        gr.Markdown(
            """
            # 📰 Article Intelligence Pipeline
            Fetch articles, run sentiment analysis, cluster by topic, and generate
            an AI-powered executive summary — all in one click.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Configuration")

                # Load S&P 500 list for the dropdown
                _sp500_path = os.path.join(os.path.dirname(__file__), "sp500_companies_clean.txt")
                with open(_sp500_path, encoding="utf-8") as f:
                    _sp500_choices = sorted([line.strip() for line in f if line.strip()])

                company_input = gr.Dropdown(
                    label="Company",
                    choices=_sp500_choices,
                    value=None,
                    filterable=True,  # lets the user type to search
                )

                begin_date_input = gr.Textbox(
                    label="Start Date",
                    placeholder="YYYYMMDD — e.g. 20240101",
                    max_lines=1,
                )
                section_input = gr.Dropdown(
                    label="Section",
                    choices=["Business", "Technology"],
                    value="Business",
                )
                envfile_input = gr.File(
                    label="Upload your .env file (with NYTIMES_KEY and GEMINI_API_KEY)",
                    file_types=[".env"],
                    type="filepath",
                )

                run_btn = gr.Button("▶ Run Pipeline", variant="primary", size="lg")

            with gr.Column(scale=2):
                gr.Markdown("### 📋 Progress Log")
                log_output = gr.Textbox(
                    label="",
                    lines=14,
                    max_lines=14,
                    interactive=False,
                )

        gr.Markdown("### 📥 Download Results")
        with gr.Row():
            csv_output = gr.File(label="📊 Article Data (CSV)", interactive=False)
            json_output = gr.File(label="🗂 Cluster Summaries (JSON)", interactive=False)
            txt_output = gr.File(label="📝 AI Analysis (TXT)", interactive=False)

        # Accumulate log lines so the box always shows the full history
        log_state = gr.State("")

        def run_and_stream(company, begin_date, section, envfile, log_so_far):
            for log_line, csv_path, json_path, txt_path in run_pipeline(
                company, begin_date, section, envfile
            ):
                log_so_far = log_so_far + log_line
                yield log_so_far, csv_path, json_path, txt_path, log_so_far

        run_btn.click(
            fn=lambda: ("", None, None, None, ""),  # clear outputs on each new run
            inputs=[],
            outputs=[log_output, csv_output, json_output, txt_output, log_state],
            queue=False,
        ).then(
            fn=run_and_stream,
            inputs=[
                company_input,
                begin_date_input,
                section_input,
                envfile_input,
                log_state,
            ],
            outputs=[log_output, csv_output, json_output, txt_output, log_state],
        )

    return demo


if __name__ == "__main__":
    build_ui().launch()
