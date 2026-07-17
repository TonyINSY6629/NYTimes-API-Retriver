import json
import typer
import pandas as pd

from a_data_api import fetcharticles
from b_sentiment import get_sentiment_score, get_sentiment_label
from c_embeddings import getembedding
from d_cluster import summarizeclusters
from e_genai import summarize_all

from dotenv import load_dotenv
import os
from google import genai
import numpy as np
from sklearn.cluster import KMeans

app = typer.Typer(help="Run the full article intelligence pipeline.")


@app.command()
def run():
    typer.echo("Welcome to the Articles to Actionable Insights Intelligence Tool")

    company = typer.prompt("Enter company of interest")
    begin_date = typer.prompt("Enter beginning date (YYYYMMDD)")
    section = typer.prompt("Enter section (Business or Technology)")
    envfile = typer.prompt("Enter env file name", default="envvars.env")

    typer.echo("\nStep 1: Fetching articles...")
    while True:
        try:
            df = fetcharticles(
                company=company,
                begin_date=begin_date,
                section=section,
                envfile=envfile,
            )
            break
        except ValueError as e:
            typer.echo(f"Error: {e}")
            # Decide which input to re-prompt for based on the error message
            if "S&P 500" in str(e):
                company = typer.prompt("Please enter a valid S&P 500 company")
            else:
                begin_date = typer.prompt("Please enter an earlier start date (YYYYMMDD)")

    typer.echo(f"Retrieved {len(df)} articles.")

    typer.echo("\nStep 2: Running sentiment analysis...")
    df["sentiment"] = df["text"].apply(get_sentiment_score)
    df["sentiment_label"] = df["sentiment"].apply(get_sentiment_label)
    typer.echo("Sentiment analysis complete.")

    typer.echo("\nStep 3: Generating embeddings...")
    load_dotenv(envfile)
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    if not GEMINI_API_KEY:
      raise Exception('GEMINI_API_KEY not found')

    client = genai.Client(api_key=GEMINI_API_KEY)
    df["embedding"] = df["text"].apply(lambda text: json.dumps(getembedding(client, text)))
    typer.echo("Embedding generation complete.")

    typer.echo("\nStep 4: Running clustering...")
    dfembedding = df["embedding"].apply(json.loads)

    validmask = dfembedding.apply(lambda x: isinstance(x, list) and len(x) > 0)
    df = df[validmask].copy()
    dfembedding = dfembedding[validmask]

    if len(df) == 0:
        raise Exception("No valid embeddings found")

    embmatrix = np.vstack(dfembedding.values)

    numclusters = min(3, len(df))
    kmeans = KMeans(n_clusters=numclusters, random_state=42, n_init="auto")
    clusterlabels = kmeans.fit_predict(embmatrix)

    df["cluster"] = clusterlabels
    clustersummaries = summarizeclusters(df)
    typer.echo("Cluster analysis complete.")

    typer.echo("\nStep 5: Generating GenAI summary...")
    sentimentsummary = {
        "numarticles": len(df),
        "numpositive": int((df["sentiment"] > 0).sum()),
        "numnegative": int((df["sentiment"] < 0).sum()),
        "numneutral": int((df["sentiment"] == 0).sum()),
        "avgsentiment": float(df["sentiment"].mean()),
    }

    daterangestr = f"{begin_date} to today"

    clusterinsightstext = summarize_all(
        client=client,
        company=company,
        section=section,
        daterangestr=daterangestr,
        sentimentsummary=sentimentsummary,
        clustersummaries=clustersummaries,
    )

    typer.echo("GenAI summary complete.")

    typer.echo("\nSaving output files...")
    df.to_csv("final_output.csv", index=False)

    with open("cluster_summaries.json", "w", encoding="utf-8") as f:
        json.dump(clustersummaries, f, indent=2)

    with open("analysis.txt", "w", encoding="utf-8") as f:
        f.write(clusterinsightstext)

    typer.echo("All steps completed successfully.")
    typer.echo("Saved files:")
    typer.echo("- final_output.csv")
    typer.echo("- cluster_summaries.json")
    typer.echo("- analysis.txt")


if __name__ == "__main__":
    run()