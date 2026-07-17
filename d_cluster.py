#Group 2E Adv Python and AI - Cluster Analysis

import json
import numpy as np
import pandas as pd
import typer
from sklearn.cluster import KMeans

app = typer.Typer()

def summarizeclusters(df):
    summaries = []

    for clusterid, group in df.groupby("cluster"):
        numarticles = len(group)

        if "sentiment" in group.columns:
            avgsentiment = float(group["sentiment"].astype(float).mean())
        else:
            avgsentiment = None

        sampleheadlines = group["headline_main"].head(3).tolist()

        summaries.append(
            {
                "clusterid": int(clusterid),
                "numarticles": numarticles,
                "avgsentiment": avgsentiment,
                "sampleheadlines": sampleheadlines,
            }
        )

    return summaries


@app.command()
def run(
    inputfile: str = typer.Argument(..., help="Input CSV file."),
    outputfile: str = typer.Argument(..., help="Output CSV file."),
    summaryfile: str = typer.Argument(..., help="Output JSON file for cluster summaries."),
    embeddingcolumn: str = typer.Option("embedding", help="Column containing embeddings."),
    numclusters: int = typer.Option(3, help="Number of clusters."),
):
    df = pd.read_csv(inputfile)

    if embeddingcolumn not in df.columns:
        raise Exception(f"Column '{embeddingcolumn}' not found in file")

    if "headline_main" not in df.columns:
        raise Exception("Column 'headline_main' not found in file")

    # convert embedding strings back into lists
    dfembedding = df[embeddingcolumn].apply(json.loads)

    # remove rows with empty embeddings
    validmask = dfembedding.apply(lambda x: isinstance(x, list) and len(x) > 0)
    df = df[validmask].copy()
    dfembedding = dfembedding[validmask]

    if len(df) == 0:
        raise Exception("No valid embeddings found")

    # Convert list-of-lists in dfembedding into a 2D NumPy array
    embmatrix = np.vstack(dfembedding.values)

    # default 3 clusters, but goes no more than the total num of articles provided
    NUMCLUSTERS = min(numclusters, len(df))

    kmeans = KMeans(
        n_clusters=NUMCLUSTERS,
        random_state=42,
        n_init="auto",
    )

    clusterlabels = kmeans.fit_predict(embmatrix)

    # Attach cluster labels to DataFrame
    df["cluster"] = clusterlabels

    clustersummaries = summarizeclusters(df)

    df.to_csv(outputfile, index=False)

    with open(summaryfile, "w", encoding="utf-8") as f:
        json.dump(clustersummaries, f, indent=2)

    typer.echo("Cluster analysis complete.")
    typer.echo(f"Output saved to {outputfile}")
    typer.echo(f"Cluster summaries saved to {summaryfile}")
    typer.echo(f"Number of clusters used: {NUMCLUSTERS}")


if __name__ == "__main__":
    app()