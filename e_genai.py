#Group 2E Adv Python and AI - GenAI & Summaries

import os
import json
import pandas as pd
import typer
from dotenv import load_dotenv
from google import genai

app = typer.Typer()


def formatclustersforprompt(clustersummaries):
    lines = []

    for c in clustersummaries:
        lines.append(f"Cluster {c['clusterid']}")
        lines.append(f"- Number of articles: {c['numarticles']}")

        if c["avgsentiment"] is not None:
            lines.append(f"- Average sentiment: {c['avgsentiment']:.3f}")
        else:
            lines.append("- Average sentiment: N/A")

        lines.append("- Sample headlines:")

        for h in c["sampleheadlines"]:
            lines.append(f"  - {h}")

        lines.append("")

    return "\n".join(lines)


def summarize_all(
    client,
    company,
    section,
    daterangestr,
    sentimentsummary,
    clustersummaries,
    model="gemini-2.5-flash-lite",
):
    sentimentblock = (
        f"- Total articles: {sentimentsummary['numarticles']}\n"
        f"- Positive: {sentimentsummary['numpositive']}\n"
        f"- Negative: {sentimentsummary['numnegative']}\n"
        f"- Neutral: {sentimentsummary['numneutral']}\n"
        f"- Average sentiment: {sentimentsummary['avgsentiment']:.3f}"
    )

    clustersblock = formatclustersforprompt(clustersummaries)

    prompt = f"""
You are a strategy analyst summarizing themes in recent New York Times coverage.

Company of interest: {company}
Section/industry: {section}
Date range searched: {daterangestr}

Overall sentiment across all articles:
{sentimentblock}

The articles have been grouped into semantic clusters based on text embeddings.
Each cluster represents a recurring theme in the coverage.

{clustersblock}

Using this information, please provide:

1. A brief overview of the main themes in the coverage. Refer to clusters by their numbers.
2. How the section/industry appears to be doing overall, based on these themes.
3. How {company} is being portrayed across these themes (e.g., which clusters are most relevant, and whether they are positive, negative, or mixed).
4. 2-3 key risks and 2-3 key opportunities for {company} or this industry implied by these clusters.
5. A short executive-style summary (3-5 sentences) that a consultant could present to a client.

Use clear headings in your response, such as:
- Themes Overview
- Industry State
- Company Positioning
- Risks & Opportunities
- Executive Summary
"""

    response = client.models.generate_content(
        model=model,
        contents=prompt.strip(),
    )

    return response.text


@app.command()
def run(
    inputfile: str = typer.Argument(..., help="CSV file with clustered article data."),
    summaryfile: str = typer.Argument(..., help="JSON file with cluster summaries."),
    outputfile: str = typer.Argument(..., help="Output text file for Gemini analysis."),
    company: str = typer.Option(..., help="Company of interest."),
    section: str = typer.Option(..., help="Section or industry searched."),
    daterangestr: str = typer.Option(..., help="Date range searched."),
    envfile: str = typer.Option("envvars.env", help="Env file name."),
    model: str = typer.Option("gemini-2.5-flash-lite", help="Gemini model name."),
):
    load_dotenv(envfile)
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    if not GEMINI_API_KEY:
      raise Exception('GEMINI_API_KEY not found')
    
    client = genai.Client(api_key=GEMINI_API_KEY)

    df = pd.read_csv(inputfile)

    with open(summaryfile, "r", encoding="utf-8") as f:
        clustersummaries = json.load(f)

    if "sentiment" not in df.columns:
        raise Exception("Column 'sentiment' not found in file")

    dfsentiment = pd.to_numeric(df["sentiment"], errors="coerce").fillna(0)

    sentimentsummary = {
        "numarticles": len(df),
        "numpositive": int((dfsentiment > 0).sum()),
        "numnegative": int((dfsentiment < 0).sum()),
        "numneutral": int((dfsentiment == 0).sum()),
        "avgsentiment": float(dfsentiment.mean()),
    }

    clusterinsightstext = summarize_all(
        client=client,
        company=company,
        section=section,
        daterangestr=daterangestr,
        sentimentsummary=sentimentsummary,
        clustersummaries=clustersummaries,
        model=model,
    )


    with open(outputfile, "w", encoding="utf-8") as f:
        f.write(clusterinsightstext)

    typer.echo("GenAI analysis complete.")
    typer.echo(f"Output saved to {outputfile}")
    typer.echo("")
    typer.echo(clusterinsightstext[:1000])


if __name__ == "__main__":
    app()