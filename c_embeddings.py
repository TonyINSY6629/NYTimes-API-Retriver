#Group 2E Adv Python and AI - Embeddings

import os
import json
import pandas as pd
import typer
from dotenv import load_dotenv
from google import genai

app = typer.Typer()

GEMINIMODEL = "gemini-embedding-001"
EMBEDDINGCOLUMNNAME = "embedding"


def getembedding(client, text):
    if pd.isna(text) or not str(text).strip():
        return []

    response = client.models.embed_content(
        model=GEMINIMODEL,
        contents=str(text),
    )

    vector = response.embeddings[0].values
    return vector


@app.command()
def run(
    inputfile: str = typer.Argument(..., help="Input CSV file."),
    outputfile: str = typer.Argument(..., help="Output CSV file."),
    envfile: str = typer.Option("envvars.env", help="Env file name."),
    textcolumn: str = typer.Option("text", help="Column containing text to embed."),
):
    load_dotenv(envfile)
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    if not GEMINI_API_KEY:
      raise Exception('GEMINI_API_KEY not found')

    client = genai.Client(api_key=GEMINI_API_KEY)

    df = pd.read_csv(inputfile)

    if textcolumn not in df.columns:
        raise Exception(f"Column '{textcolumn}' not found in file")

    # generate embeddings for all articles
    dfembedding = df[textcolumn].apply(lambda text: getembedding(client, text))

    # save embeddings back into dataframe
    df[EMBEDDINGCOLUMNNAME] = dfembedding.apply(json.dumps)

    df.to_csv(outputfile, index=False)

    typer.echo("Embedding generation complete.")
    typer.echo(f"Output saved to {outputfile}")

    if len(dfembedding) > 0 and len(dfembedding.iloc[0]) > 0:
        typer.echo(f"Embedding vector length: {len(dfembedding.iloc[0])}")


if __name__ == "__main__":
    app()