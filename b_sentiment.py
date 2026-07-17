#Group 2E Adv Python and AI - Sentiment Analysis

import pandas as pd
import typer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = typer.Typer(help="Run sentiment analysis on article text using VADER.")

analyzer = SentimentIntensityAnalyzer()

def get_sentiment_score(text: str) -> float:
    """
    Takes one piece of article text and returns the VADER compound sentiment score.

    The compound score ranges from:
    - -1 = very negative
    -  0 = neutral
    -  1 = very positive

    We use the article's combined text column from the notebook process.
    """
    # If the text is missing or blank, return 0.0 so the program does not crash.
    if pd.isna(text) or not str(text).strip():
        return 0.0

    scores = analyzer.polarity_scores(str(text))
    return scores["compound"]

def get_sentiment_label(score: float) -> str:
    """
    Converts a numeric sentiment score into a label.

    We use common VADER cutoffs:
    - score >= 0.05  -> positive
    - score <= -0.05 -> negative
    - otherwise      -> neutral
    """
    if score >= 0.05:
        return "positive"
    if score <= -0.05:
        return "negative"
    return "neutral"

@app.command()
def run(
    input_file: str = typer.Argument(..., help="Path to input CSV file."),
    output_file: str = typer.Argument(..., help="Path to output CSV file."),
    text_column: str = typer.Option("text","--text-column",help="Column containing article text.")
) -> None:
    """
    This function:
    1. Reads the input CSV file
    2. Checks that the text column exists
    3. Calculates sentiment scores
    4. Adds sentiment labels
    5. Saves the updated CSV
    6. Prints a small summary to the console
    """

    # Read the CSV file into a pandas DataFrame.
    df = pd.read_csv(input_file)

    # Makes sure the chosen text column actually exists. If not, stop the program and show a helpful error
    if text_column not in df.columns:
        raise typer.BadParameter(
            f"Column '{text_column}' not found in input file. "
            f"Available columns: {list(df.columns)}"
        )

    df["sentiment"] = df[text_column].apply(get_sentiment_score)
    df["sentiment_label"] = df["sentiment"].apply(get_sentiment_label)
    
    df.to_csv(output_file, index=False)

    typer.echo("Sentiment analysis complete.")
    typer.echo(f"Input rows: {len(df)}")
    typer.echo(f"Output saved to: {output_file}")

    summary = df["sentiment_label"].value_counts().to_dict()
    avg_score = df["sentiment"].mean()

    # Print the summary results
    typer.echo(f"Average sentiment score: {avg_score:.4f}")
    typer.echo(f"Label counts: {summary}")


if __name__ == "__main__":
    app()
