#Group 2E Adv Python and AI - data & nyt api

import requests
import os
from dotenv import load_dotenv
import datetime as dt
import pandas as pd
from urllib.parse import urlencode, quote
from collections import Counter

# Load S&P 500 company list
_SP500_FILE = os.path.join(os.path.dirname(__file__), "sp500_companies_clean.txt")
try:
    with open(_SP500_FILE, encoding="utf-8") as f:
        SP500_COMPANIES = {line.strip().lower() for line in f if line.strip()}
except FileNotFoundError:
    SP500_COMPANIES = set()


def datestr(d: dt.date) -> str:
    return d.strftime("%Y%m%d")


def fetcharticles(company, begin_date, section, envfile="env_vars.env"):
    if envfile and os.path.isfile(envfile):
        load_dotenv(envfile)
        
    NYTIMES_KEY = os.getenv('NYTIMES_KEY')
    if not NYTIMES_KEY:
      raise Exception('NYTIMES_KEY not found')

    if not SP500_COMPANIES:
        raise Exception("S&P 500 company list could not be loaded. Check that sp500_companies_clean.txt is present.")

    if company.strip().lower() not in SP500_COMPANIES:
        raise ValueError(
            f"'{company}' is not in the S&P 500 list. "
            "Please enter a valid S&P 500 company name (e.g. Apple, Microsoft, Tesla)."
        )

    today = dt.date.today()
    end_date_for_NYTimes_url = datestr(today)
    new_end_date = int(end_date_for_NYTimes_url) - 1

    if not begin_date.isdigit() or len(begin_date) != 8:
        raise Exception("Invalid format! Please enter exactly 8 digits, e.g. 20260101.")

    if int(begin_date) > new_end_date:
        raise Exception("Begin date cannot be later than the end date.")

    selected_section = section

    if selected_section == "Business":
        desk_or_section_name = "desk"
    else:
        desk_or_section_name = "section_name"

    q = company
    qtext = q

    def get_data_json():
        params = {
            "begin_date": begin_date,
            "end_date": new_end_date,
            "fq": f'{desk_or_section_name} = {selected_section}, organization.contains = {q}',
            "q": qtext,
            "sort": "relevance",
            "api-key": NYTIMES_KEY,
        }

        print("Request URL params:")
        print(params)

        query_string = urlencode(params, quote_via=quote)
        url = f"https://api.nytimes.com/svc/search/v2/articlesearch.json?{query_string}"

        # later when you call the API:
        response = requests.get(url)
        print("#Full request URL:", response.url)  # shows encoded URL

        if response.status_code != 200:
            print(f"API Request failed with status code {response.status_code}")
            print("Response Text:", response.text)
            return {"response": {"docs": []}}

        try:
            data = response.json()
        except Exception as e:
            print("Failed to decode JSON. Response was:", response.text)
            return {"response": {"docs": []}}

        docs = data.get("response", {}).get("docs", [])
        total_docs = len(docs)
        print("Number of articles retrieved:", total_docs)
        return data

    data = get_data_json()

    # Building the related articles counter
    def related_articles_counter() -> int:
        count = 0
        target = q.strip().lower()
        if "response" in data and "docs" in data["response"]:
            for doc in data["response"]["docs"]:
                for kw in doc.get("keywords", []):
                    # NYT API uses "organizations", so we check case-insensitively
                    name = kw.get("name", "").lower()
                    if name in ["organizations", "organization"] and target in kw.get("value", "").lower():
                        count += 1
                        break  # count each document only once

        print(f"{q.strip()}: {count}")
        return count

    if related_articles_counter() == 0:
        raise ValueError(
            f"No articles related to '{q}' were found between {begin_date} and today. "
            "Please try an earlier start date."
        )

    def normalize_article(doc):
        headline = doc.get("headline") or {}
        keywords = doc.get("keywords") or []

        keyword_values = [kw.get("value") for kw in keywords if kw.get("value")]

        return {
            "web_url": doc.get("web_url"),
            "pub_date": doc.get("pub_date"),
            "section_name": doc.get("section_name"),
            "headline_main": headline.get("main"),
            "abstract": doc.get("abstract"),
            "snippet": doc.get("snippet"),
            "keywords": keyword_values,
        }

    # Filter documents to only include those related to the target company
    target = q.strip().lower()
    filtered_docs = []
    if "response" in data and "docs" in data["response"]:
        for doc in data["response"]["docs"]:
            for kw in doc.get("keywords", []):
                name = kw.get("name", "").lower()
                if name in ["organizations", "organization"] and target in kw.get("value", "").lower():
                    filtered_docs.append(doc)
                    break

    rows = [normalize_article(doc) for doc in filtered_docs]
    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df["text"] = (
        df["headline_main"].fillna("")
        + " "
        + df["abstract"].fillna("")
        + " "
        + df["snippet"].fillna("")
    )

    return df


if __name__ == "__main__":
    envfile = input("Please enter your API key env file name: ").strip()
    company = input("Enter company of interest: ").strip()
    begin_date = input("Please enter the wished beginning date, in YYYYMMDD: ").strip()
    section = input("Enter section (Business or Technology): ").strip()

    df = fetcharticles(company=company, begin_date=begin_date, section=section, envfile=envfile)
    print(df[["headline_main", "text"]])