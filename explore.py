import marimo

__generated_with = "0.11.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    return mo, pl


@app.cell
def _():
    from sqlalchemy import create_engine
    import os
    DATABASE_URL = "sqlite:///data/social_network_anonymized.db"
    engine = create_engine(DATABASE_URL)
    return DATABASE_URL, create_engine, engine, os


@app.cell
def _(engine, mo):
    df = mo.sql(
        f"""
        select * from `Activity`
        """,
        output=False,
        engine=engine
    )
    return (df,)


@app.cell
def _():
    entries = ["commented-on-facebook", "shared-a-post-on-facebook", "posted-to-story-on-facebook"]
    return (entries,)


@app.cell
def _():
    from google import genai
    from google.genai import types
    import base64
    return base64, genai, types


@app.cell
def _(df, entries, pl, translate_with_gemini):
    df.filter(
        # filter entries with content
        (pl.col("type").is_in(entries)) & (pl.col("content") != "")
    )[:3].with_columns(pl.col("content").map_elements(translate_with_gemini))
    return


@app.cell
def _(genai):
    import enum
    from pydantic import BaseModel

    class TrafficLikelihood(enum.Enum):
        ONE = 1
        TWO = 2
        THREE = 3
        FOUR = 4
        FIVE = 5

    class SuspiciousActions(enum.Enum):
        S = "selling"
        B = "buying"
        AD = "advertising"
        HU = "hunting"
        O = "Other suspicious action"

    class Response(BaseModel):
        translated_content: str
        language: str
        traffic_likelihood: int
        species_being_mentioned: list[str]
        location: list[str]
        pii: list[str]
        actions: list[SuspiciousActions]


    client = genai.Client(
        vertexai=True,
        project="electricwin25lon-511",
        location="europe-west1",
    )

    def translate_with_gemini(text):
        return client.models.generate_content(
            model='gemini-2.0-flash',
            contents= f"""
            Provide all the answers in as much detail as is available
            Translate this text into English: {text}, 
            return the language being used
            then give it a likelihood rating of mentioning illegal animal trafficking,
            list out any animal species in being mentioned, 
            list out any location being mentioned,
            list out any personal identifiable information (pii) as: `typeofPII_PII` e.g name_Jack
            list out any suspicious actions
            """,
            config={
                'response_mime_type': 'application/json',
                'response_schema': Response,
            },
        ).text

    translate_with_gemini("hola, hablo español")
    return (
        BaseModel,
        Response,
        SuspiciousActions,
        TrafficLikelihood,
        client,
        enum,
        translate_with_gemini,
    )


@app.cell
def _(df, entries, os, pl, translate_with_gemini):
    if os.path.isfile("data/translated_posts.parquet"):
        unnested = pl.read_parquet("data/translated_posts.parquet")
    else:
        translated = df.filter(
            (pl.col("type").is_in(entries)) & (pl.col("content") != "")
        )[:500].with_columns(
            pl.col("content").map_elements(translate_with_gemini)
        )
        unnested = translated.select(pl.all().exclude("content"),  pl.col("content").str.json_decode()).unnest("content")
        unnested.write_parquet("translated_posts.parquet")
    return translated, unnested


@app.cell
def _(mo):
    mo.md(r"""## Visualising""")
    return


@app.cell
def _(engine, mo):
    pa = mo.sql(
        f"""
        SELECT * from `ProfileActivity`
        """,
        output=False,
        engine=engine
    )
    return (pa,)


@app.cell
def _(engine, mo):
    p = mo.sql(
        f"""
        select * from `Profiles`
        """,
        output=False,
        engine=engine
    )
    return (p,)


@app.cell
def _(p, pa, unnested):
    result = (
        p
        .join(
            pa,
            left_on="id",
            right_on="profile_id",
            how="inner"
        )
        .join(
            unnested,
            left_on="activity_id",
            right_on="id",
            how="inner",
            suffix="_activity"
        )
    )
    return (result,)


@app.cell
def _(pl, result):
    most_suspicious_people = result.group_by(
        "id", "name"
    ).agg(
        pl.col("traffic_likelihood").sum()
    ).sort(
        by = "traffic_likelihood",
        descending=True
    )
    return (most_suspicious_people,)


@app.cell
def _(mo, most_suspicious_people):
    row = mo.ui.table(most_suspicious_people, selection='single')
    row
    return (row,)


@app.cell
def _(selected):
    selected
    return


@app.cell
def _(pl, result, row):
    selected = result.filter(
            pl.col("id") == row.value["id"]
        ).with_columns(
            pl.from_epoch(pl.col("timestamp"), time_unit="ms").alias("datetime")
        )
    return (selected,)


@app.cell
def _(pl, selected):
    person_info = selected.group_by(
        "name", "profile_url", "region",  
    ).agg(
        pl.col("traffic_likelihood").sum(),
        pl.col("pii").explode(),
        pl.col("species_being_mentioned").explode(),
        pl.col("location").explode()
    )

    person_info
    return (person_info,)


@app.cell
def _():
    import altair as alt
    return (alt,)


@app.cell
def _(mo):
    mo.md(r"""## Over time suspiciousness""")
    return


@app.cell
def _(alt, mo, selected):
    _chart = alt.Chart(selected).mark_point().encode(
        x = "datetime",
        y = "traffic_likelihood"
    )

    c = mo.ui.altair_chart(_chart)
    c
    return (c,)


@app.cell
def _(c):
    c.value.select(
        "traffic_likelihood", "datetime", "translated_content", 
        "species_being_mentioned", "location", "pii", "actions"
    )
    return


if __name__ == "__main__":
    app.run()
