import marimo

__generated_with = "0.11.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    from sqlalchemy import create_engine
    import os
    DATABASE_URL = "sqlite:///data/social_network_anonymized.db"
    engine = create_engine(DATABASE_URL)
    return DATABASE_URL, create_engine, engine, os


@app.cell
def _(engine, media, mo):
    _df = mo.sql(
        f"""
        select * from media limit 6
        """,
        engine=engine
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
