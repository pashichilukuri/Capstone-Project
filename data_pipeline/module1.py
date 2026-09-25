
import requests
from bs4 import BeautifulSoup
import csv
import time
import re
from urllib.parse import urljoin
import numpy as np
import pandas as pd

base_url = "https://books.toscrape.com/"
categorical_url = urljoin("base_url", "catalogue/")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/153.0.0.0 Safari/537.36"
}

RATING_MAP = {
    "One": "One",
    "Two": "Two",
    "Three": "Three",
    "Four": "Four",
    "Five": "Five"
}

def get_soup(url):
    """
    Download a webpage and return a BeautifulSoup object.
    If the request fails, return None instead of crashing.
    """

    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=15
        )

        # Check HTTP status code
        response.raise_for_status()

        return BeautifulSoup(
            response.text,
            "html.parser"
        )

    except requests.RequestException as e:
        print(f"Request failed: {url}")
        print(f"Error: {e}")

        return None

all_books = []

for page_number in range(1, 6):

    # Build an absolute URL
    if page_number == 1:
        page_url = (
            "https://books.toscrape.com/"
            "index.html"
        )
    else:
        page_url = (
            "https://books.toscrape.com/"
            f"catalogue/page-{page_number}.html"
        )

    print(f"\nScraping page {page_number}")
    print(page_url)

    soup = get_soup(page_url)

    if soup is None:
        print("Skipping this page.")
        continue

    books = soup.select("article.product_pod")

    print(f"Books found: {len(books)}")

    for book in books:

        # ----------------------------------------
        # TITLE
        # ----------------------------------------

        title_tag = book.select_one("h3 a")

        if title_tag:
            title = title_tag.get("title")
        else:
            title = None


        # ----------------------------------------
        # PRICE
        # ----------------------------------------

        price_tag = book.select_one(".price_color")

        if price_tag:
            price = price_tag.get_text(strip=True)
        else:
            price = None


        # ----------------------------------------
        # STAR RATING
        # ----------------------------------------

        rating_tag = book.select_one(".star-rating")

        if rating_tag:

            rating_classes = rating_tag.get(
                "class",
                []
            )

            star_rating = next(
                (
                    value
                    for value in RATING_MAP
                    if value in rating_classes
                ),
                None
            )

        else:
            star_rating = None


        # ----------------------------------------
        # AVAILABILITY
        # ----------------------------------------

        availability_tag = book.select_one(
            ".availability"
        )

        if availability_tag:
            availability = availability_tag.get_text(
                " ",
                strip=True
            )
        else:
            availability = None


        # ----------------------------------------
        # CATEGORY
        # ----------------------------------------

        category = None

        if title_tag:

            relative_book_url = title_tag.get(
                "href"
            )

            if relative_book_url:

                book_url = urljoin(
                    page_url,
                    relative_book_url
                )

                detail_soup = get_soup(
                    book_url
                )

                if detail_soup:

                    breadcrumb = detail_soup.select(
                        ".breadcrumb li"
                    )

                    if len(breadcrumb) >= 3:

                        category = breadcrumb[
                            2
                        ].get_text(strip=True)


        # ----------------------------------------
        # STORE RECORD
        # ----------------------------------------

        all_books.append({
            "title": title,
            "price": price,
            "star_rating": star_rating,
            "availability": availability,
            "category": category
        })
        
df = pd.DataFrame(all_books)

print("\n--------------------------------")
print(f"Total books scraped: {len(all_books)}")
print("--------------------------------")

# Display first 10 records
print("\nFirst 10 all_books:")
print(df.head(10).to_string(index=False))

# Basic validation
print("\nDataset shape:")
print(df.shape)

print("\nBooks per category:")
print(df["category"].value_counts())

print("\nMissing values:")
print(df.isnull().sum())

#Convert price to price_gbp

df["price_gbp"] = (
    df["price"]
    .astype("string")
    .str.replace("Â£", "", regex=False)
    .str.strip()
)

df["price_gbp"] = pd.to_numeric(
    df["price_gbp"],
    errors="coerce"
)

df[["price", "price_gbp"]].head()

#Convert price_gbp to a price_inr column using the project's fixed baseline conversion rate:
# Required fixed project conversion rate

GBP_TO_INR = 105.50

# Convert GBP to INR
df["price_inr"] = (df["price_gbp"] * GBP_TO_INR).round(2)

# Display the result
df[["price_gbp", "price_inr"]].head(10)

#Convert star rating
df["rating"] = df["star_rating"].map(
    RATING_MAP
)

df[["star_rating", "rating"]].head()
#Parse availability
df["in_stock"] = (
    df["availability"]
    .astype("string")
    .str.strip()
    .str.lower()
    .str.startswith("in stock")
)

df[[
    "availability",
    "in_stock"
]].head()

#Handle parsing failures
numeric_columns = [
    "price_gbp",
    "rating"
]

for column in numeric_columns:

    missing_count = df[column].isna().sum()

    if missing_count > 0:

        median_value = df[column].median()

        df[column] = df[column].fillna(
            median_value
        )

        print(
            f"{column}: "
            f"{missing_count} missing values "
            f"replaced with median {median_value}"
        )
required_columns = [
    "title",
    "star_rating",
    "availability",
    "category"
]

before = len(df)

df = df.dropna(
    subset=required_columns
).copy()

after = len(df)

print(
    f"Dropped {before - after} rows "
    "with missing essential text fields."
)
#Arrange the final columns
Arrange_df = df[
    [
        "title",
        "price",
        "price_gbp",
        "price_inr",
        "star_rating",
        "rating",
        "availability",
        "in_stock",
        "category"
    ]
]

df.head(10)

#covert to DataFrame
df =pd.DataFrame(all_books)

# Convert to DataFrame
df = pd.DataFrame(all_books)

# Save dataset
df.to_csv(
    "books_to_scrape_dataset.csv",
    index=False
)

''''sql data base creation and insertion of data into the database'''

import sqlite3
import pandas as pd

# ---------------------------------------------------------
# Create SQLite database
# ---------------------------------------------------------

conn = sqlite3.connect("books.db")

# Enable foreign-key enforcement
conn.execute("PRAGMA foreign_keys = ON")

cursor = conn.cursor()


# ---------------------------------------------------------
# Drop existing tables if re-running the notebook
# ---------------------------------------------------------

cursor.execute("DROP TABLE IF EXISTS books")
cursor.execute("DROP TABLE IF EXISTS categories")


# ---------------------------------------------------------
# Create categories table
# ---------------------------------------------------------

cursor.execute("""
CREATE TABLE categories (
    category_id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT UNIQUE NOT NULL
)
""")


# ---------------------------------------------------------
# Create books table
# ---------------------------------------------------------

cursor.execute("""
CREATE TABLE books (
    book_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    price_gbp REAL,
    price_inr REAL,
    rating INTEGER,
    in_stock INTEGER,
    category_id INTEGER NOT NULL,

    FOREIGN KEY (category_id)
        REFERENCES categories(category_id)
)
""")


conn.commit()

print("SQLite database and tables created successfully.")

# Get unique categories
categories_df = (
    df[["category"]]
    .drop_duplicates()
    .rename(columns={"category": "category_name"})
    .reset_index(drop=True)
)

print(categories_df)

categories_df.to_sql(
    "categories",
    conn,
    if_exists="append",
    index=False
)

print("Categories inserted successfully.")

pd.read_sql(
    "SELECT * FROM categories",
    conn
)

category_lookup = pd.read_sql(
    """
    SELECT category_id, category_name
    FROM categories
    """,
    conn
)

category_lookup.head()

books_df = df.merge(
    category_lookup,
    left_on="category",
    right_on="category_name",
    how="left"
)

books_df.head()

books_to_insert = books_df[
    [
        "title",
        "price_gbp",
        "price_inr",
        "rating",
        "in_stock",
        "category_id"
    ]
].copy()

books_to_insert["in_stock"] = (
    books_to_insert["in_stock"]
    .astype(int)
)
books_to_insert.to_sql(
    "books",
    conn,
    if_exists="append",
    index=False
)

conn.commit()

print(
    f"{len(books_to_insert)} books inserted successfully."
)

query1 = """
SELECT
    book_id,
    title,
    price_gbp,
    rating,
    in_stock
FROM books
WHERE rating = 5
"""

result1 = pd.read_sql(query1, conn)

print(result1)

query2 = """
SELECT
    title,
    price_gbp,
    price_inr,
    rating
FROM books
ORDER BY price_gbp DESC
LIMIT 10
"""

result2 = pd.read_sql(query2, conn)

print(result2)

query3 = """
SELECT DISTINCT category_name
FROM categories
ORDER BY category_name
"""

result3 = pd.read_sql(query3, conn)

print(result3)

query4 = """
SELECT
    title,
    price_gbp,
    rating
FROM books
WHERE price_gbp BETWEEN 20 AND 30
ORDER BY price_gbp
"""

result4 = pd.read_sql(query4, conn)

print(result4)

query5 = """
SELECT
    title,
    price_gbp,
    rating,
    in_stock
FROM books
WHERE rating IN (4, 5)
ORDER BY rating DESC, price_gbp DESC
"""

result5 = pd.read_sql(query5, conn)

print(result5)

query6 = """
SELECT
    b.title,
    c.category_name,
    b.price_gbp,
    b.price_inr,
    b.rating,
    b.in_stock
FROM books AS b
JOIN categories AS c
    ON b.category_id = c.category_id
ORDER BY b.rating DESC, b.price_gbp DESC
LIMIT 10
"""

result6 = pd.read_sql(query6, conn)

print(result6)

queries = {
    "query_1_select_where": query1,
    "query_2_order_by_limit": query2,
    "query_3_distinct": query3,
    "query_4_between": query4,
    "query_5_in": query5,
    "query_6_join": query6
}

results = {
    "query_1_select_where": result1,
    "query_2_order_by_limit": result2,
    "query_3_distinct": result3,
    "query_4_between": result4,
    "query_5_in": result5,
    "query_6_join": result6
}

for name in queries:

    print("\n" + "=" * 70)
    print(name.upper())
    print("=" * 70)

    print("\nSQL:")
    print(queries[name])

    print("\nOUTPUT:")
    print(results[name].to_string(index=False))
    
    df_query2 = pd.read_sql(
    query2,
    conn
)

print(df_query2)

df_query4 = pd.read_sql(
    query4,
    conn
)

print(df_query4)

merge_result = df.merge(
    category_lookup,
    left_on="category",
    right_on="category_name",
    how="inner"
)


merge_result = merge_result[
    [
        "title",
        "category_name",
        "price_gbp",
        "price_inr",
        "rating",
        "in_stock"
    ]
].copy()

merge_result = merge_result.rename(
    columns={
        "category_name": "category_name"
    }
)

merge_result = (
    merge_result
    .sort_values(
        ["rating", "price_gbp"],
        ascending=[False, False]
    )
    .head(10)
    .reset_index(drop=True)
)

print(merge_result)

sql_join_result = result6.copy()

sql_join_result = sql_join_result.reset_index(
    drop=True
)

merge_result = merge_result.reset_index(
    drop=True
)

print("SQL JOIN result:")
print(sql_join_result)

print("\npd.merge() result:")
print(merge_result)

equivalent = sql_join_result.equals(
    merge_result
)

print(
    "\nAre the SQL JOIN and pd.merge() "
    "results equivalent?",
    equivalent
)

equivalent = (
    sql_join_result.astype(str).equals(
        merge_result.astype(str)
    )
)

print(
    "Equivalent after normalizing data types:",
    equivalent
)

# Check that every book has a valid category
orphaned_books = pd.read_sql(
    """
    SELECT b.*
    FROM books b
    LEFT JOIN categories c
        ON b.category_id = c.category_id
    WHERE c.category_id IS NULL
    """,
    conn
)

print(
    "Books with invalid category references:",
    len(orphaned_books)
)

total_books = pd.read_sql(
    "SELECT COUNT(*) AS total_books FROM books",
    conn
)

total_categories = pd.read_sql(
    "SELECT COUNT(*) AS total_categories FROM categories",
    conn
)

print(total_books)
print(total_categories)

conn.close()

print("Database connection closed.")