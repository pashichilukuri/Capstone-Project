# Books to Scrape — Scraping, Cleaning & SQLite Pipeline

An end-to-end project that scrapes book data from [books.toscrape.com](https://books.toscrape.com/) (a sandbox site built for scraping practice), cleans and enriches it with `pandas`, loads it into a normalized **SQLite** database, and runs a series of SQL queries — validated against equivalent `pandas` operations.

## Project Structure

```
.
├── books_scraper.py / .ipynb        # main script/notebook (your code)
├── books_to_scrape_dataset.csv      # cleaned dataset exported from pandas
├── books.db                          # SQLite database (books + categories tables)
├── requirements.txt
└── README.md
```

## What the Code Does

### 1. Web Scraping
- Iterates over the first 5 listing pages of books.toscrape.com using `requests` + `BeautifulSoup`.
- For each book on a listing page, extracts:
  - **Title**, **price**, **star rating**, **availability** (from the listing card).
  - **Category**, by following the book's detail-page link and reading the breadcrumb trail.
- Wraps every HTTP request in a `get_soup()` helper with a custom `User-Agent`, a 15s timeout, and error handling so a failed request skips that page instead of crashing the run.
- Collects all records into a list of dicts (`all_books`).

### 2. Data Cleaning & Feature Engineering (pandas)
- Converts `all_books` into a DataFrame.
- **`price_gbp`**: strips the currency symbol from `price` and converts it to a numeric column.
- **`price_inr`**: converts GBP → INR using a fixed project rate (`GBP_TO_INR = 105.50`).
- **`rating`**: maps the scraped word-form star rating (`"One"`–`"Five"`) to a normalized value via `RATING_MAP`.
- **`in_stock`**: parses the `availability` text into a boolean (`True` if it starts with "In stock").
- Fills missing numeric values (`price_gbp`, `rating`) with the column median, logging how many values were imputed.
- Drops rows missing essential text fields (`title`, `star_rating`, `availability`, `category`).
- Reorders columns into a final, presentation-ready layout and saves the result to `books_to_scrape_dataset.csv`.

### 3. SQLite Database
Builds a small relational schema with foreign-key enforcement:
- **`categories`** — `category_id` (PK), `category_name` (unique).
- **`books`** — `book_id` (PK), `title`, `price_gbp`, `price_inr`, `rating`, `in_stock`, `category_id` (FK → `categories`).

Loads data in two steps:
1. Inserts distinct categories into `categories`, then reads back the generated `category_id`s.
2. Merges the book DataFrame with the category lookup table on category name, then inserts the resulting rows into `books`.

### 4. SQL Queries
Six queries demonstrating core SQL patterns, each run with `pd.read_sql`:
1. `SELECT ... WHERE rating = 5`
2. `SELECT ... ORDER BY price_gbp DESC LIMIT 10`
3. `SELECT DISTINCT category_name ... ORDER BY category_name`
4. `SELECT ... WHERE price_gbp BETWEEN 20 AND 30`
5. `SELECT ... WHERE rating IN (4, 5) ORDER BY rating DESC, price_gbp DESC`
6. `JOIN` between `books` and `categories`, ordered by rating and price, limited to 10 rows

### 5. SQL vs. pandas Cross-Check
- Reproduces the JOIN query (#6) using `pandas.merge()` instead of SQL.
- Compares the SQL result and the `pandas` result row-for-row with `.equals()`, then again after normalizing dtypes with `.astype(str)`, to confirm the two approaches agree.

### 6. Data Integrity Checks
- Checks for **orphaned books** — rows in `books` whose `category_id` doesn't match any row in `categories` (via a `LEFT JOIN ... WHERE category_id IS NULL`).
- Prints total row counts for `books` and `categories` as a final sanity check.
- Closes the database connection.

## Setup

```bash
pip install -r requirements.txt
```

`sqlite3` is part of Python's standard library, so it needs no separate install.

Run the script/notebook top to bottom; it will create `books_to_scrape_dataset.csv` and `books.db` in the working directory.

## Notes / Known Issues to Fix Before Running

The pasted code has a few problems that will cause it to fail or behave incorrectly as-is:

1. **`urljoin` is called on a string literal, not the `base_url` variable** (quotes around `base_url`), so `categorical_url` is wrong and unused anyway:
   ```python
   categorical_url = urljoin("base_url", "catalogue/")   # "base_url" is a literal string, not the variable
   ```
   This line isn't actually needed by the rest of the script and can be removed, or fixed to `urljoin(base_url, "catalogue/")`.

2. **`df` is used before it's created.** The block that prints `df.head(10)`, `df.shape`, `df["category"].value_counts()`, etc. runs *before* `df = pd.DataFrame(all_books)` is defined later in the script. Move the line
   ```python
   df = pd.DataFrame(all_books)
   ```
   to **immediately after** the scraping loop finishes (right after `print(f"Total books scraped: {len(all_books)}")`), before any `df.*` calls.

3. **`df = pd.DataFrame(all_books)` appears twice** near the CSV-export step — the duplicate can be removed once the DataFrame is created earlier (see #2).

4. **Column reordering happens before the DataFrame is (re)built** in the current line order — once #2 is fixed, make sure the "Arrange the final columns" step runs *after* all the derived columns (`price_gbp`, `price_inr`, `rating`, `in_stock`) have been created, which it already does structurally, it just needs `df` to exist from the start.

5. **`RATING_MAP`** is currently a same-to-same mapping (`"One": "One"`, etc.) — the `rating` column produced by `df["star_rating"].map(RATING_MAP)` will just duplicate `star_rating` as text, not convert it to a number. If you want `rating` to be numeric (as the SQLite schema's `INTEGER` column implies), change it to:
   ```python
   RATING_MAP = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}
   ```

6. **Scraping performance/etiquette**: the script fetches a detail page for *every single book* to get its category, with no delay between requests (`time` is imported but never used). Consider adding `time.sleep(...)` between requests to be polite to the server, especially if you scale beyond 5 pages.

## Key Outputs

- A cleaned, analysis-ready CSV (`books_to_scrape_dataset.csv`) with price in GBP and INR, numeric/boolean rating and stock flags, and category labels.
- A normalized SQLite database (`books.db`) with referential integrity between `books` and `categories`.
- Six SQL query results plus a verified pandas-equivalent of the JOIN query, confirming the data and relationships are consistent.

## License

For educational/portfolio use with the public books.toscrape.com sandbox site.
