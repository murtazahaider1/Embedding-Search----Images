"""
product_store.py

SQLite database for mutable product fields: title, price, url, image paths.
Completely separate from ChromaDB so you can update title/price without
touching embeddings. ChromaDB only stores the product_id as the link key.

Usage:
    from product_store import ProductStore
    store = ProductStore()
    store.upsert(product_id="42", title="Slim Fit Jeans", price="3849", ...)
    row = store.get("42")
"""

import sqlite3
from pathlib import Path

DB_PATH = "zarr_products.db"


class ProductStore:
    def __init__(self, db_path: str = DB_PATH):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_table()

    def _create_table(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS products (
                product_id   TEXT PRIMARY KEY,
                title        TEXT,
                price        TEXT,
                url          TEXT,
                image_url    TEXT,
                local_image  TEXT,
                updated_at   TEXT DEFAULT (datetime('now'))
            )
        """)
        self.conn.commit()

    def upsert(self, product_id: str, title: str, price: str,
               url: str, image_url: str, local_image: str):
        self.conn.execute("""
            INSERT INTO products (product_id, title, price, url, image_url, local_image)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(product_id) DO UPDATE SET
                title       = excluded.title,
                price       = excluded.price,
                url         = excluded.url,
                image_url   = excluded.image_url,
                local_image = excluded.local_image,
                updated_at  = datetime('now')
        """, (product_id, title, price, url, image_url, local_image))
        self.conn.commit()

    def get(self, product_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM products WHERE product_id = ?", (product_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_many(self, product_ids: list[str]) -> dict[str, dict]:
        if not product_ids:
            return {}
        placeholders = ",".join("?" * len(product_ids))
        rows = self.conn.execute(
            f"SELECT * FROM products WHERE product_id IN ({placeholders})",
            product_ids,
        ).fetchall()
        return {row["product_id"]: dict(row) for row in rows}

    def all_ids(self) -> set[str]:
        rows = self.conn.execute("SELECT product_id FROM products").fetchall()
        return {r["product_id"] for r in rows}

    def delete(self, product_id: str):
        self.conn.execute("DELETE FROM products WHERE product_id = ?", (product_id,))
        self.conn.commit()

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]

    def update_title(self, product_id: str, title: str):
        self.conn.execute(
            "UPDATE products SET title = ?, updated_at = datetime('now') WHERE product_id = ?",
            (title, product_id)
        )
        self.conn.commit()

    def update_price(self, product_id: str, price: str):
        self.conn.execute(
            "UPDATE products SET price = ?, updated_at = datetime('now') WHERE product_id = ?",
            (price, product_id)
        )
        self.conn.commit()
