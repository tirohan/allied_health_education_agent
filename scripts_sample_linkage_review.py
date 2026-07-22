"""
Export a random sample of cross-source links for human review (RQ2, Section VII
of the BigData submission). This script does NOT judge correctness itself --
it only samples real rows from the database into a reviewable CSV. A human
(ideally with allied-health domain knowledge) fills in the `correct` column,
then `scripts_score_linkage_review.py` computes precision/recall-style
statistics from the filled-in file.

Usage:
    python scripts_sample_linkage_review.py --out linkage_review_sample.csv --n-per-type 40
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
import random
from pathlib import Path

import asyncpg
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")


async def sample_paper_topics(conn: asyncpg.Connection, n: int) -> list[dict]:
    rows = await conn.fetch(
        """
        SELECT pt.paper_id, pt.topic_name, rp.title
        FROM paper_topics pt
        JOIN research_papers rp ON rp.paper_id = pt.paper_id
        WHERE rp.title IS NOT NULL
        ORDER BY random()
        LIMIT $1
        """,
        n,
    )
    return [
        {
            "link_type": "paper_topic",
            "id_a": r["paper_id"],
            "label_a": r["title"],
            "id_b": None,
            "label_b": r["topic_name"],
            "question": "Does the assigned topic genuinely relate to this paper's title?",
        }
        for r in rows
    ]


async def sample_program_institution(conn: asyncpg.Connection, n: int) -> list[dict]:
    rows = await conn.fetch(
        """
        SELECT p.program_id, p.program_title, p.unitid, i.institution_name
        FROM programs p
        JOIN institutions i ON i.unitid = p.unitid
        WHERE p.program_title IS NOT NULL AND i.institution_name IS NOT NULL
        ORDER BY random()
        LIMIT $1
        """,
        n,
    )
    return [
        {
            "link_type": "program_institution",
            "id_a": r["program_id"],
            "label_a": r["program_title"],
            "id_b": r["unitid"],
            "label_b": r["institution_name"],
            "question": "Is this program plausibly offered at this institution?",
        }
        for r in rows
    ]


async def sample_accreditation_institution(conn: asyncpg.Connection, n: int) -> list[dict]:
    rows = await conn.fetch(
        """
        SELECT a.accreditation_id, a.institution_name AS accred_name, a.unitid, i.institution_name
        FROM accreditation_records a
        JOIN institutions i ON i.unitid = a.unitid
        WHERE a.unitid IS NOT NULL
        ORDER BY random()
        LIMIT $1
        """,
        n,
    )
    return [
        {
            "link_type": "accreditation_institution",
            "id_a": r["accreditation_id"],
            "label_a": r["accred_name"],
            "id_b": r["unitid"],
            "label_b": r["institution_name"],
            "question": "Do these two institution names refer to the same real institution?",
        }
        for r in rows
    ]


async def sample_shortage_county(conn: asyncpg.Connection, n: int) -> list[dict]:
    rows = await conn.fetch(
        """
        SELECT w.record_id, w.county_name AS shortage_county_name, w.county_fips, c.county_name
        FROM workforce_shortage_records w
        JOIN county_profiles c ON c.county_fips = w.county_fips
        ORDER BY random()
        LIMIT $1
        """,
        n,
    )
    return [
        {
            "link_type": "shortage_county",
            "id_a": r["record_id"],
            "label_a": r["shortage_county_name"],
            "id_b": r["county_fips"],
            "label_b": r["county_name"],
            "question": "Do these two county names/FIPS refer to the same county?",
        }
        for r in rows
    ]


async def main(out_path: str, n_per_type: int, seed: int) -> None:
    random.seed(seed)
    conn = await asyncpg.connect(os.environ["DATABASE_URL"])
    try:
        rows: list[dict] = []
        rows += await sample_paper_topics(conn, n_per_type)
        rows += await sample_program_institution(conn, n_per_type)
        rows += await sample_accreditation_institution(conn, n_per_type)
        rows += await sample_shortage_county(conn, n_per_type)
    finally:
        await conn.close()

    random.shuffle(rows)
    for i, row in enumerate(rows, start=1):
        row["sample_id"] = i
        row["correct"] = ""  # human fills in: Y / N / UNSURE
        row["notes"] = ""

    fieldnames = ["sample_id", "link_type", "id_a", "label_a", "id_b", "label_b", "question", "correct", "notes"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows ({n_per_type} per link_type x 4 types) to {out_path}")
    print("Fill in the 'correct' column (Y/N/UNSURE) by hand, then run scripts_score_linkage_review.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="linkage_review_sample.csv")
    parser.add_argument("--n-per-type", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    asyncio.run(main(args.out, args.n_per_type, args.seed))
