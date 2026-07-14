from __future__ import annotations

from typing import Any

from backend.app.db.postgres import Postgres


async def find_education_gaps(
    db: Postgres,
    *,
    topic_keywords: str = "opioid OR substance OR behavioral OR interprofessional",
    county: str | None = None,
    state: str = "GA",
    limit: int = 20,
) -> dict[str, Any]:
    """Compare county shortage severity with nearby teaching resources and programs."""

    if county:
        counties = await db.fetch(
            """
            SELECT
                c.county_fips,
                c.county_name,
                c.state,
                c.is_priority_county,
                c.poverty_percentage,
                c.uninsured_percentage,
                COALESCE(s.number_hpsa_designations, 0) AS hpsa_count,
                COALESCE(s.max_hpsa_score, 0) AS max_hpsa_score,
                COALESCE(s.any_shortage_flag, FALSE) AS any_shortage_flag
            FROM county_profiles c
            LEFT JOIN georgia_county_shortage_summary s
                ON c.county_fips = s.county_fips
            WHERE c.state = $1
              AND (
                  c.county_name ILIKE $2
                  OR c.county_name ILIKE $3
              )
            ORDER BY
                COALESCE(s.max_hpsa_score, 0) DESC,
                COALESCE(s.number_hpsa_designations, 0) DESC,
                c.is_priority_county DESC
            LIMIT $4
            """,
            state,
            f"%{county}%",
            f"%{county.replace(' County', '').strip()}%",
            limit,
        )
    else:
        counties = await db.fetch(
            """
            SELECT
                c.county_fips,
                c.county_name,
                c.state,
                c.is_priority_county,
                c.poverty_percentage,
                c.uninsured_percentage,
                COALESCE(s.number_hpsa_designations, 0) AS hpsa_count,
                COALESCE(s.max_hpsa_score, 0) AS max_hpsa_score,
                COALESCE(s.any_shortage_flag, FALSE) AS any_shortage_flag
            FROM county_profiles c
            LEFT JOIN georgia_county_shortage_summary s
                ON c.county_fips = s.county_fips
            WHERE c.state = $1
            ORDER BY
                COALESCE(s.max_hpsa_score, 0) DESC,
                COALESCE(s.number_hpsa_designations, 0) DESC,
                c.is_priority_county DESC
            LIMIT $2
            """,
            state,
            limit,
        )

    resource_count_row = await db.fetchrow(
        """
        SELECT COUNT(*)::int AS resource_count
        FROM resources
        WHERE to_tsvector(
                  'english',
                  coalesce(title,'') || ' ' || coalesce(description,'')
              ) @@ websearch_to_tsquery('english', $1)
        """,
        topic_keywords,
    )
    simulation_count_row = await db.fetchrow(
        """
        SELECT COUNT(*)::int AS simulation_count
        FROM simulation_cases
        WHERE to_tsvector(
                  'english',
                  coalesce(title,'') || ' ' || coalesce(abstract_or_summary,'')
              ) @@ websearch_to_tsquery('english', $1)
        """,
        topic_keywords,
    )
    program_count_row = await db.fetchrow(
        """
        SELECT COUNT(*)::int AS program_count
        FROM programs p
        JOIN institutions i ON p.unitid = i.unitid
        WHERE i.state = $1
          AND p.allied_health_category IN ('Core Allied Health', 'Allied Health Adjacent')
        """,
        state,
    )

    resource_count = int((resource_count_row or {}).get("resource_count") or 0)
    simulation_count = int((simulation_count_row or {}).get("simulation_count") or 0)
    program_count = int((program_count_row or {}).get("program_count") or 0)

    gaps: list[dict[str, Any]] = []
    for county in counties:
        hpsa_score = float(county.get("max_hpsa_score") or 0)
        hpsa_count = int(county.get("hpsa_count") or 0)
        shortage_high = hpsa_score >= 12 or hpsa_count >= 2 or bool(county.get("any_shortage_flag"))
        # Teaching materials are statewide in this MVP, so we flag relative scarcity by topic.
        teaching_low = resource_count < 25 or simulation_count < 10
        gap_level = "High" if shortage_high and teaching_low else "Moderate" if shortage_high else "Lower"
        recommendation = (
            "Prioritize new local case design and OER curation."
            if gap_level == "High"
            else "Review existing materials and adapt to local indicators."
            if gap_level == "Moderate"
            else "Monitor and reuse existing materials with local context."
        )
        gaps.append(
            {
                "county_fips": county["county_fips"],
                "county_name": county["county_name"],
                "state": county["state"],
                "is_priority_county": county.get("is_priority_county"),
                "poverty_percentage": float(county["poverty_percentage"])
                if county.get("poverty_percentage") is not None
                else None,
                "uninsured_percentage": float(county["uninsured_percentage"])
                if county.get("uninsured_percentage") is not None
                else None,
                "hpsa_count": hpsa_count,
                "max_hpsa_score": hpsa_score,
                "gap_level": gap_level,
                "what_is_missing": (
                    "High shortage signal with relatively limited topic matched teaching materials."
                    if gap_level == "High"
                    else "Shortage is present and materials should be adapted carefully."
                    if gap_level == "Moderate"
                    else "No major shortage and materials gap detected for this topic slice."
                ),
                "recommendation": recommendation,
            }
        )

    high_gaps = [row for row in gaps if row["gap_level"] == "High"]
    recommendations = [
        "Focus new case design on High gap counties first.",
        "Pair each shortage county with at least one OER and one simulation case.",
        "Use Curriculum Builder to turn gap findings into a printable module outline.",
    ]
    if county and not gaps:
        recommendations = [
            "No matching county was found. Try a shorter county name such as Coffee or Bibb.",
        ]
    elif high_gaps:
        recommendations.insert(
            0,
            f"Start with {high_gaps[0]['county_name']}: shortage is high and topic matched materials are relatively limited.",
        )

    return {
        "topic_keywords": topic_keywords,
        "state": state,
        "county_filter": county,
        "statewide_teaching_signal": {
            "matched_resources": resource_count,
            "matched_simulation_cases": simulation_count,
            "allied_health_programs_in_state": program_count,
        },
        "metrics": {
            "shortage_severity": (
                "High"
                if gaps and gaps[0]["gap_level"] == "High"
                else "Moderate"
                if gaps
                else "Unknown"
            ),
            "nearby_programs": program_count,
            "available_resources": resource_count,
            "simulation_cases": simulation_count,
        },
        "gaps": [
            {
                "County": row["county_name"],
                "Gap level": row["gap_level"],
                "What is missing": row["what_is_missing"],
                "HPSA score": row["max_hpsa_score"],
                "Next step": row["recommendation"],
            }
            for row in gaps
        ],
        "county_gaps": gaps,
        "counties": gaps,
        "recommendations": recommendations,
        "summary": (
            f"Across {state}, the topic search matched {resource_count} resources and "
            f"{simulation_count} simulation cases. {program_count} allied health related "
            "programs were found in state institutions. Counties at the top of the list "
            "are the best places to focus local curriculum adaptation."
        ),
        "educator_summary": (
            f"Across {state}, the topic search matched {resource_count} resources and "
            f"{simulation_count} simulation cases. {program_count} allied health related "
            "programs were found in state institutions. Counties at the top of the list "
            "are the best places to focus local curriculum adaptation."
        ),
    }
