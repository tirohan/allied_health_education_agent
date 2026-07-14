from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class EducatorRole(StrEnum):
    PT_FACULTY = "pt_faculty"
    OT_FACULTY = "ot_faculty"
    NURSING_FACULTY = "nursing_faculty"
    SIMULATION_DIRECTOR = "simulation_director"
    PROGRAM_CHAIR = "program_chair"
    RESEARCHER = "researcher"


ROLE_LABELS: dict[EducatorRole, str] = {
    EducatorRole.PT_FACULTY: "PT faculty member",
    EducatorRole.OT_FACULTY: "OT faculty member",
    EducatorRole.NURSING_FACULTY: "Nursing faculty member",
    EducatorRole.SIMULATION_DIRECTOR: "Simulation director",
    EducatorRole.PROGRAM_CHAIR: "Program chair",
    EducatorRole.RESEARCHER: "Researcher",
}

ROLE_FOCUS: dict[EducatorRole, list[str]] = {
    EducatorRole.PT_FACULTY: ["physical therapy", "rehabilitation", "mobility", "IPE"],
    EducatorRole.OT_FACULTY: ["occupational therapy", "daily living", "functional", "IPE"],
    EducatorRole.NURSING_FACULTY: ["nursing", "patient care", "clinical", "IPE"],
    EducatorRole.SIMULATION_DIRECTOR: ["simulation", "debrief", "case", "scenario"],
    EducatorRole.PROGRAM_CHAIR: ["program", "accreditation", "workforce", "enrollment"],
    EducatorRole.RESEARCHER: ["evidence", "literature", "gap", "policy"],
}

ENTITY_PLAIN: dict[str, str] = {
    "Topic": "Teaching topic",
    "Paper": "Research article",
    "Resource": "Teaching resource",
    "Program": "Allied health program",
    "County": "Georgia county context",
    "Competency": "Competency",
    "SimulationCase": "Simulation case",
    "Institution": "Institution",
    "ShortageArea": "Workforce shortage area",
    "Discipline": "Discipline",
    "Author": "Author",
}

EVIDENCE_PLAIN: dict[str, str] = {
    "CONFIRMED": "Strong evidence from database records",
    "INFERRED": "Possible link that may need expert review",
    "UNVERIFIED": "Not yet verified against a primary record",
    "REFUTED": "Contradicted by the database and hidden from planning views",
}

SOURCE_PLAIN: dict[str, str] = {
    "research_papers": "peer-reviewed literature",
    "resources": "educational resource library",
    "programs": "program and institution records",
    "county_profiles": "Georgia county health profiles",
    "simulation_cases": "simulation case library",
    "topic_modules": "controlled teaching topics",
    "competencies": "competency framework",
    "workforce_shortage_records": "HRSA shortage designations",
}


class GuidedStarter(BaseModel):
    id: str
    title: str
    description: str
    query: str
    collections: list[str]
    filters: dict[str, Any] = Field(default_factory=dict)
    next_action: str


GUIDED_STARTERS: list[GuidedStarter] = [
    GuidedStarter(
        id="opioid_ipe_rural_ga",
        title="Build an opioid IPE module for rural Georgia",
        description="Connect topics, teaching resources, simulation cases, and rural county context.",
        query=(
            "What interprofessional education resources address opioid education "
            "in rural Georgia counties?"
        ),
        collections=["papers", "resources", "programs", "communities", "simulation_cases"],
        filters={"state": "GA"},
        next_action="Open Curriculum Builder",
    ),
    GuidedStarter(
        id="community_sim_cases",
        title="Find simulation cases grounded in local community data",
        description="Link simulation cases to Georgia county indicators and shortage context.",
        query=(
            "Which simulation cases support community based allied health training "
            "for Georgia counties with local health needs?"
        ),
        collections=["simulation_cases", "communities", "resources"],
        filters={"state": "GA"},
        next_action="Compare counties",
    ),
    GuidedStarter(
        id="programs_near_shortages",
        title="See which allied health programs exist near shortage counties",
        description="Explore program availability around high need Georgia counties.",
        query=(
            "Which allied health programs are available near Georgia health "
            "professional shortage counties?"
        ),
        collections=["programs", "communities"],
        filters={"state": "GA"},
        next_action="Open Gap Finder",
    ),
    GuidedStarter(
        id="competency_oer_map",
        title="Map competencies to available open educational resources",
        description="Find OER and agency resources that support competency based teaching.",
        query=(
            "Which open educational resources address interprofessional collaboration "
            "and substance use competencies for allied health learners?"
        ),
        collections=["resources", "papers"],
        filters={},
        next_action="Save to teaching list",
    ),
    GuidedStarter(
        id="gap_shortage_resources",
        title="Identify gaps: shortage high, teaching resources low",
        description="Highlight places where workforce need is high but teaching materials are thin.",
        query=(
            "Where do Georgia shortage counties have high need but limited opioid "
            "or behavioral health teaching resources?"
        ),
        collections=["communities", "resources", "programs", "simulation_cases"],
        filters={"state": "GA"},
        next_action="Open Gap Finder",
    ),
]


class EducatorNodeCard(BaseModel):
    node_id: str
    label: str
    what_it_is: str
    why_it_appeared: str
    who_it_is_for: str
    evidence_strength: str
    evidence_plain: str
    next_actions: list[str]
    source_plain: str
    source_url: str | None = None
    confidence: float
    entity_type: str
    verification_status: str
    advanced: dict[str, Any] = Field(default_factory=dict)


def role_options() -> list[dict[str, str]]:
    return [{"id": role.value, "label": ROLE_LABELS[role]} for role in EducatorRole]


def audience_for_role(role: EducatorRole | str | None) -> str:
    role_enum = EducatorRole(role) if role else EducatorRole.RESEARCHER
    focuses = ROLE_FOCUS[role_enum]
    if role_enum == EducatorRole.SIMULATION_DIRECTOR:
        return "Simulation educators and IPE facilitators"
    if role_enum == EducatorRole.PROGRAM_CHAIR:
        return "Program leaders planning curriculum and workforce alignment"
    if role_enum in {
        EducatorRole.PT_FACULTY,
        EducatorRole.OT_FACULTY,
        EducatorRole.NURSING_FACULTY,
    }:
        return f"{ROLE_LABELS[role_enum]} and interprofessional teaching teams ({', '.join(focuses[:3])})"
    return "Allied health educators, researchers, and curriculum planners"


def next_actions_for_entity(entity_type: str) -> list[str]:
    mapping = {
        "Resource": ["Use in syllabus", "Open resource", "Save to teaching list"],
        "Paper": ["Use in syllabus", "Open resource", "Save to teaching list"],
        "SimulationCase": ["Use in syllabus", "Open resource", "Save to teaching list"],
        "County": ["Compare counties", "Open Gap Finder", "Save to teaching list"],
        "Program": ["Compare counties", "Save to teaching list"],
        "Topic": ["Open Curriculum Builder", "Save to teaching list"],
        "Competency": ["Map to resources", "Save to teaching list"],
    }
    return mapping.get(entity_type, ["Save to teaching list"])


def plain_evidence_sentence(node: dict[str, Any]) -> str:
    source_table = str(node.get("source_table") or "")
    source_name = SOURCE_PLAIN.get(source_table, "the allied health education database")
    source_url = str(node.get("source_url") or "")
    label = str(node.get("label") or "")
    if "mededportal" in source_url.lower() or "mededportal" in label.lower():
        source_name = "MedEdPORTAL"
    status = str(node.get("verification_status") or "UNVERIFIED")
    status_text = EVIDENCE_PLAIN.get(status, "Evidence status is available for review")
    entity = ENTITY_PLAIN.get(str(node.get("entity_type")), "Item")
    payload = node.get("payload") if isinstance(node.get("payload"), dict) else {}
    tags = []
    for key in ("topics", "disciplines", "competencies", "keywords"):
        value = payload.get(key) if payload else None
        if isinstance(value, list) and value:
            tags.extend(str(item) for item in value[:2])
        elif isinstance(value, str) and value:
            tags.append(value)
    if not tags and any(token in label.lower() for token in ("opioid", "substance", "ipe", "interprofessional")):
        if "opioid" in label.lower() or "substance" in label.lower():
            tags.append("opioid education")
        if "ipe" in label.lower() or "interprofessional" in label.lower():
            tags.append("interprofessional collaboration")
    tag_text = ""
    if tags:
        tag_text = " · tagged for " + ", ".join(tags[:3])
    link_hint = ""
    entity_type = str(node.get("entity_type") or "")
    if entity_type in {"Resource", "Paper", "SimulationCase"}:
        link_hint = " · linked to interprofessional teaching planning"
    elif entity_type == "County":
        link_hint = " · linked to local community and shortage context"
    elif entity_type == "Program":
        link_hint = " · linked to allied health program capacity"
    verified = " · verified in the database" if status == "CONFIRMED" else f" · {status_text.lower()}"
    return (
        f"Found in {source_name}{tag_text}{link_hint}{verified}. "
        f"Shown as a {entity.lower()}."
    )


def why_appeared(node: dict[str, Any], query: str, role: str | None) -> str:
    entity_type = str(node.get("entity_type") or "Item")
    role_text = ROLE_LABELS.get(EducatorRole(role), "educators") if role else "educators"
    if entity_type == "County":
        return (
            f"This county helps ground your question about local community need for {role_text}."
        )
    if entity_type == "Resource":
        return "This teaching resource matched key ideas in your planning question."
    if entity_type == "Paper":
        return "This article provides evidence that supports the teaching topic you asked about."
    if entity_type == "SimulationCase":
        return "This simulation case can turn the topic into practice based learning."
    if entity_type == "Program":
        return "This program shows where related allied health education capacity already exists."
    if entity_type == "Topic":
        return f"This is the central teaching topic inferred from your question: {query}"
    return "This item was retrieved because it is related to your teaching or planning question."


def enrich_node(
    node: dict[str, Any],
    query: str,
    role: str | None = None,
) -> EducatorNodeCard:
    entity_type = str(node.get("entity_type") or "Item")
    status = str(node.get("verification_status") or "UNVERIFIED")
    source_table = str(node.get("source_table") or "")
    return EducatorNodeCard(
        node_id=str(node.get("id")),
        label=str(node.get("label") or node.get("id")),
        what_it_is=ENTITY_PLAIN.get(entity_type, entity_type),
        why_it_appeared=why_appeared(node, query, role),
        who_it_is_for=audience_for_role(role),
        evidence_strength=EVIDENCE_PLAIN.get(status, status),
        evidence_plain=plain_evidence_sentence(node),
        next_actions=next_actions_for_entity(entity_type),
        source_plain=SOURCE_PLAIN.get(source_table, "allied health education database"),
        source_url=str(node["source_url"]) if node.get("source_url") else None,
        confidence=float(node.get("confidence") or 0.0),
        entity_type=entity_type,
        verification_status=status,
        advanced={
            "source_table": source_table,
            "source_id": node.get("source_id"),
            "cluster": node.get("cluster"),
            "tooltip": node.get("tooltip"),
        },
    )


def enrich_graph(
    graph: dict[str, Any],
    query: str,
    role: str | None = None,
) -> list[EducatorNodeCard]:
    return [enrich_node(node, query, role) for node in graph.get("nodes", [])]


def build_curriculum_outline(
    graph: dict[str, Any],
    query: str,
    role: str | None = None,
) -> dict[str, Any]:
    nodes = graph.get("nodes", [])
    by_type: dict[str, list[dict[str, Any]]] = {}
    for node in nodes:
        by_type.setdefault(str(node.get("entity_type")), []).append(node)

    topic = (by_type.get("Topic") or [{}])[0]
    resources = by_type.get("Resource", [])[:5]
    papers = by_type.get("Paper", [])[:3]
    cases = by_type.get("SimulationCase", [])[:2]
    programs = by_type.get("Program", [])[:3]
    counties = by_type.get("County", [])[:3]
    competencies = by_type.get("Competency", [])[:4]

    learning_objectives = [
        f"Explain key concepts related to {topic.get('label') or 'the selected teaching topic'}.",
        "Apply interprofessional collaboration in a community informed learning activity.",
        "Connect local workforce or community context to educational planning decisions.",
    ]
    if cases:
        learning_objectives.append(
            f"Participate in a simulation based on {cases[0].get('label')}."
        )

    suggested_sequence = [
        "Introduce the topic and competency focus",
        "Assign one research reading and one teaching resource",
        "Run a simulation or case discussion",
        "Connect learner discussion to local county or shortage context",
        "Close with reflection and curriculum adaptation notes",
    ]
    outline = {
        "title": f"Curriculum outline for {topic.get('label') or 'allied health teaching'}",
        "role": ROLE_LABELS.get(EducatorRole(role), "Educator") if role else "Educator",
        "planning_question": query,
        "recommended_topic": topic.get("label"),
        "learning_objectives": learning_objectives,
        "competencies": [item.get("label") for item in competencies],
        "readings": [item.get("label") for item in papers],
        "teaching_resources": [item.get("label") for item in resources],
        "simulation_case": cases[0].get("label") if cases else None,
        "simulation_cases": [item.get("label") for item in cases],
        "programs": [item.get("label") for item in programs],
        "community_context": [item.get("label") for item in counties],
        "suggested_sequence": suggested_sequence,
        "next_steps": [
            "Open Teaching Pack to export readings, a case, and county context.",
            "Use Gap Finder to check where local need outpaces teaching materials.",
            "Mark useful items in Evidence and Review so colleagues can trust the pack.",
        ],
    }
    lines = [
        f"# {outline['title']}",
        "",
        f"Role: {outline['role']}",
        f"Planning question: {query}",
        "",
        "## Learning objectives",
    ]
    for item in learning_objectives:
        lines.append(f"- {item}")
    lines.extend(["", "## Recommended topic", outline["recommended_topic"] or "—", "", "## Competencies"])
    for item in outline["competencies"] or ["—"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Readings"])
    for item in outline["readings"] or ["—"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Teaching resources"])
    for item in outline["teaching_resources"] or ["—"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Simulation case",
            f"- {outline['simulation_case'] or 'None selected yet'}",
            "",
            "## Community context",
        ]
    )
    for item in outline["community_context"] or ["—"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Programs"])
    for item in outline["programs"] or ["—"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Suggested sequence"])
    for item in suggested_sequence:
        lines.append(f"- {item}")
    outline["printable_markdown"] = "\n".join(lines) + "\n"
    return outline
