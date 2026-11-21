from agents.gap_discovery_agent import discover_research_gaps
from tests.sample_data import COMPARISON, PAPER_PROFILES

result = discover_research_gaps(COMPARISON, PAPER_PROFILES)
print(result.model_dump_json(indent=2))