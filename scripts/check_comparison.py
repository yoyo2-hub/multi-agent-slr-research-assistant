from agents.comparison_agent import compare_research_idea
from tests.sample_data import IDEA_PROFILE, PAPER_PROFILES

result = compare_research_idea(IDEA_PROFILE, PAPER_PROFILES)
print(result.model_dump_json(indent=2))