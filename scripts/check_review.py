from agents.critical_reviewer import review_research_idea
from tests.sample_data import COMPARISON, GAPS, IDEA_PROFILE

result = review_research_idea(IDEA_PROFILE, COMPARISON, GAPS, corpus_size=2)
print(result.model_dump_json(indent=2))