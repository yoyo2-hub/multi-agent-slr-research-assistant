from database.db import get_connection, load_papers

papers = load_papers(get_connection())
decided = [p for p in papers if p.screening_reason]

print("Total:", len(papers), "| décidés:", len(decided), "| non décidés:", len(papers) - len(decided))
print("Inclus:", sum(p.included for p in decided))
print("\n--- EXCLUS ---")
for p in decided:
    if not p.included:
        print(f"- {p.title[:80]}\n    {p.screening_reason}")