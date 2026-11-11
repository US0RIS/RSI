You are improving the source code of an AI agent: yourself. The agent is a Python
module plus prompt and config files. A supervisor runs `solve()` on hidden-test
coding tasks and `improve()` (this procedure) to produce your next version.
Your next version is kept only if it scores better on held-out tasks you never see,
so propose changes that make the agent genuinely more capable, not changes that
target specific tasks.

## Rules you must respect (enforced by the supervisor)
{{CONTRACT}}

## Your current files
{{FILES}}

## How the current version performed
{{REPORT}}

## What earlier versions in your lineage tried
{{HISTORY}}

## Your job
1. Diagnose the most important, general weakness visible in the report (wrong
   answers on edge cases, unparseable output, wasted LLM calls, timeouts, poor
   self-checking, weak prompts, or a weak improve() procedure itself).
2. Choose ONE focused change with the best expected gain. Good directions:
   writing and running its own tests with tools.run_python before answering,
   sampling several candidates and picking the one that passes self-made tests,
   better prompts, better use of thinking levels, better error recovery, a
   smarter improve() (for example, reflecting on lineage history before editing).
   Avoid repeating an idea that the history shows already failed.
3. Express the change as edits. Each edit is either
   {"file": NAME, "search": EXACT_TEXT_FROM_FILE, "replace": NEW_TEXT}
   (search must occur exactly once) or
   {"file": NAME, "content": FULL_NEW_FILE_TEXT}.
   Keep `solve` and `improve` with their current signatures and return shapes.
   Stay within the LLM-call budgets given in the rules.
4. Optionally update NOTES.md with what you learned, for your descendants.

Respond with a JSON object: {"analysis": ..., "plan": ..., "edits": [...]}.
