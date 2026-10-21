"""CrewAI: a task guardrail that asks Screwtape.ai to judge the output.

pass  -> guardrail returns (True, output) and the crew moves on
retry -> (False, reasons): CrewAI re-runs the task with the reasons, up to guardrail_max_retries
fail  -> raises, so a wrong answer never reaches the next task

pip install -r requirements.txt
export OPENAI_API_KEY=...  SCREWTAPE_API_KEY=...
python crew.py
"""
from typing import Any

from crewai import Agent, Crew, Task, TaskOutput

from screwtape import judge

EVIDENCE = ["The mission launched on 2024-03-01 from Cape Canaveral.", "The crew size was 4."]
CRITERIA = ["crew size", "Cape Canaveral"]
TASK_TEXT = "State the crew size and launch site of the mission, using only the evidence."


class JudgedFail(RuntimeError):
    pass


def screwtape_guardrail(output: TaskOutput) -> tuple[bool, Any]:
    v = judge(TASK_TEXT, output.raw, EVIDENCE, CRITERIA)
    print(f"screwtape: {v['verdict']} score={v['score']} id={v['request_id']} receipt={(v.get('receipt') or {}).get('receipt_id', '-')}")
    if v["verdict"] == "pass":
        return True, output.raw
    if v["verdict"] == "fail":
        raise JudgedFail("; ".join(v["reasons"]))
    return False, "Fix these issues: " + "; ".join(v["reasons"])


researcher = Agent(
    role="Researcher",
    goal="Answer strictly from the evidence provided.",
    backstory="You never add facts that are not in the evidence.",
)

task = Task(
    description=TASK_TEXT + "\nEvidence:\n- " + "\n- ".join(EVIDENCE),
    expected_output="One or two sentences.",
    agent=researcher,
    guardrail=screwtape_guardrail,
    guardrail_max_retries=2,
)

if __name__ == "__main__":
    result = Crew(agents=[researcher], tasks=[task]).kickoff()
    print("ANSWER:", result.raw)
