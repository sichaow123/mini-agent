from pathlib import Path

from typing_extensions import TypedDict

from agent.planner import Planner
from agent.skills import SkillManager
from runtime.path_mapper import PathMapper
from runtime.sandbox import DockerSandbox
from tracing.tracer import TaskMetrics


class AgentState(TypedDict):
    workdir: Path
    memory_dir: Path
    memory_index: Path
    sandbox: DockerSandbox
    planner: Planner
    skill_manager: SkillManager
    path_mapper: PathMapper
    metrics: TaskMetrics | None


def init_agent_state(
    workdir: str, sandbox: DockerSandbox, metrics: TaskMetrics | None = None
):
    workdir = Path(workdir)
    skills_dir = workdir / "skills"
    memory_dir = workdir / ".memory"
    memory_index = memory_dir / "MEMORY.md"
    agent_state = AgentState(
        workdir=workdir,
        memory_dir=memory_dir,
        memory_index=memory_index,
        sandbox=sandbox,
        planner=Planner(),
        skill_manager=SkillManager(skills_dir=skills_dir),
        path_mapper=PathMapper(workdir),
        metrics=metrics,
    )
    return agent_state
