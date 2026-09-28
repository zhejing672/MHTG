"""MAA2C for the asynchronous government-household economic environment."""
from agents.MGPPO.MGPPO import MGPPO
from agents.MAA2C.MAA2C_agent import MAA2CAgent


class MAA2C(MGPPO):
    """Centralized-training A2C with separate competitive-agent objectives."""

    algorithm_name = "MAA2C"

    def source_file_names(self):
        return super().source_file_names() + (
            "agents/MAPPO/MAPPO_agent.py",
            "agents/MAPPO/agent.py",
            "agents/MAPPO/policy.py",
            "agents/MAA2C/MAA2C.py",
            "agents/MAA2C/MAA2C_agent.py",
        )

    def create_agents(self):
        for index, name in enumerate(self.agent_list):
            self.agents.append(MAA2CAgent(self.envs, self.args, index, name))
