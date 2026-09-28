"""IPPO aligned with the common rollout, evaluation and logging protocol."""
from agents.MGPPO.MGPPO import MGPPO
from agents.IPPO.IPPO_agent import IPPOAgent


class IPPO(MGPPO):
    """Independent PPO with one local actor and critic per strategic agent."""

    algorithm_name = "IPPO"

    def source_file_names(self):
        return super().source_file_names() + (
            "agents/IPPO/IPPO.py",
            "agents/IPPO/IPPO_agent.py",
            "agents/IPPO/policy.py",
        )

    def create_agents(self):
        for index, name in enumerate(self.agent_list):
            self.agents.append(IPPOAgent(self.envs, self.args, index, name))
