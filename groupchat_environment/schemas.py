from abc import ABC, abstractmethod
from naptha_sdk.schemas import EnvironmentConfig
from pydantic import BaseModel, computed_field, Field
import random
import string
from typing import Any, Dict, List, Literal, Optional, Union, Tuple, Type


class LocalAction(BaseModel, ABC):
    """Represents an action for a single agent."""
    agent_id: str
    action: Any

    @classmethod
    @abstractmethod
    def sample(cls, agent_id: str) -> 'LocalAction':
        """Sample a random action for the given agent_id."""
        pass

class GlobalAction(BaseModel):
    """Represents actions for all agents."""
    actions: Dict[str, LocalAction]

    def locals(self) -> Dict[str, LocalAction]:
        """Get the local actions for all agents."""
        return self.actions

    @classmethod
    def from_local_actions(cls, local_actions: Dict[str, LocalAction]) -> "GlobalAction":
        """Create a global action from local actions."""
        return cls(actions=local_actions)

class LocalObservation(BaseModel, ABC):
    """Represents an observation for a single agent."""
    agent_id: str
    observation: BaseModel

class GlobalObservation(BaseModel):
    """Represents observations for all agents."""
    observations: Dict[str, LocalObservation]
    

    def locals(self) -> Dict[str, LocalObservation]:
        """Get the local observations for all agents."""
        return self.observations
    
    @property
    @computed_field
    def global_obs(self) -> Optional[Any]:
        """Get the global observation for all agents."""
        return None

    @classmethod
    def from_local_observations(cls, local_observations: Dict[str, LocalObservation]) -> "GlobalObservation":
        """Create a global observation from local observations."""
        return cls(observations=local_observations)

    def to_local(self, agent_id: str) -> LocalObservation:
        """Convert global observation to local observation for a specific agent."""
        return self.observations[agent_id]

class LocalEnvironmentStep(BaseModel):
    """Represents the output of a single environment step for a single agent."""
    observation: LocalObservation
    reward: Optional[float] = Field(default=None, description="Reward for the agent at this step")
    done: bool
    info: Dict[str, Any]

class EnvironmentStep(BaseModel):
    """Represents the output of a single environment step."""
    global_observation: GlobalObservation
    done: bool
    info: Dict[str, Any]

    @classmethod
    def from_local_steps(cls, local_steps: Dict[str, LocalEnvironmentStep]) -> "EnvironmentStep":
        """Create a global environment step from local steps."""
        observations = {agent_id: step.observation for agent_id, step in local_steps.items()}
        done = all(step.done for step in local_steps.values())
        info = {}
        return cls(
            global_observation=GlobalObservation.from_local_observations(observations),
            done=done,
            info=info
        )

    def get_local_step(self, agent_id: str) -> LocalEnvironmentStep:
        """Get the local step for a single agent."""
        return LocalEnvironmentStep(
            observation=self.global_observation.to_local(agent_id),
            done=self.done,
            info=self.info
        )

class EnvironmentHistory(BaseModel):
    """Represents the history of environment steps."""
    steps: List[Tuple[GlobalAction, EnvironmentStep]] = Field(default_factory=list)

    def add_step(self, action: GlobalAction, step: EnvironmentStep):
        """Add a step to the history."""
        self.steps.append((action, step))

class ActionSpace(BaseModel):
    allowed_actions: List[Type[LocalAction]] = Field(default_factory=list, description="List of allowed action types")

    def sample(self, agent_id: str) -> LocalAction:
        """Sample a random action from the allowed actions."""
        if not self.allowed_actions:
            raise ValueError("No allowed actions defined")
        action_type = random.choice(self.allowed_actions)
        return action_type.sample(agent_id)
    
    def get_action_schema(self) -> Dict[str, Any]:
        """Get the schema for the allowed actions."""
        if not self.allowed_actions:
            raise ValueError("No allowed actions defined")
        # Assuming all allowed actions have the same schema
        return self.allowed_actions[0].model_json_schema() 

class ObservationSpace(BaseModel):
    allowed_observations: List[Type[LocalObservation]] = Field(default_factory=list)

class StrObservation(LocalObservation):
    observation: str

    @classmethod
    def sample(cls, agent_id: str, min_length: int = 1, max_length: int = 100) -> 'StrObservation':
        content = ''.join(random.choices(string.ascii_letters + string.digits + string.punctuation + ' ', k=random.randint(min_length, max_length)))
        return cls(agent_id=agent_id, observation=content)


class GroupChatMessage(BaseModel):
    content: str
    message_type: Literal["propose_topic", "group_message"]
    agent_id: str
    cohort_id: str
    sub_round: int

class GroupChatAction(LocalAction):
    action: GroupChatMessage

    @classmethod
    def sample(cls, agent_id: str) -> 'GroupChatAction':
        return cls(
            agent_id=agent_id, 
            action=GroupChatMessage(
                content="Sample message", 
                message_type="group_message",
                agent_id=agent_id,
                cohort_id="sample_cohort",
                sub_round=1
            )
        )

    @classmethod
    def action_schema(cls) -> Dict[str, Any]:
        return cls.model_json_schema()

class GroupChatGlobalAction(GlobalAction):
    actions: Dict[str, Dict[str, Any]]

class GroupChatObservation(BaseModel):
    messages: List[GroupChatMessage]
    current_topic: str

class GroupChatLocalObservation(LocalObservation):
    observation: GroupChatObservation

class GroupChatGlobalObservation(GlobalObservation):
    observations: Dict[str, GroupChatLocalObservation]
    all_messages: List[GroupChatMessage]
    current_topic: str

class GroupChatActionSpace(ActionSpace):
    allowed_actions: List[Type[LocalAction]] = [GroupChatAction]

class GroupChatObservationSpace(ObservationSpace):
    allowed_observations: List[Type[LocalObservation]] = [GroupChatLocalObservation]

class GroupChatConfig(EnvironmentConfig):
    max_rounds: int = 5
    initial_topic: str = "Initial Market Discussion"
    sub_rounds: int = 3
    group_size: int = 5

class InputSchema(BaseModel):
    function_name: str
    function_input_data: Optional[Dict[str, Any]] = None
