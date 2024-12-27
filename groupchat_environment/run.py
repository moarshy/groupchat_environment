#!/usr/bin/env python
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from groupchat_environment.schemas import GroupChatConfig, GlobalAction, InputSchema, LocalAction, GroupChatAction, GroupChatGlobalAction, GroupChatMessage, GroupChatObservation, GroupChatLocalObservation, GroupChatGlobalObservation, LocalEnvironmentStep, EnvironmentStep, EnvironmentHistory, ActionSpace, ObservationSpace, GroupChatActionSpace, GroupChatObservationSpace, GlobalObservation
from naptha_sdk.schemas import EnvironmentRunInput
from naptha_sdk.utils import get_logger
from pydantic import BaseModel, Field
import random
from typing import Any, Dict, List, Optional, Union

load_dotenv()

logger = get_logger(__name__)

class Mechanism(BaseModel, ABC):
    sequential: bool = Field(default=False, description="Whether the mechanism is sequential")
    @abstractmethod
    def step(self, action: Union[LocalAction, GlobalAction]) -> Union[LocalEnvironmentStep, EnvironmentStep]:
        """Execute a step in the mechanism."""
        pass

    @abstractmethod
    def get_global_state(self) -> Any:
        """Get the global state of the mechanism."""
        pass

class GroupChat(Mechanism):
    max_rounds: int = Field(..., description="Maximum number of chat rounds")
    current_round: int = Field(default=0, description="Current round number")
    messages: List[GroupChatMessage] = Field(default_factory=list)
    topics: Dict[str, str] = Field(default_factory=dict)  # cohort_id -> topic
    sequential: bool = Field(default=False, description="Whether the mechanism is sequential")

    def step(self, action: Union[GroupChatAction, GroupChatGlobalAction, Dict[str, Any]]) -> Union[LocalEnvironmentStep, EnvironmentStep]:
        logger.debug(f"Received action of type: {type(action).__name__}")
        logger.debug(f"Action content: {action}")

        if self.sequential:
            # Sequential mode: expect a LocalAction
            if isinstance(action, dict):
                try:
                    action = GroupChatAction.parse_obj(action)
                    logger.debug("Parsed action into GroupChatAction.")
                except Exception as e:
                    logger.error(f"Failed to parse action into GroupChatAction: {e}")
                    raise
            if not isinstance(action, GroupChatAction):
                logger.error(f"Expected GroupChatAction, got {type(action).__name__}")
                raise TypeError(f"Expected GroupChatAction, got {type(action).__name__}")

            # Check if it's the current agent's turn
            if action.agent_id != self.speaker_order[self.current_speaker_index]:
                raise ValueError(f"It's not agent {action.agent_id}'s turn to speak.")

            self.current_round += 1
            logger.debug(f"Processing round {self.current_round} with action: {action}")

            # Process the action
            self.messages.append(action.action)

            # Update topic if necessary
            if action.action.message_type == "propose_topic":
                self._update_topic(action.action.content)

            # Create observation for the agent
            observation = self._create_observation([action.action], action.agent_id)
            done = self.current_round >= self.max_rounds

            # Update the current speaker
            self._select_next_speaker()
            logger.debug(f"Next speaker selected: {self.speaker_order[self.current_speaker_index]}")

            local_step = LocalEnvironmentStep(
                observation=observation,
                reward=0,
                done=done,
                info={
                    "current_round": self.current_round,
                    "current_topic": self.current_topic,
                    "all_messages": [message.dict() for message in self.messages],
                    "speaker_order": self.speaker_order
                }
            )

            return local_step
        else:
            # Non-sequential mode: expect a GlobalAction
            if isinstance(action, dict):
                try:
                    action = GroupChatGlobalAction.parse_obj(action)
                    logger.debug("Parsed actions into GroupChatGlobalAction.")
                except Exception as e:
                    logger.error(f"Failed to parse actions into GroupChatGlobalAction: {e}")
                    raise

            # Ensure action is GroupChatGlobalAction
            if not isinstance(action, GroupChatGlobalAction):
                logger.error(f"Expected GroupChatGlobalAction, got {type(action).__name__}")
                raise TypeError(f"Expected GroupChatGlobalAction, got {type(action).__name__}")

            self.current_round += 1
            logger.debug(f"Processing round {self.current_round} with actions: {action}")

            new_messages = self._process_actions(action)
            self.messages.extend(new_messages)

            observations = self._create_observations(new_messages)
            done = self.current_round >= self.max_rounds

            # Update topics if a propose_topic message is found
            for message in new_messages:
                if message.message_type == "propose_topic":
                    self._update_topic(message.cohort_id, message.content)

            # Create global_observation
            global_observation = GroupChatGlobalObservation(
                observations=observations,
                all_messages=self.messages,
                current_topic="",
            )

            # Return an EnvironmentStep with your custom global_observation
            env_step = EnvironmentStep(
                global_observation=global_observation,
                done=done,
                info={
                    "current_round": self.current_round,
                    "all_messages": [message.dict() for message in self.messages],
                }
            )

            return env_step

    def _process_actions(self, global_action: GroupChatGlobalAction) -> List[GroupChatMessage]:
        new_messages = []
        for agent_id, action_dict in global_action.actions.items():
            try:
                action = GroupChatAction.parse_obj(action_dict)
                new_messages.append(action.action)
            except Exception as e:
                logger.error(f"Failed to parse action for agent {agent_id}: {e}")
                continue 
        return new_messages

    def _update_topic(self, cohort_id: str, new_topic: str):
        self.topics[cohort_id] = new_topic
        logger.info(f"Updated topic for cohort {cohort_id} to: {new_topic}")

    def _create_observations(self, new_messages: List[GroupChatMessage]) -> Dict[str, GroupChatLocalObservation]:
        observations = {}
        for message in new_messages:
            agent_id = message.agent_id
            observation = GroupChatObservation(
                messages=[msg for msg in new_messages if msg.cohort_id == message.cohort_id],
                current_topic=self.topics.get(message.cohort_id, "")
            )
            observations[agent_id] = GroupChatLocalObservation(
                agent_id=agent_id,
                observation=observation
            )
        return observations

    def get_global_state(self) -> Dict[str, Any]:
        return {
            "current_round": self.current_round,
            "messages": [message.dict() for message in self.messages],
            "topics": self.topics,
        }

    def reset(self) -> None:
        self.current_round = 0
        self.messages = []
        logger.info("GroupChat mechanism has been reset.")

class MultiAgentEnvironment(BaseModel):
    """
    Base class for multi-agent environments. With batched or sequential actions.
    """
    name: str = Field(..., description="Name of the environment")
    address: Optional[str] = Field(default=None, description="Address of the environment for orchestrator linking")
    current_step: int = Field(default=0, description="Current step/round of the simulation")
    max_steps: int = Field(default=10, description="Maximum number of steps/rounds for this environment")
    action_space: ActionSpace = Field(default_factory=GroupChatActionSpace, description="Action space of the environment")
    observation_space: ObservationSpace = Field(default_factory=GroupChatObservationSpace, description="Observation space of the environment")
    history: EnvironmentHistory = Field(default_factory=EnvironmentHistory, description="History of environment steps")
    mechanism: Mechanism = Field(default_factory=GroupChat, description="Mechanism of the environment that determines the rules of the game P(s, a, s')")

    def step(self, actions: GlobalAction) -> EnvironmentStep:
        """
        Run one timestep of the environment's dynamics using the batched agent actions.
        
        Args:
            actions (GlobalAction): A batched action containing actions for each agent.

        Returns:
            EnvironmentStep: The result of taking a step in the environment.
        """
        if self.mechanism.sequential:
            # if it is sequential, we need to run the mechanism for each agent
            local_steps: Dict[str, LocalEnvironmentStep] = {}  # Correct type annotation
            for agent_id, local_action in actions.locals().items():
                local_step = self.mechanism.step(local_action)
                assert isinstance(local_step, LocalEnvironmentStep)
                local_steps[agent_id] = local_step
            global_step = EnvironmentStep.from_local_steps(local_steps)
        else:
            global_step = self.mechanism.step(actions)
            assert isinstance(global_step, EnvironmentStep)
        self.current_step += 1
        self.update_history(actions, global_step)
        return global_step

    def reset(self) -> GlobalObservation:
        """
        Reset the environment and return the initial global observation.

        Returns:
            GlobalObservation: Initial global observation of the environment.
        """
        self.current_step = 0
        self.global_state = {}
        self.history = EnvironmentHistory()
        return GlobalObservation(observations={})

    def render(self):
        """
        Render the environment.
        """
        print(self.get_global_state())

    def close(self):
        """
        Close the environment, do any necessary cleanup.
        """
        pass  # No specific cleanup needed for the basic environment

    def get_global_state(self) -> Any:
        """
        Return a summary of the global state.

        Returns:
            Any: The global state.
        """
        return self.mechanism.get_global_state()

    def get_current_step(self) -> int:
        """
        Return the current step/round of the simulation.

        Returns:
            int: The current step.
        """
        return self.current_step

    def get_history(self) -> EnvironmentHistory:
        """
        Return the environment history.

        Returns:
            EnvironmentHistory: The environment history.
        """
        return self.history

    def update_history(self, action: GlobalAction, step: EnvironmentStep):
        """
        Update the environment history with the latest step.
        """
        self.history.add_step(action, step)

    def random_action_test(self, num_agents: int, num_steps: int):
        """
        Run a test with random actions for the specified number of agents and steps.
        """
        agent_ids = [f"Agent{i}" for i in range(num_agents)]

        print(f"\n=== Random Action Test for {self.name} ===\n")

        for step in range(num_steps):
            print(f"\nStep {step + 1}:")

            actions = {}
            for agent_id in agent_ids:
                action_type = random.choice(self.action_space.allowed_actions)
                actions[agent_id] = action_type.sample(agent_id)

            global_action = GlobalAction(actions=actions)
            step_result = self.step(global_action)
            self._print_step_results(step_result, actions, agent_ids)

        print("\nTest completed.")
        self.close()

    def _print_step_results(self, step_result: EnvironmentStep, actions: Dict[str, LocalAction], agent_ids: List[str]):
        """
        Print the results of a single step. This method can be overridden in subclasses for custom output.
        """
        for agent_id in agent_ids:
            local_step = step_result.get_local_step(agent_id)
            print(f"{agent_id} action: {actions[agent_id].action}")
            print(f"{agent_id} observation: {step_result.global_observation.observations[agent_id].observation}")

        print("\nGlobal state:")
        print(self.get_global_state())
        print("\n" + "="*50)

def create_environment(environment_run):

    group_chat = GroupChat(
        max_rounds=environment_run.deployment.environment_config.max_rounds,
        current_topic=environment_run.deployment.environment_config.initial_topic,
        speaker_order=["0"]
    )

    environment = MultiAgentEnvironment(
        name=environment_run.deployment.environment_config.config_name,
        address="group_chat_address",
        max_steps=environment_run.deployment.environment_config.max_rounds,
        action_space=GroupChatActionSpace(),
        observation_space=GroupChatObservationSpace(),
        mechanism=group_chat
    )

    return environment

def run(environment_run: EnvironmentRunInput):

    environment = create_environment(environment_run)

    method = getattr(environment, environment_run.inputs.function_name, None)

    if environment_run.inputs.function_input_data:
        return method(**environment_run.inputs.function_input_data)
    else:
        return method()

if __name__ == "__main__":
    from naptha_sdk.client.naptha import Naptha
    from naptha_sdk.configs import load_environment_deployments

    naptha = Naptha()

    environment_deployments = load_environment_deployments("groupchat_environment/configs/environment_deployments.json", config_schema=GroupChatConfig())

    # input_params = InputSchema(
    #     function_name="get_global_state",
    #     function_input_data=None,
    # )

    # environment_run = EnvironmentRunInput(
    #     inputs=input_params,
    #     environment_deployment=environment_deployments[0],
    #     consumer_id=naptha.user.id,
    # )
    # response = run(environment_run)
    # print(response)

    input_params = InputSchema(
        function_name="get_global_state",
        function_input_data=None,
    )

    environment_run = EnvironmentRunInput(
        inputs=input_params,
        deployment=environment_deployments[0],
        consumer_id=naptha.user.id,
    )
    response = run(environment_run)
    print(response)
