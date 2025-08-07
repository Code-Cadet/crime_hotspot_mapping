import numpy as np
import random
from typing import Tuple, List, Dict, Any
from dataclasses import dataclass
from config import AGENT_CONFIG, SIMULATION_CONFIG

@dataclass
class AgentState:
    position: Tuple[int, int]
    reputation: float
    risk_score: float
    nearby_agents: List[str]
    time_step: int

class BaseAgent:
    def __init__(self, agent_id: str, position: Tuple[int, int], agent_type: str):
        self.agent_id = agent_id
        self.position = position
        self.agent_type = agent_type
        self.reputation = 0.5
        self.q_table = {}
        self.learning_rate = SIMULATION_CONFIG["learning_rate"]
        self.discount_factor = SIMULATION_CONFIG["discount_factor"]
        self.exploration_rate = SIMULATION_CONFIG["exploration_rate"]
        self.movement_range = AGENT_CONFIG["movement_range"]
        self.vision_range = AGENT_CONFIG["vision_range"]
        
    def get_state_key(self, environment_state: Dict[str, Any]) -> str:
        risk_level = "high" if environment_state["risk_score"] > 0.7 else "medium" if environment_state["risk_score"] > 0.4 else "low"
        reputation_level = "high" if self.reputation > 0.7 else "medium" if self.reputation > 0.4 else "low"
        return f"{risk_level}_{reputation_level}_{len(environment_state['nearby_agents'])}"
    
    def choose_action(self, available_actions: List[str], environment_state: Dict[str, Any]) -> str:
        state_key = self.get_state_key(environment_state)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = {action: 0.0 for action in available_actions}
        
        if random.random() < self.exploration_rate:
            return random.choice(available_actions)
        
        q_values = self.q_table[state_key]
        max_q = max(q_values.values())
        best_actions = [action for action, q in q_values.items() if q == max_q]
        return random.choice(best_actions)
    
    def update_q_value(self, state: str, action: str, reward: float, next_state: str, next_actions: List[str]):
        if state not in self.q_table:
            self.q_table[state] = {}
        if action not in self.q_table[state]:
            self.q_table[state][action] = 0.0
            
        if next_state not in self.q_table:
            self.q_table[next_state] = {action: 0.0 for action in next_actions}
        
        max_next_q = max(self.q_table[next_state].values()) if self.q_table[next_state] else 0.0
        current_q = self.q_table[state][action]
        
        self.q_table[state][action] = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
    
    def move(self, grid_size: Tuple[int, int]) -> Tuple[int, int]:
        x, y = self.position
        max_x, max_y = grid_size
        
        dx = random.randint(-self.movement_range, self.movement_range)
        dy = random.randint(-self.movement_range, self.movement_range)
        
        new_x = max(0, min(max_x - 1, x + dx))
        new_y = max(0, min(max_y - 1, y + dy))
        
        self.position = (new_x, new_y)
        return self.position
    
    def update_reputation(self, change: float):
        self.reputation = max(0.0, min(1.0, self.reputation + change))
        self.reputation *= SIMULATION_CONFIG["reputation_decay"]

class OffenderAgent(BaseAgent):
    def __init__(self, agent_id: str, position: Tuple[int, int]):
        super().__init__(agent_id, position, "offender")
        self.assault_reputation = 0.3
        self.arrest_reputation = 0.1
        self.successful_crimes = 0
        self.failed_attempts = 0
        
    def decide_action(self, environment_state: Dict[str, Any]) -> Dict[str, Any]:
        available_actions = ["move", "assault", "hide"]
        
        if environment_state["risk_score"] > SIMULATION_CONFIG["risk_threshold"]:
            available_actions.remove("assault")
        
        action = self.choose_action(available_actions, environment_state)
        
        if action == "assault":
            success_prob = self.calculate_assault_success(environment_state)
            if random.random() < success_prob:
                self.successful_crimes += 1
                self.assault_reputation += 0.1
                reward = 10.0
                action_result = "successful_assault"
            else:
                self.failed_attempts += 1
                self.assault_reputation -= 0.05
                reward = -5.0
                action_result = "failed_assault"
        elif action == "hide":
            reward = 2.0
            action_result = "hid"
        else:  # move
            reward = 1.0
            action_result = "moved"
        
        return {
            "action": action,
            "action_result": action_result,
            "reward": reward,
            "new_position": self.position
        }
    
    def calculate_assault_success(self, environment_state: Dict[str, Any]) -> float:
        base_prob = AGENT_CONFIG["base_offense_probability"]
        risk_factor = 1.0 - environment_state["risk_score"]
        reputation_factor = self.assault_reputation
        guardian_factor = 1.0 - (len([a for a in environment_state["nearby_agents"] if a.startswith("guardian")]) * 0.1)
        
        return min(0.9, base_prob * risk_factor * reputation_factor * guardian_factor)

class TargetAgent(BaseAgent):
    def __init__(self, agent_id: str, position: Tuple[int, int]):
        super().__init__(agent_id, position, "target")
        self.vulnerability = random.uniform(0.2, 0.8)
        self.awareness = random.uniform(0.3, 0.9)
        
    def decide_action(self, environment_state: Dict[str, Any]) -> Dict[str, Any]:
        available_actions = ["move", "seek_help", "evade"]
        
        if environment_state["risk_score"] > 0.6:
            available_actions = ["evade", "seek_help"]
        
        action = self.choose_action(available_actions, environment_state)
        
        if action == "evade":
            reward = 5.0
            action_result = "evaded"
        elif action == "seek_help":
            reward = 3.0
            action_result = "sought_help"
        else:  # move
            reward = 1.0
            action_result = "moved"
        
        return {
            "action": action,
            "action_result": action_result,
            "reward": reward,
            "new_position": self.position
        }

class GuardianAgent(BaseAgent):
    def __init__(self, agent_id: str, position: Tuple[int, int]):
        super().__init__(agent_id, position, "guardian")
        self.arrest_reputation = 0.6
        self.patrol_efficiency = random.uniform(0.4, 0.9)
        
    def decide_action(self, environment_state: Dict[str, Any]) -> Dict[str, Any]:
        available_actions = ["patrol", "arrest", "investigate"]
        
        nearby_offenders = [a for a in environment_state["nearby_agents"] if a.startswith("offender")]
        
        if nearby_offenders:
            available_actions = ["arrest", "investigate"]
        
        action = self.choose_action(available_actions, environment_state)
        
        if action == "arrest":
            success_prob = self.calculate_arrest_success(environment_state)
            if random.random() < success_prob:
                self.arrest_reputation += 0.1
                reward = 15.0
                action_result = "successful_arrest"
            else:
                self.arrest_reputation -= 0.05
                reward = -3.0
                action_result = "failed_arrest"
        elif action == "investigate":
            reward = 4.0
            action_result = "investigated"
        else:  # patrol
            reward = 2.0
            action_result = "patrolled"
        
        return {
            "action": action,
            "action_result": action_result,
            "reward": reward,
            "new_position": self.position
        }
    
    def calculate_arrest_success(self, environment_state: Dict[str, Any]) -> float:
        base_prob = AGENT_CONFIG["base_arrest_probability"]
        reputation_factor = self.arrest_reputation
        patrol_factor = self.patrol_efficiency
        risk_factor = environment_state["risk_score"]
        
        return min(0.9, base_prob * reputation_factor * patrol_factor * risk_factor) 