import json
import random
import numpy as np
from typing import Dict, List, Tuple, Any
from datetime import datetime, timedelta
from pathlib import Path

from .agents import OffenderAgent, TargetAgent, GuardianAgent
from config import SIMULATION_CONFIG, AGENT_CONFIG, PROCESSED_DATA_DIR, FILE_PATTERNS
from risk.risk_model import RiskTerrainModel

class CrimeHotspotSimulator:
    def __init__(self, grid_size: Tuple[int, int] = None):
        self.grid_size = grid_size or SIMULATION_CONFIG["grid_size"]
        self.agents = {}
        self.risk_model = RiskTerrainModel(self.grid_size)
        self.simulation_log = []
        self.episode_data = []
        
    def initialize_agents(self):
        """Initialize all agents with random positions"""
        agent_id_counter = 0
        
        # Initialize offenders
        for i in range(AGENT_CONFIG["offender_count"]):
            position = self._get_random_position()
            agent_id = f"offender_{agent_id_counter}"
            self.agents[agent_id] = OffenderAgent(agent_id, position)
            agent_id_counter += 1
        
        # Initialize targets
        for i in range(AGENT_CONFIG["target_count"]):
            position = self._get_random_position()
            agent_id = f"target_{agent_id_counter}"
            self.agents[agent_id] = TargetAgent(agent_id, position)
            agent_id_counter += 1
        
        # Initialize guardians
        for i in range(AGENT_CONFIG["guardian_count"]):
            position = self._get_random_position()
            agent_id = f"guardian_{agent_id_counter}"
            self.agents[agent_id] = GuardianAgent(agent_id, position)
            agent_id_counter += 1
    
    def _get_random_position(self) -> Tuple[int, int]:
        """Generate random position within grid bounds"""
        return (
            random.randint(0, self.grid_size[0] - 1),
            random.randint(0, self.grid_size[1] - 1)
        )
    
    def _get_nearby_agents(self, position: Tuple[int, int], vision_range: int) -> List[str]:
        """Find agents within vision range of a position"""
        nearby = []
        x, y = position
        
        for agent_id, agent in self.agents.items():
            agent_x, agent_y = agent.position
            distance = np.sqrt((x - agent_x)**2 + (y - agent_y)**2)
            if distance <= vision_range:
                nearby.append(agent_id)
        
        return nearby
    
    def _get_environment_state(self, position: Tuple[int, int], agent_id: str) -> Dict[str, Any]:
        """Get environment state for an agent at a specific position"""
        risk_score = self.risk_model.get_risk_score(position)
        nearby_agents = self._get_nearby_agents(position, AGENT_CONFIG["vision_range"])
        
        # Remove self from nearby agents
        if agent_id in nearby_agents:
            nearby_agents.remove(agent_id)
        
        return {
            "risk_score": risk_score,
            "nearby_agents": nearby_agents,
            "position": position,
            "grid_size": self.grid_size
        }
    
    def _update_agent_positions(self):
        """Update agent positions based on their actions"""
        for agent_id, agent in self.agents.items():
            if hasattr(agent, 'move') and callable(getattr(agent, 'move')):
                agent.move(self.grid_size)
    
    def _process_agent_interactions(self, step: int) -> List[Dict[str, Any]]:
        """Process all agent interactions for a single time step"""
        step_events = []
        
        for agent_id, agent in self.agents.items():
            environment_state = self._get_environment_state(agent.position, agent_id)
            
            # Get agent's decision
            action_result = agent.decide_action(environment_state)
            
            # Update agent position if action involves movement
            if action_result["action"] in ["move", "evade", "patrol"]:
                agent.move(self.grid_size)
                action_result["new_position"] = agent.position
            
            # Create event record
            event = {
                "step": step,
                "agent_id": agent_id,
                "agent_type": agent.agent_type,
                "position": agent.position,
                "action": action_result["action"],
                "action_result": action_result["action_result"],
                "reward": action_result["reward"],
                "risk_score": environment_state["risk_score"],
                "reputation": agent.reputation,
                "nearby_agents": environment_state["nearby_agents"]
            }
            
            step_events.append(event)
            
            # Update agent reputation based on action result
            if action_result["action_result"] in ["successful_assault", "successful_arrest"]:
                agent.update_reputation(0.1)
            elif action_result["action_result"] in ["failed_assault", "failed_arrest"]:
                agent.update_reputation(-0.05)
        
        return step_events
    
    def run_episode(self, episode: int) -> Dict[str, Any]:
        """Run a single simulation episode"""
        print(f"Starting episode {episode}")
        
        # Initialize agents for this episode
        self.initialize_agents()
        
        episode_events = []
        total_rewards = {agent_id: 0.0 for agent_id in self.agents.keys()}
        
        for step in range(SIMULATION_CONFIG["steps_per_episode"]):
            step_events = self._process_agent_interactions(step)
            episode_events.extend(step_events)
            
            # Accumulate rewards
            for event in step_events:
                total_rewards[event["agent_id"]] += event["reward"]
            
            # Update risk terrain model periodically
            if step % 100 == 0:
                self.risk_model.update_risk_factors(episode_events[-100:])
        
        # Calculate episode statistics
        episode_stats = self._calculate_episode_stats(episode_events, total_rewards)
        
        # Save episode data
        episode_filename = FILE_PATTERNS["simulation_log"].format(episode=episode)
        episode_path = PROCESSED_DATA_DIR / episode_filename
        
        with open(episode_path, 'w') as f:
            json.dump({
                "episode": episode,
                "events": episode_events,
                "statistics": episode_stats,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)
        
        print(f"Episode {episode} completed. Events: {len(episode_events)}")
        return episode_stats
    
    def _calculate_episode_stats(self, events: List[Dict], total_rewards: Dict[str, float]) -> Dict[str, Any]:
        """Calculate statistics for the episode"""
        crime_events = [e for e in events if e["action_result"] in ["successful_assault", "failed_assault"]]
        arrest_events = [e for e in events if e["action_result"] in ["successful_arrest", "failed_arrest"]]
        
        # Spatial statistics
        crime_positions = [e["position"] for e in crime_events]
        arrest_positions = [e["position"] for e in arrest_events]
        
        # Time-based statistics
        hourly_crimes = {}
        for event in crime_events:
            hour = (event["step"] // 60) % 24  # Convert steps to hours
            hourly_crimes[hour] = hourly_crimes.get(hour, 0) + 1
        
        return {
            "total_events": len(events),
            "crime_events": len(crime_events),
            "arrest_events": len(arrest_events),
            "successful_crimes": len([e for e in crime_events if e["action_result"] == "successful_assault"]),
            "successful_arrests": len([e for e in arrest_events if e["action_result"] == "successful_arrest"]),
            "crime_positions": crime_positions,
            "arrest_positions": arrest_positions,
            "hourly_crime_distribution": hourly_crimes,
            "total_rewards": total_rewards,
            "average_risk_score": np.mean([e["risk_score"] for e in events]),
            "agent_type_counts": {
                "offender": len([e for e in events if e["agent_type"] == "offender"]),
                "target": len([e for e in events if e["agent_type"] == "target"]),
                "guardian": len([e for e in events if e["agent_type"] == "guardian"])
            }
        }
    
    def run_simulation(self, episodes: int = None) -> List[Dict[str, Any]]:
        """Run the complete simulation for multiple episodes"""
        episodes = episodes or SIMULATION_CONFIG["episodes"]
        all_episode_stats = []
        
        print(f"Starting simulation with {episodes} episodes")
        
        for episode in range(episodes):
            episode_stats = self.run_episode(episode)
            all_episode_stats.append(episode_stats)
            
            # Print progress
            if (episode + 1) % 10 == 0:
                print(f"Completed {episode + 1}/{episodes} episodes")
        
        # Save overall simulation summary
        simulation_summary = {
            "total_episodes": episodes,
            "episode_statistics": all_episode_stats,
            "simulation_config": SIMULATION_CONFIG,
            "agent_config": AGENT_CONFIG,
            "timestamp": datetime.now().isoformat()
        }
        
        summary_path = PROCESSED_DATA_DIR / "simulation_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(simulation_summary, f, indent=2)
        
        print(f"Simulation completed. Summary saved to {summary_path}")
        return all_episode_stats

if __name__ == "__main__":
    simulator = CrimeHotspotSimulator()
    simulator.run_simulation() 