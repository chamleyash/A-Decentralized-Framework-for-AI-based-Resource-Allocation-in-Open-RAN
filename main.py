from utils.config_loader import ConfigLoader
from models.policy_network import PolicyNetwork
from simulation.environment import NetworkEnvironment
from edge.edge_agent import EdgeAgent
from edge.local_trainer import LocalTrainer
from edge.data_collector import DataCollector
from regional.regional_aggregator import RegionalAggregator
from global_server.global_server import GlobalServer
from global_server.fed_orchestrator import FederatedOrchestrator
from communication.message_bus import MessageBus
from communication.serializer import Serializer
import torch.optim as optim
import time

def main():
    config = ConfigLoader()
    sys_cfg = config.system_config
    model_cfg = config.model_config
    
    bus = MessageBus(
    latency_ms=sys_cfg["communication"]["latency_ms"],
    packet_loss_prob=sys_cfg["communication"]["packet_loss_prob"]
    )

    # Create edge agents
    edges = []
    for i in range(sys_cfg["system"]["num_edge_agents"]):

        # Model prototype
        model = PolicyNetwork(
            model_cfg["model"]["state_dim"],
            model_cfg["model"]["action_dim"],
            model_cfg["model"]["hidden_layers"]
        )
        optimizer = optim.Adam(
            model.parameters(),
            lr=model_cfg["training"]["learning_rate"]
        )
        trainer = LocalTrainer(model, optimizer)
        
        env = NetworkEnvironment(
            model_cfg["model"]["state_dim"],
            model_cfg["model"]["action_dim"]
        )
        agent = EdgeAgent(
            agent_id=i,
            model=model,
            env=env,
            trainer=trainer,
            collector=DataCollector()
        )
        edges.append(agent)
    print("Edges Ok")
    # Regional aggregators
    regions = [RegionalAggregator(region_id=i) for i in range(sys_cfg["system"]["num_regions"])]

    # Global server
    global_server = GlobalServer()
    orchestrator = FederatedOrchestrator(global_server, regions, bus)

    # Placeholder execution loop
    print("STARTING TRAINING LOOP")
    for r in range(sys_cfg["system"]["aggregation_rounds"]):
        print(f"ROUND {r} START")
        for edge in edges:
            edge.step()
            update = edge.local_update()
            serialized = Serializer.serialize(update)
            payload = bus.send(serialized)

            if payload is not None:
                deserialized = Serializer.deserialize(payload)
                regions[0].receive_update(deserialized)
            print("EDGE→REGIONAL sent:", payload is not None)

        start_time = time.time()
        global_weights = orchestrator.run_round()
        coordination_time = time.time() - start_time

        print(f"ROUND {r} coordination_time={coordination_time:.4f}s")
        print(f"ROUND {r} DONE, global_weights is None: {global_weights is None}")
    
    
if __name__ == "__main__":
    main()