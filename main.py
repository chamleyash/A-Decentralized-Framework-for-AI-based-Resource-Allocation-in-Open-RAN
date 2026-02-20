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
import matplotlib.pyplot as plt
import torch.optim as optim
import time
import os
import json

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
    # Regional aggregators
    regions = [RegionalAggregator(region_id=i) for i in range(sys_cfg["system"]["num_regions"])]

    # Global server
    global_server = GlobalServer()
    orchestrator = FederatedOrchestrator(global_server, regions, bus)

    # Placeholder execution loop
    print("STARTING DECENTRALIZED TRAINING")
    
    # -------------------------------
    # Metrics
    # -------------------------------

    loss_history = []
    successful_rounds = 0
    failed_rounds = 0
    total_coord_time = 0.0

    # -------------------------------
    # Training loop
    # -------------------------------

    for r in range(sys_cfg["system"]["aggregation_rounds"]):

        round_losses = []

        # Local training at edges
        for edge in edges:
            edge.step()
            update, loss = edge.local_update()   # make sure local_update returns (weights, loss)
            round_losses.append(loss)

            serialized = Serializer.serialize(update)
            payload = bus.send(serialized)

            if payload is not None:
                deserialized = Serializer.deserialize(payload)
                regions[0].receive_update(deserialized)

        start_time = time.time()
        global_weights = orchestrator.run_round()
        coordination_time = time.time() - start_time

        total_coord_time += coordination_time

        if global_weights is not None:
            successful_rounds += 1

            for edge in edges:
                edge.set_weights(global_weights)

            avg_round_loss = sum(round_losses) / len(round_losses)
            loss_history.append(avg_round_loss)

        else:
            failed_rounds += 1

    # -------------------------------
    # Final Results
    # -------------------------------

    avg_coord_time = total_coord_time / successful_rounds

    print("\n===== HIERARCHICAL RESULTS =====")
    print("Total rounds:", sys_cfg["system"]["aggregation_rounds"])
    print("Successful rounds:", successful_rounds)
    print("Failed rounds:", failed_rounds)
    print("Average coordination time per round:", avg_coord_time)

    if loss_history:
        os.makedirs("Loss", exist_ok=True)
        with open("Loss/hierarchical_loss.json", "w") as f:
            json.dump({
                "loss_history": loss_history,
                "avg_coord_time": avg_coord_time
            }, f)
        print("Initial loss:", loss_history[0])
        print("Final loss:", loss_history[-1])
        print("Average loss:", sum(loss_history) / len(loss_history))
    plt.plot(loss_history, label="Loss")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Rounds")
    plt.legend()
    plt.show()
    
    
if __name__ == "__main__":
    main()