from utils.config_loader import ConfigLoader
from models.policy_network import PolicyNetwork
from simulation.environment import NetworkEnvironment
from edge.edge_agent import EdgeAgent
from edge.local_trainer import LocalTrainer
from edge.data_collector import DataCollector
from global_server.global_server import GlobalServer
import torch.optim as optim
import time


def main():

    config = ConfigLoader()
    sys_cfg = config.system_config
    model_cfg = config.model_config

    print("STARTING CENTRALIZED TRAINING")

    # -------------------------------
    # Create single global model
    # -------------------------------

    global_model = PolicyNetwork(
        model_cfg["model"]["state_dim"],
        model_cfg["model"]["action_dim"],
        model_cfg["model"]["hidden_layers"]
    )

    optimizer = optim.Adam(
        global_model.parameters(),
        lr=model_cfg["training"]["learning_rate"]
    )

    trainer = LocalTrainer(global_model, optimizer)

    global_model = PolicyNetwork(
            model_cfg["model"]["state_dim"],
            model_cfg["model"]["action_dim"],
            model_cfg["model"]["hidden_layers"]
        )
    optimizer = optimizer = optim.Adam(
            global_model.parameters(),
            lr=model_cfg["training"]["learning_rate"]
        )
    trainer = LocalTrainer(global_model, optimizer)

    # -------------------------------
    # Create edges (data producers only)
    # -------------------------------

    edges = []

    for i in range(sys_cfg["system"]["num_edge_agents"]):

        env = NetworkEnvironment(
            model_cfg["model"]["state_dim"],
            model_cfg["model"]["action_dim"]
        )

        agent = EdgeAgent(
            agent_id=i,
            model=global_model,      # shared global model reference
            env=env,
            trainer=None,            # no local trainer
            collector=DataCollector()
        )

        edges.append(agent)

    # -------------------------------
    # Metrics
    # -------------------------------

    total_coord_time = 0.0

    # -------------------------------
    # Training loop
    # -------------------------------

    for r in range(sys_cfg["system"]["aggregation_rounds"]):

        edge_batches = []

        # Edges generate experience only
        for edge in edges:
            edge.step()
            batch = edge.collector.get_data()
            edge_batches.append(batch)

        # Combine data centrally
        combined_batch = []
        for batch in edge_batches:
            combined_batch.extend(batch)

        # Train global model directly
        start_time = time.time()

        if combined_batch:
            trainer.train(combined_batch)

        coordination_time = time.time() - start_time
        total_coord_time += coordination_time

    # -------------------------------
    # Final metrics
    # -------------------------------

    avg_coord_time = total_coord_time / sys_cfg["system"]["aggregation_rounds"]

    print("\n===== CENTRALIZED RESULTS =====")
    print("Total rounds:", sys_cfg["system"]["aggregation_rounds"])
    print("Average coordination time per round:", avg_coord_time)


if __name__ == "__main__":
    main()