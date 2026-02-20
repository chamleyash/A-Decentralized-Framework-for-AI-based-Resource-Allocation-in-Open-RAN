import torch

class EdgeAgent:
    def __init__(self, agent_id, model, env, trainer, collector):
        self.agent_id = agent_id
        self.model = model
        self.env = env
        self.trainer = trainer
        self.collector = collector

    def act(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        return self.model(state_tensor)

    def step(self):
        logits = self.act(self.env.state)
        action = torch.argmax(logits).item()
        next_state, reward = self.env.step(action)
        self.collector.collect(self.env.state, action, reward, next_state)

    def local_update(self):
        data = self.collector.get_data()
        self.trainer.train(data)
        self.collector.clear()
        return self.trainer.get_model_weights()
    
    def set_weights(self, global_weights):
        if global_weights is None:
            return
        self.model.load_state_dict(global_weights)