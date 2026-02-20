import time
import random

class MessageBus:
    def __init__(self, latency_ms=0, packet_loss_prob=0.0):
        self.latency_ms = latency_ms
        self.packet_loss_prob = packet_loss_prob

    def send(self, payload):
        if random.random() < self.packet_loss_prob:
            return None
        time.sleep(self.latency_ms / 1000.0)
        return payload
