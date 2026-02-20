import torch
import io

class Serializer:
    @staticmethod
    def serialize(model_weights):
        buffer = io.BytesIO()
        torch.save(model_weights, buffer)
        buffer.seek(0)
        return buffer

    @staticmethod
    def deserialize(buffer):
        buffer.seek(0)
        return torch.load(buffer)
