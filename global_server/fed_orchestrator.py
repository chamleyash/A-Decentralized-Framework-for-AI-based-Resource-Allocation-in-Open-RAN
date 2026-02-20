from communication.serializer import Serializer

class FederatedOrchestrator:
    def __init__(self, global_server, regions, message_bus):
        self.global_server = global_server
        self.regions = regions
        self.bus = message_bus

    def run_round(self):
        for region in self.regions:
            regional_weights = region.aggregate()
            if regional_weights is None:
                continue
            serialized = Serializer.serialize(regional_weights)
            payload = self.bus.send(serialized)
            if payload is None:
                continue
            deserialized = Serializer.deserialize(payload)
            self.global_server.receive_update(deserialized)

        return self.global_server.aggregate()
