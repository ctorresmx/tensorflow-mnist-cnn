import tensorflow as tf

class BasicModel:
    def infer(self, input):
        raise NotImplementedError()

    def cost(self, logits, labels):
        raise NotImplementedError()

    def optimize(self, cost, learning_rate):
        raise NotImplementedError()

    def evaluate(self, logits, labels):
        raise NotImplementedError()
