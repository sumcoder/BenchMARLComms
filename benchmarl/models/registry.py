attention_registry = {}


def register_attention(name):
    def register_class(cls):
        attention_registry[name] = cls
        return cls
    return register_class