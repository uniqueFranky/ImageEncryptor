class Registry:
    def __init__(self):
        self.registry = {}

    def register(self, name):
        def decorator(cls):
            self.registry[name] = cls
            return cls
        return decorator
    
    def get_class(self, name):
        cls = self.registry.get(name)
        if cls is None:
            raise ValueError(f'Unregistered Encryptor: {name}')
        return cls

    def build(self, name, *args, **kwargs):
        cls = self.get_class(name)
        return cls(*args, **kwargs)
    

encryptor_registry = Registry()
chaos_operation_registry = Registry()
chaos_mapping_registry = Registry()

