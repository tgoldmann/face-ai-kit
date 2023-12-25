


class FaceDetectorFactory:
    providers = {}

    @classmethod
    def create_detector(cls, provider, model_path, cfg):
        if provider in cls.providers:
            return cls.providers[provider](provider,model_path, cfg)
        else:
            raise ValueError(f"Invalid provider for retinaface ")

    @classmethod
    def register_provider(cls, provider_name, provider_class):
        cls.providers[provider_name] = provider_class