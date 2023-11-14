


class N19LandmarksFactory:
    providers = {}

    @classmethod
    def create(cls, algorithm, provider, model_path):

        if provider in cls.providers[algorithm]:
            return cls.providers[algorithm][provider](provider,model_path)
        else:
            raise ValueError("Invalid provider")

    @classmethod
    def register_provider(cls, algorithm, provider_name, provider_class):
        cls.providers[algorithm]={}
        cls.providers[algorithm][provider_name] = provider_class