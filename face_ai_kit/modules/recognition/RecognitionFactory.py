"""
Description: Factory to create face recognition module.

Author: Tomas Goldmann
Date Created: Dec 26, 2023
Date Modified: Dec 26, 2023
License: MIT License
"""


class RecognitionFactory:
    providers = {}

    @classmethod
    def create(cls, algorithm, provider, model_path):

        if provider in cls.providers[algorithm]:
            return cls.providers[algorithm][provider](provider,model_path)
        else:
            raise ValueError(f"Invalid face recognition provider for {algorithm}")

    @classmethod
    def register_provider(cls, algorithm, provider_name, provider_class):
        if algorithm not in cls.providers:
            cls.providers[algorithm]={}
        cls.providers[algorithm][provider_name] = provider_class