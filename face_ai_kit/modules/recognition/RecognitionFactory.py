"""
Description: Factory to create face recognition module.

Author: Tomas Goldmann
Date Created: Dec 26, 2023
Date Modified: Dec 26, 2023
License: MIT License
"""

import importlib


class RecognitionFactory:
    providers = {}

    @classmethod
    def create(cls, algorithm, provider, model_path):

        if provider in cls.providers[algorithm]:
            module = importlib.import_module( '.' + algorithm + '.inference.' +cls.providers[algorithm][provider], 'face_ai_kit.modules.recognition')
            module_class = getattr(module, cls.providers[algorithm][provider])
            return module_class(provider,model_path)
            return cls.providers[algorithm][provider](provider,model_path)
        else:
            raise ValueError(f"Invalid face recognition provider for {algorithm}")

    @classmethod
    def register_provider(cls, algorithm, provider_name, provider_class):
        if algorithm not in cls.providers:
            cls.providers[algorithm]={}
        cls.providers[algorithm][provider_name] = provider_class