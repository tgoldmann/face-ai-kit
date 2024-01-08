"""
Description: Factory for create Mediapipe face landmarks detector 

Author: Tomas Goldmann
Date Created: Dec 26, 2023
Date Modified: Dec 26, 2023
License: MIT License
"""

import importlib


class MediapipeLandmarksFactory:
    providers = {}

    @classmethod
    def create(cls, algorithm, provider, model_path):

        if provider in cls.providers[algorithm]:
            module = importlib.import_module('.inference.' +cls.providers[algorithm][provider], 'face_ai_kit.modules.mediapipe_landmarks')
            module_class = getattr(module, cls.providers[algorithm][provider])
            return module_class(provider,model_path)
        else:
            raise ValueError("Invalid provider")

    @classmethod
    def register_provider(cls, algorithm, provider_name, provider_class):
        cls.providers[algorithm]={}
        cls.providers[algorithm][provider_name] = provider_class