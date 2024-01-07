"""
Description: Factory to create face detection module.

Author: Tomas Goldmann
Date Created: Dec 26, 2023
Date Modified: Dec 26, 2023
License: MIT License
"""


import importlib



class FaceDetectorFactory:
    providers = {}

    @classmethod
    def create_detector(cls, provider, model_path, cfg):
        if provider in cls.providers:

            print(__name__, cls.providers[provider])
            module = importlib.import_module('.inference.' +cls.providers[provider], 'face_ai_kit.modules.retinaface_detector')
            module_class = getattr(module, cls.providers[provider])
            return module_class(provider,model_path, cfg)
        else:
            raise ValueError(f"Invalid provider for retinaface ")

    @classmethod
    def register_provider(cls, provider_name, provider_class_name):
        cls.providers[provider_name] = provider_class_name