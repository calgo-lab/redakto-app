class DomainException(Exception):
    """Base exception for domain errors"""
    
class ModelNotFoundError(DomainException):
    def __init__(self, entity_set_id: str, model_id: str):
        super().__init__(
            f"Model {model_id} not found for entity set {entity_set_id}"
        )

class UnsupportedModelTypeError(DomainException):
    """Raised when an unsupported model type is encountered"""
    
    def __init__(self, entity_set_id: str, model_id: str, model_type: str):
        super().__init__(
            f"Unsupported model type: {model_type} for model {model_id} in entity set {entity_set_id}"
        )


class InvalidModelConfigError(DomainException):
    """Raised when model configuration is invalid"""