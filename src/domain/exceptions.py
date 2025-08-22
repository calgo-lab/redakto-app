class DomainException(Exception):
    """
    Base exception for domain errors
    """
    
class ModelNotFoundError(DomainException):
    """
    Raised when a model is not found in the registry for the given entity set id and model id
    """
    def __init__(self, entity_set_id: str, model_id: str):
        super().__init__(
            f"Model {model_id} not found for entity set {entity_set_id}"
        )

class UnsupportedModelLoadingStrategyError(DomainException):
    """
    Raised when an unsupported model loading strategy is encountered
    """
    def __init__(self, entity_set_id: str, model_id: str, strategy: str):
        super().__init__(
            f"Unsupported model loading strategy '{strategy}' for model {model_id} in entity set {entity_set_id}"
        )

class UnsupportedModelImplTypeError(DomainException):
    """
    Raised when an unsupported model impl type is encountered
    """
    def __init__(self, entity_set_id: str, model_id: str, model_impl: str):
        super().__init__(
            f"Unsupported model impl type: {model_impl} for model {model_id} in entity set {entity_set_id}"
        )

class InvalidModelConfigError(DomainException):
    """
    Raised when model configuration is invalid
    """
    