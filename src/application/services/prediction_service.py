from domain.repositories.model_repository import ModelRepository

class PredictionService:
    def __init__(self, model_repo: ModelRepository):
        self.repo = model_repo
        
    def process(self, entity_set_id: str, model_id: str, input_data):
        model_data = self.repo.get_model(entity_set_id, model_id)
        config = self.repo.get_model_config(entity_set_id, model_id)
        # Processing logic using abstract interface