from pydantic import BaseModel, Field
from typing import List

class DetectEntitiesAndPseudonymizeRequest(BaseModel):
    entity_set_id: str
    model_id: str
    input_texts: List[str]
    repeat: int

class EntityItem(BaseModel):
    token_id: str = Field(..., alias='Token_ID')
    label: str = Field(..., alias='Label')
    start: int = Field(..., alias='Start')
    end: int = Field(..., alias='End')
    token: str = Field(..., alias='Token')
    pseudonym: str = Field(..., alias='Pseudonym')

    class Config:
        populate_by_name = True
        allow_population_by_field_name = True

class TextItem(BaseModel):
    entities: List[EntityItem]
    pseudonymized_text: str

class DetectEntitiesAndPseudonymizeResponse(BaseModel):
    output: List[List[TextItem]]