from pydantic import BaseModel, Field
from typing import List

class DetectEntitiesRequest(BaseModel):
    entity_set_id: str
    model_id: str
    input_texts: List[str]

class EntityItem(BaseModel):
    token_id: str = Field(..., alias='Token_ID')
    label: str = Field(..., alias='Label')
    start: int = Field(..., alias='Start')
    end: int = Field(..., alias='End')
    token: str = Field(..., alias='Token')

    class Config:
        populate_by_name = True
        allow_population_by_field_name = True

class DetectEntitiesResponse(BaseModel):
    output: List[List[EntityItem]]