from fastapi import APIRouter, Depends
from src.application.schemas import ApiRequest, ApiResponse
from src.application.services import PredictionService

router = APIRouter(prefix="/predict", tags=["prediction"])

@router.post("/", response_model=ApiResponse)
async def predict(request: ApiRequest, 
                  service: PredictionService = Depends()):
    return service.process(request)