"""Map-only route geometry for an already ordered trip itinerary."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, model_validator

from src.adapters.neshan_client import get_neshan_client

router = APIRouter()


class MapPoint(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    latitude: float = Field(ge=-90, le=90, allow_inf_nan=False)
    longitude: float = Field(ge=-180, le=180, allow_inf_nan=False)


class OrderedStop(MapPoint):
    order: int = Field(ge=1, le=10)


class MapRouteRequest(BaseModel):
    origin: MapPoint
    stops: list[OrderedStop] = Field(min_length=1, max_length=10)
    round_trip: bool = False

    @model_validator(mode="after")
    def check_order(self):
        if [stop.order for stop in self.stops] != list(range(1, len(self.stops) + 1)):
            raise ValueError("Stops must be in itinerary visiting order")
        coordinates = [(point.latitude, point.longitude) for point in [self.origin, *self.stops]]
        if len(set(coordinates)) != len(coordinates):
            raise ValueError("Origin and stops must have distinct coordinates")
        return self


@router.post("/map/route")
async def map_route(request: MapRouteRequest):
    """Route each consecutive pair, then return to origin when requested."""
    client = get_neshan_client()
    if not client.enabled:
        raise HTTPException(status_code=503, detail="سرویس نقشه موقتاً در دسترس نیست.")

    points = [request.origin, *request.stops]
    if request.round_trip:
        points.append(request.origin)
    legs = []
    for start, end in zip(points, points[1:]):
        leg = await client.route_geometry(
            (start.latitude, start.longitude), (end.latitude, end.longitude)
        )
        if not leg:
            raise HTTPException(status_code=502, detail="مسیر جاده‌ای یکی از بخش‌های سفر پیدا نشد.")
        legs.append({"from_name": start.name, "to_name": end.name, **leg})
    return {"legs": legs,
            "total_distance_km": round(sum(leg["distance_km"] for leg in legs), 1),
            "total_duration_hours": round(sum(leg["duration_hours"] for leg in legs), 2)}
