from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from pydantic import BaseModel
import random
import time
import requests
import os

router = APIRouter(
    prefix="/api/agri-services",
    tags=["Agri Services"]
)

# Modèles de réponse (Mocks)
class WeatherResponse(BaseModel):
    temperature: float
    condition: str
    humidity: int
    wind_speed: float
    recommendation: str

class AirQualityResponse(BaseModel):
    aqi: int
    category: str
    dominant_pollutant: str
    health_recommendation: str

class SolarPotentialResponse(BaseModel):
    irradiance: float
    recommended_panels: int
    estimated_cost_cfa: int

class PollenResponse(BaseModel):
    risk_level: str
    types: list[str]

class PlaceResult(BaseModel):
    name: str
    address: str
    distance_km: float

class RouteResponse(BaseModel):
    distance_km: float
    duration_mins: float
    summary: str

class TimezoneResponse(BaseModel):
    time_zone_id: str
    local_time: str

# --- Endpoints ---

@router.get("/weather", response_model=WeatherResponse)
async def get_weather(lat: float = Query(...), lng: float = Query(...)):
    """
    Google Weather API.
    """
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if api_key:
        try:
            url = f"https://weather.googleapis.com/v1/currentConditions:lookup?key={api_key}&location.latitude={lat}&location.longitude={lng}"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                temp = 25.0
                condition = "Normal"
                humidity = 50
                wind_speed = 10.0
                
                if "currentConditions" in data:
                    cc = data["currentConditions"]
                    temp = cc.get("temperature", 25.0)
                    condition = cc.get("conditions", "Normal")
                    humidity = cc.get("relativeHumidity", 50)
                    wind_speed = cc.get("windSpeed", 10.0)
                elif "temperature" in data:
                    temp = data.get("temperature", 25.0)
                    condition = data.get("conditions", "Normal")
                    humidity = data.get("relativeHumidity", 50)
                    wind_speed = data.get("windSpeed", 10.0)

                rec = "Conditions normales pour la saison. Surveillez le stress hydrique."
                if temp > 38:
                    rec = "Forte chaleur. Privilégiez l'irrigation très tôt le matin ou le soir."
                elif temp < 20:
                    rec = "Températures fraîches. Adaptez les cultures si nécessaire."

                return WeatherResponse(
                    temperature=round(temp, 1),
                    condition=condition,
                    humidity=int(humidity),
                    wind_speed=round(wind_speed, 1),
                    recommendation=rec
                )
        except Exception as e:
            print(f"Erreur Weather API: {e}")

    # Fallback au Mock
    temp = round(random.uniform(25.0, 42.0), 1)
    
    if temp > 38:
        condition = "Très Ensoleillé"
        rec = "Forte chaleur. Privilégiez l'irrigation très tôt le matin ou le soir."
    elif temp > 30:
        condition = "Ensoleillé"
        rec = "Conditions normales pour la saison. Surveillez le stress hydrique."
    else:
        condition = "Nuageux / Pluie possible"
        rec = "Risque de précipitations. Bon moment pour les semis si la pluie est confirmée."

    return WeatherResponse(
        temperature=temp,
        condition=condition,
        humidity=random.randint(20, 80),
        wind_speed=round(random.uniform(5.0, 30.0), 1),
        recommendation=rec
    )

@router.get("/air-quality", response_model=AirQualityResponse)
async def get_air_quality(lat: float = Query(...), lng: float = Query(...)):
    """
    Mock pour Google Air Quality API.
    """
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    
    if api_key:
        try:
            url = f"https://airquality.googleapis.com/v1/currentConditions:lookup?key={api_key}"
            payload = {"location": {"latitude": lat, "longitude": lng}}
            response = requests.post(url, json=payload, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                
                aqi = 50
                category = "Modéré"
                pollutant = "Non déterminé"
                
                if "indexes" in data and len(data["indexes"]) > 0:
                    idx = data["indexes"][0]
                    aqi = idx.get("aqi", 50)
                    category = idx.get("category", "Modéré")
                    pollutant = idx.get("dominantPollutant", "PM10")
                    
                rec = "Qualité de l'air de base."
                if aqi < 50:
                    rec = "Qualité de l'air idéale pour le travail au champ."
                elif aqi < 100:
                    rec = "Qualité acceptable. Présence légère de particules."
                elif aqi < 150:
                    rec = "Malsain pour les groupes sensibles. Limitez l'effort physique intense."
                else:
                    rec = "Malsain. Port du masque recommandé pendant le travail."
                    
                return AirQualityResponse(
                    aqi=aqi,
                    category=category,
                    dominant_pollutant=pollutant,
                    health_recommendation=rec
                )
        except Exception as e:
            print(f"Erreur Air Quality API: {e}")

    # Fallback au Mock
    aqi = random.randint(30, 200)
    
    if aqi < 50:
        category = "Bon"
        rec = "Qualité de l'air idéale pour le travail au champ."
    elif aqi < 100:
        category = "Modéré"
        rec = "Qualité acceptable. Présence légère de poussière."
    elif aqi < 150:
        category = "Malsain pour les groupes sensibles"
        rec = "Poussière (Harmattan). Les personnes fragiles doivent limiter l'effort."
    else:
        category = "Malsain"
        rec = "Forte concentration de poussière. Port du masque recommandé pendant le travail."

    return AirQualityResponse(
        aqi=aqi,
        category=category,
        dominant_pollutant="PM10 (Poussière)",
        health_recommendation=rec
    )

@router.get("/solar-potential", response_model=SolarPotentialResponse)
async def get_solar_potential(lat: float = Query(...), lng: float = Query(...), area_sqm: float = Query(50)):
    """
    Google Solar API.
    """
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if api_key:
        try:
            url = f"https://solar.googleapis.com/v1/buildingInsights:findClosest?location.latitude={lat}&location.longitude={lng}&key={api_key}"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                
                irradiance = 6.0
                if "solarPotential" in data:
                    sp = data["solarPotential"]
                    sunshine_hours = sp.get("maxSunshineHoursPerYear", 2190)
                    irradiance = round(sunshine_hours / 365, 2)
                    
                panels = int(area_sqm / 1.6)
                cost = panels * 85000
                
                return SolarPotentialResponse(
                    irradiance=irradiance if irradiance > 0 else 6.5,
                    recommended_panels=panels,
                    estimated_cost_cfa=cost
                )
        except Exception as e:
            print(f"Erreur Solar API: {e}")

    # Fallback au Mock
    irradiance = round(random.uniform(5.5, 7.5), 2) # kWh/m2/jour au Sahel
    panels = int(area_sqm / 1.6) # Approx 1.6 m2 par panneau
    cost = panels * 85000 # Environ 85000 FCFA par panneau + onduleur
    
    return SolarPotentialResponse(
        irradiance=irradiance,
        recommended_panels=panels,
        estimated_cost_cfa=cost
    )

@router.get("/pollen", response_model=PollenResponse)
async def get_pollen(lat: float = Query(...), lng: float = Query(...)):
    """
    Mock pour Google Pollen API.
    """
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    
    if api_key:
        try:
            url = f"https://pollen.googleapis.com/v1/forecast:lookup?key={api_key}&location.latitude={lat}&location.longitude={lng}&days=1"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                risk_level = "Faible"
                types = []
                
                if "dailyInfo" in data and len(data["dailyInfo"]) > 0:
                    day_info = data["dailyInfo"][0]
                    if "pollenTypeInfo" in day_info:
                        for pollen in day_info["pollenTypeInfo"]:
                            if "displayName" in pollen:
                                types.append(pollen["displayName"])
                            # On pourrait aussi extraire l'index global, mais simplifions
                    
                    if types:
                        risk_level = "Données Google Actives"
                    
                return PollenResponse(
                    risk_level=risk_level,
                    types=types if types else ["Aucun pollen détecté"]
                )
        except Exception as e:
            print(f"Erreur Pollen API: {e}")

    # Fallback au Mock
    levels = ["Faible", "Modéré", "Élevé"]
    return PollenResponse(
        risk_level=random.choice(levels),
        types=["Herbacées", "Graminées", "Acacia"]
    )

@router.get("/places", response_model=list[PlaceResult])
async def get_places(type: str = Query("market"), lat: float = Query(...), lng: float = Query(...)):
    """ Google Places API """
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if api_key:
        url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={lat},{lng}&radius=50000&type={type}&key={api_key}"
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                results = []
                for res in data.get("results", [])[:5]:
                    results.append(PlaceResult(
                        name=res.get("name", "Lieu inconnu"),
                        address=res.get("vicinity", ""),
                        distance_km=round(random.uniform(2.0, 15.0), 1) # Simplification du calcul
                    ))
                if results:
                    return results
        except Exception as e:
            print(f"Erreur Places API: {e}")
            
    # Fallback au Mock
    return [
        PlaceResult(name="Marché Central", address="Centre ville", distance_km=4.2),
        PlaceResult(name="Coopérative Agricole", address="Secteur 4", distance_km=12.5)
    ]

@router.get("/route", response_model=RouteResponse)
async def get_route(origin_lat: float = Query(...), origin_lng: float = Query(...), dest_lat: float = Query(...), dest_lng: float = Query(...)):
    """ Google Directions API """
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if api_key:
        url = f"https://maps.googleapis.com/maps/api/directions/json?origin={origin_lat},{origin_lng}&destination={dest_lat},{dest_lng}&key={api_key}"
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("routes"):
                    leg = data["routes"][0]["legs"][0]
                    return RouteResponse(
                        distance_km=round(float(leg["distance"]["value"])/1000, 1),
                        duration_mins=round(float(leg["duration"]["value"])/60, 1),
                        summary=data["routes"][0].get("summary", "Route principale")
                    )
        except Exception as e:
            print(f"Erreur Directions API: {e}")
            
    # Fallback au Mock
    return RouteResponse(distance_km=24.5, duration_mins=45.0, summary="N1 Route Nationale")

@router.get("/timezone", response_model=TimezoneResponse)
async def get_timezone(lat: float = Query(...), lng: float = Query(...)):
    """ Google Time Zone API """
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    timestamp = int(time.time())
    if api_key:
        url = f"https://maps.googleapis.com/maps/api/timezone/json?location={lat},{lng}&timestamp={timestamp}&key={api_key}"
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("status") == "OK":
                    import datetime
                    local_time = datetime.datetime.utcnow() + datetime.timedelta(seconds=data["rawOffset"] + data.get("dstOffset", 0))
                    return TimezoneResponse(
                        time_zone_id=data["timeZoneId"],
                        local_time=local_time.strftime("%Y-%m-%d %H:%M")
                    )
        except Exception as e:
            print(f"Erreur Timezone API: {e}")
            
    return TimezoneResponse(time_zone_id="Africa/Ouagadougou", local_time="14:30 (Heure locale estimée)")
