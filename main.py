from fastapi import FastAPI, HTTPException, Depends, Security, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, BigInteger, func, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from datetime import datetime, timedelta, timezone
from typing import Optional, List
import math
import os
import secrets
from dotenv import load_dotenv
import logging

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

# Load environment variables
load_dotenv()

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database connection from environment variables
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:password@localhost:5432/network_monitor"
)

# Create engine with connection pooling for production
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,  # Verify connections before using
    pool_recycle=3600,   # Recycle connections after 1 hour
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

# API Key Authentication
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
API_KEYS = set(filter(None, os.getenv("API_KEYS", "").split(",")))

if not API_KEYS:
    # Generate a random key on startup if none are configured, and log it
    _default_key = secrets.token_urlsafe(32)
    API_KEYS.add(_default_key)
    logging.getLogger(__name__).warning(
        f"No API_KEYS env var set. Generated temporary key: {_default_key}"
    )


def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    if not api_key or api_key not in API_KEYS:
        raise HTTPException(
            status_code=401, detail="Invalid or missing API key")
    return api_key

# Database Model


class DeviceReading(Base):
    __tablename__ = "device_readings"

    id = Column(BigInteger, primary_key=True, index=True)
    source = Column(String(50), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    latitude = Column(Float)
    longitude = Column(Float)
    level = Column(Integer)
    asu = Column(Integer)
    rsrp = Column(Integer)
    rssi = Column(Integer)
    altitude = Column(Float)
    rsrq = Column(Integer)
    network_type = Column(String(20))
    operator = Column(String(100))
    cell_id = Column(String(100))
    physical_cell_id = Column(Integer)
    tracking_area_code = Column(Integer)
    country = Column(String(100))
    city = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


# Create tables
Base.metadata.create_all(bind=engine)

# Pydantic Models


class NetworkDataRequest(BaseModel):
    source: str = Field(..., min_length=1, max_length=50)
    timestamp: Optional[str] = None
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)
    altitude: float = Field(..., ge=-430, le=8850, description="Altitude in metres")
    level: Optional[int] = None
    asu: Optional[int] = None
    rsrp: Optional[int] = None
    rssi: Optional[int] = None
    altitude: Optional[float] = None
    rsrq: Optional[int] = None
    networkType: Optional[str] = Field(None, max_length=20)
    operator: Optional[str] = Field(None, max_length=100)
    cellId: Optional[str] = Field(None, max_length=100)
    physicalCellId: Optional[int] = None
    trackingAreaCode: Optional[int] = None
    country: Optional[str] = Field(None, max_length=100)
    city: Optional[str] = Field(None, max_length=100)


class NetworkDataResponse(BaseModel):
    id: int
    source: str
    timestamp: str
    latitude: Optional[float]
    longitude: Optional[float]
    altitude: Optional[float]
    level: Optional[int]
    asu: Optional[int]
    rsrp: Optional[int]
    rssi: Optional[int]
    altitude: Optional[float]
    rsrq: Optional[int]
    network_type: Optional[str]
    operator: Optional[str]
    cell_id: Optional[str]
    physical_cell_id: Optional[int]
    tracking_area_code: Optional[int]
    created_at: str


class BatchNetworkDataRequest(BaseModel):
    """Request model for batch processing multiple sensor readings"""
    readings: List[NetworkDataRequest] = Field(
        ...,
        min_items=1,
        max_items=100,
        description="Array of sensor readings (max 100 per request)"
    )


class BatchNetworkDataResponse(BaseModel):
    """Response model for batch processing results"""
    total_submitted: int
    successful: int
    failed: int
    details: List[dict]

# ---------------------------------------------------------------------------
# Pydantic – mobile response models
# ---------------------------------------------------------------------------
class OverviewResponse(BaseModel):
    mean_rsrp: Optional[float]
    mean_rsrq: Optional[float]
    coverage_quality_percent: Optional[float]
    measurements_count: int
    density_score: Optional[float]


class MapPoint(BaseModel):
    latitude: float
    longitude: float
    rsrp: Optional[int]
    rsrq: Optional[int]


class MapResponse(BaseModel):
    points: List[MapPoint]


class TrendPoint(BaseModel):
    timestamp: str
    mean_rsrp: Optional[float]
    mean_rsrq: Optional[float]


class TrendsResponse(BaseModel):
    points: List[TrendPoint]


class FiltersResponse(BaseModel):
    operators: List[str]


class DeviceSummaryResponse(BaseModel):
    device_id: str
    reading_count: int
    last_reading: Optional[str]


class DeviceReadingResponse(BaseModel):
    device_id: str
    timestamp: str
    latitude: Optional[float]
    longitude: Optional[float]
    altitude: Optional[float]
    level: Optional[int]
    asu: Optional[int]
    rsrp: Optional[int]
    rssi: Optional[int]
    rsrq: Optional[int]
    network_type: Optional[str]
    operator: Optional[str]
    cell_id: Optional[str]
    physical_cell_id: Optional[int]
    tracking_area_code: Optional[int]
    country: Optional[str]
    city: Optional[str]
    created_at: str


class DeviceLocationPoint(BaseModel):
    device_id: str
    latitude: float
    longitude: float
    rsrp: Optional[int]
    rsrq: Optional[int]
    level: Optional[int]
    operator: Optional[str]
    network_type: Optional[str]
    timestamp: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Haversine distance in km (used as a Python-side filter fallback;
# for large datasets consider a PostGIS extension instead)
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# Haversine distance expressed as a SQLAlchemy expression (pure SQL, no PostGIS)
def haversine_sql_km(lat: float, lon: float):
    """Returns a SQLAlchemy column expression for distance in km from (lat, lon)."""
    R = 6371.0
    lat_r = math.radians(lat)
    lon_r = math.radians(lon)
    return (
        R * 2 * func.atan2(
            func.sqrt(
                func.pow(func.sin((func.radians(DeviceReading.latitude) - lat_r) / 2), 2)
                + func.cos(lat_r)
                * func.cos(func.radians(DeviceReading.latitude))
                * func.pow(func.sin((func.radians(DeviceReading.longitude) - lon_r) / 2), 2)
            ),
            func.sqrt(
                1 - (
                    func.pow(func.sin((func.radians(DeviceReading.latitude) - lat_r) / 2), 2)
                    + func.cos(lat_r)
                    * func.cos(func.radians(DeviceReading.latitude))
                    * func.pow(func.sin((func.radians(DeviceReading.longitude) - lon_r) / 2), 2)
                )
            ),
        )
    )


GOOD_RSRP_THRESHOLD = -100  # dBm – LTE "acceptable" signal floor

PERIOD_DELTA = {
    "24h": timedelta(hours=24),
    "week": timedelta(days=7),
    "month": timedelta(days=30),
}

TRENDS_TRUNC = {
    "24h": "hour",
    "week": "day",
    "month": "day",
}


def apply_mobile_filters(
    query,
    operator: Optional[str],
    network_type: Optional[str],
    period: Optional[str],
    source: Optional[str],
    lat: Optional[float],
    lon: Optional[float],
    radius_km: Optional[float],
):
    """Apply all common mobile query filters and return the modified query."""
    if operator:
        query = query.filter(DeviceReading.operator == operator)
    if network_type:
        query = query.filter(DeviceReading.network_type == network_type)
    if period and period in PERIOD_DELTA:
        cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - PERIOD_DELTA[period]
        query = query.filter(DeviceReading.timestamp >= cutoff)
    if source and source.lower() != "all":
        query = query.filter(DeviceReading.source == source)
    if lat is not None and lon is not None and radius_km is not None:
        dist = haversine_sql_km(lat, lon)
        query = query.filter(
            DeviceReading.latitude.isnot(None),
            DeviceReading.longitude.isnot(None),
            dist <= radius_km,
        )
    return query

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

# FastAPI App
app = FastAPI(
    title="Network Monitor API",
    description="API for collecting and retrieving network signal data",
    version="2.0.0"
)

# CORS Configuration - Update with your frontend domain in production
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# System endpoints
# ---------------------------------------------------------------------------


@app.get("/")
def root():
    return {
        "message": "Network Monitor API is running!",
        "status": "OK",
        "version": "2.0.0"
    }


@app.get("/health")
def health(db: Session = Depends(get_db), _: str = Depends(verify_api_key)):
    try:
        # Test database connection
        db.execute(text("SELECT 1"))
        return {
            "status": "healthy",
            "database": "connected",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=503, detail="Database connection failed")

# ---------------------------------------------------------------------------
# Ingest endpoints
# ---------------------------------------------------------------------------

@app.post("/api/network-data", response_model=dict)
def create_network_data(
    data: NetworkDataRequest,
    db: Session = Depends(get_db),
    _: str = Depends(verify_api_key),
):
    try:
        ts = _parse_timestamp(data.timestamp)
        db_reading = _build_reading(data, ts)
        db.add(db_reading)
        db.commit()
        db.refresh(db_reading)
        logger.info(f"Data saved for source {data.source}, ID: {db_reading.id}")
        return {"message": "Data saved successfully", "id": db_reading.id}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error saving data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/api/network-data/batch", response_model=BatchNetworkDataResponse)
def create_batch_network_data(
    batch_data: BatchNetworkDataRequest,
    db: Session = Depends(get_db),
    _: str = Depends(verify_api_key),
):
    """Process up to 100 sensor readings in a single request."""
    successful = 0
    failed = 0
    details: List[dict] = []
    db_readings: List[tuple] = []

    for idx, data in enumerate(batch_data.readings):
        try:
            ts = _parse_timestamp(data.timestamp)
            db_readings.append((idx, data.source, _build_reading(data, ts)))
        except HTTPException as e:
            failed += 1
            details.append({"index": idx, "source": data.source, "status": "failed", "error": e.detail})
        except Exception as e:
            failed += 1
            details.append({"index": idx, "source": data.source, "status": "failed", "error": str(e)})

    if db_readings:
        try:
            for _, _, reading in db_readings:
                db.add(reading)
            db.commit()
            for idx, source, reading in db_readings:
                db.refresh(reading)
                successful += 1
                details.append({"index": idx, "source": source, "status": "success", "id": reading.id})
            logger.info(f"Batch processed: {successful} successful, {failed} failed")
        except Exception as e:
            db.rollback()
            failed += len(db_readings)
            logger.error(f"Batch commit error: {str(e)}")
            details = [
                {"index": idx, "source": src, "status": "failed", "error": "Database commit error"}
                for idx, src, _ in db_readings
            ] + [d for d in details if d["status"] == "failed"]

    return BatchNetworkDataResponse(
        total_submitted=len(batch_data.readings),
        successful=successful,
        failed=failed,
        details=details,
    )


# ---------------------------------------------------------------------------
# Device read endpoints
# ---------------------------------------------------------------------------

@app.get("/api/devices", response_model=List[DeviceSummaryResponse])
def get_devices(
    db: Session = Depends(get_db),
    _: str = Depends(verify_api_key),
):
    """Return all distinct device IDs (`source`) with reading count and latest timestamp."""
    try:
        rows = (
            db.query(
                DeviceReading.source.label("device_id"),
                func.count(DeviceReading.id).label("reading_count"),
                func.max(DeviceReading.timestamp).label("last_reading"),
            )
            .filter(DeviceReading.source.isnot(None))
            .group_by(DeviceReading.source)
            .order_by(DeviceReading.source)
            .all()
        )

        return [
            DeviceSummaryResponse(
                device_id=row.device_id,
                reading_count=row.reading_count or 0,
                last_reading=row.last_reading.isoformat() if row.last_reading else None,
            )
            for row in rows
        ]
    except Exception as e:
        logger.error(f"Devices error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/api/readings/latest", response_model=Optional[DeviceReadingResponse])
def get_latest_reading(
    device_id: str = Query(..., min_length=1),
    db: Session = Depends(get_db),
    _: str = Depends(verify_api_key),
):
    """Return the latest reading for one device (`source`)."""
    try:
        reading = (
            db.query(DeviceReading)
            .filter(DeviceReading.source == device_id)
            .order_by(DeviceReading.timestamp.desc(), DeviceReading.id.desc())
            .first()
        )
        if not reading:
            return None
        return _device_reading_to_response(reading)
    except Exception as e:
        logger.error(f"Latest reading error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/api/readings/history", response_model=List[DeviceReadingResponse])
def get_device_reading_history(
    device_id: str = Query(..., min_length=1),
    limit: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db),
    _: str = Depends(verify_api_key),
):
    """Return recent readings for one device (`source`)."""
    try:
        rows = (
            db.query(DeviceReading)
            .filter(DeviceReading.source == device_id)
            .order_by(DeviceReading.timestamp.desc(), DeviceReading.id.desc())
            .limit(limit)
            .all()
        )
        return [_device_reading_to_response(row) for row in reversed(rows)]
    except Exception as e:
        logger.error(f"History error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/api/readings/locations", response_model=List[DeviceLocationPoint])
def get_device_locations(
    db: Session = Depends(get_db),
    _: str = Depends(verify_api_key),
):
    """Return the latest geo-located reading for each device."""
    try:
        latest_subquery = (
            db.query(
                DeviceReading.source.label("device_id"),
                func.max(DeviceReading.timestamp).label("max_timestamp"),
            )
            .filter(
                DeviceReading.source.isnot(None),
                DeviceReading.latitude.isnot(None),
                DeviceReading.longitude.isnot(None),
            )
            .group_by(DeviceReading.source)
            .subquery()
        )

        rows = (
            db.query(DeviceReading)
            .join(
                latest_subquery,
                (DeviceReading.source == latest_subquery.c.device_id)
                & (DeviceReading.timestamp == latest_subquery.c.max_timestamp),
            )
            .order_by(DeviceReading.source)
            .all()
        )

        deduped = {}
        for row in rows:
            deduped[row.source] = row

        return [
            DeviceLocationPoint(
                device_id=row.source,
                latitude=row.latitude,
                longitude=row.longitude,
                rsrp=row.rsrp,
                rsrq=row.rsrq,
                level=row.level,
                operator=row.operator,
                network_type=row.network_type,
                timestamp=row.timestamp.isoformat(),
            )
            for row in deduped.values()
            if row.latitude is not None and row.longitude is not None
        ]
    except Exception as e:
        logger.error(f"Locations error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# ---------------------------------------------------------------------------
# Query endpoints (legacy / admin)
# ---------------------------------------------------------------------------

# REMOVED

# ---------------------------------------------------------------------------
# Mobile endpoints
# ---------------------------------------------------------------------------
@app.get("/api/mobile/overview", response_model=OverviewResponse)
def mobile_overview(
    operator: Optional[str] = Query(None),
    network_type: Optional[str] = Query(None),
    period: Optional[str] = Query(None, pattern="^(24h|week|month)$"),
    source: Optional[str] = Query(None, description="measured | prediction | all"),
    lat: Optional[float] = Query(None, ge=-90, le=90),
    lon: Optional[float] = Query(None, ge=-180, le=180),
    radius_km: Optional[float] = Query(None, gt=0),
    db: Session = Depends(get_db),
    _: str = Depends(verify_api_key),
):
    """
    Aggregate signal overview for the given filters.

    - **coverage_quality_percent**: % of samples with RSRP ≥ -100 dBm
    - **density_score**: samples / area (km²),  area = π x r²
    """
    try:
        q = apply_mobile_filters(
            db.query(DeviceReading),
            operator, network_type, period, source, lat, lon, radius_km,
        )

        agg = q.with_entities(
            func.avg(DeviceReading.rsrp).label("mean_rsrp"),
            func.avg(DeviceReading.rsrq).label("mean_rsrq"),
            func.count(DeviceReading.id).label("total"),
            func.sum(
                func.cast(DeviceReading.rsrp >= GOOD_RSRP_THRESHOLD, Integer)
            ).label("good_count"),
        ).one()

        total = agg.total or 0
        good = agg.good_count or 0
        coverage_pct = round((good / total) * 100, 2) if total else None

        density: Optional[float] = None
        if radius_km and total:
            area_km2 = math.pi * radius_km ** 2
            density = round(total / area_km2, 4)

        return OverviewResponse(
            mean_rsrp=round(agg.mean_rsrp, 2) if agg.mean_rsrp is not None else None,
            mean_rsrq=round(agg.mean_rsrq, 2) if agg.mean_rsrq is not None else None,
            coverage_quality_percent=coverage_pct,
            measurements_count=total,
            density_score=density,
        )
    except Exception as e:
        logger.error(f"Overview error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/api/mobile/map", response_model=MapResponse)
def mobile_map(
    operator: Optional[str] = Query(None),
    network_type: Optional[str] = Query(None),
    period: Optional[str] = Query(None, pattern="^(24h|week|month)$"),
    source: Optional[str] = Query(None),
    lat: Optional[float] = Query(None, ge=-90, le=90),
    lon: Optional[float] = Query(None, ge=-180, le=180),
    radius_km: Optional[float] = Query(None, gt=0),
    db: Session = Depends(get_db),
    _: str = Depends(verify_api_key),
):
    """
    Returns geo-located signal samples filtered by the given criteria.

    Points are deduplicated by rounding coordinates to ~1 km grid cells
    (3 decimal places) and returning the average RSRP / RSRQ per cell
    to avoid overwhelming the client with redundant data.
    """
    try:
        q = apply_mobile_filters(
            db.query(DeviceReading),
            operator, network_type, period, source, lat, lon, radius_km,
        ).filter(
            DeviceReading.latitude.isnot(None),
            DeviceReading.longitude.isnot(None),
        )

        # Group by ~1 km grid cell (3 d.p. ≈ 110 m resolution)
        lat_cell = func.round(func.cast(DeviceReading.latitude, Float), 3)
        lon_cell = func.round(func.cast(DeviceReading.longitude, Float), 3)

        rows = (
            q.with_entities(
                lat_cell.label("lat_cell"),
                lon_cell.label("lon_cell"),
                func.avg(DeviceReading.rsrp).label("mean_rsrp"),
                func.avg(DeviceReading.rsrq).label("mean_rsrq"),
            )
            .group_by(lat_cell, lon_cell)
            .limit(5000)          # hard cap to protect client
            .all()
        )

        return MapResponse(
            points=[
                MapPoint(
                    latitude=float(r.lat_cell),
                    longitude=float(r.lon_cell),
                    rsrp=int(round(r.mean_rsrp)) if r.mean_rsrp is not None else None,
                    rsrq=int(round(r.mean_rsrq)) if r.mean_rsrq is not None else None,
                )
                for r in rows
            ]
        )
    except Exception as e:
        logger.error(f"Map error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/api/mobile/trends", response_model=TrendsResponse)
def mobile_trends(
    operator: Optional[str] = Query(None),
    network_type: Optional[str] = Query(None),
    period: Optional[str] = Query(None, pattern="^(24h|week|month)$"),
    source: Optional[str] = Query(None),
    lat: Optional[float] = Query(None, ge=-90, le=90),
    lon: Optional[float] = Query(None, ge=-180, le=180),
    radius_km: Optional[float] = Query(None, gt=0),
    db: Session = Depends(get_db),
    _: str = Depends(verify_api_key),
):
    """
    Time-series of mean RSRP / RSRQ.

    Bucket size:
    - **24h** → 1-hour buckets
    - **week** / **month** → 1-day buckets
    """
    try:
        trunc_unit = TRENDS_TRUNC.get(period or "24h", "hour")

        q = apply_mobile_filters(
            db.query(DeviceReading),
            operator, network_type, period, source, lat, lon, radius_km,
        )

        bucket = func.date_trunc(trunc_unit, DeviceReading.timestamp).label("bucket")

        rows = (
            q.with_entities(
                bucket,
                func.avg(DeviceReading.rsrp).label("mean_rsrp"),
                func.avg(DeviceReading.rsrq).label("mean_rsrq"),
            )
            .group_by(bucket)
            .order_by(bucket)
            .all()
        )

        return TrendsResponse(
            points=[
                TrendPoint(
                    timestamp=r.bucket.isoformat() + "Z",
                    mean_rsrp=round(r.mean_rsrp, 2) if r.mean_rsrp is not None else None,
                    mean_rsrq=round(r.mean_rsrq, 2) if r.mean_rsrq is not None else None,
                )
                for r in rows
            ]
        )
    except Exception as e:
        logger.error(f"Trends error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/api/mobile/operators/unique", response_model=FiltersResponse)
def mobile_filters(
    db: Session = Depends(get_db),
    _: str = Depends(verify_api_key),
):
    """Return all unique non-null operator values stored in the database."""
    try:
        rows = (
            db.query(DeviceReading.operator)
            .filter(DeviceReading.operator.isnot(None))
            .distinct()
            .order_by(DeviceReading.operator)
            .all()
        )
        return FiltersResponse(operators=[r.operator for r in rows if r.operator])
    except Exception as e:
        logger.error(f"Operators error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------
def _parse_timestamp(raw: Optional[str]) -> datetime:
    if not raw:
        return datetime.utcnow()
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid timestamp format. Use ISO-8601.")


def _build_reading(data: NetworkDataRequest, ts: datetime) -> DeviceReading:
    return DeviceReading(
        source=data.source,
        timestamp=ts,
        latitude=data.latitude,
        longitude=data.longitude,
        altitude=data.altitude,
        level=data.level,
        asu=data.asu,
        rsrp=data.rsrp,
        rssi=data.rssi,
        rsrq=data.rsrq,
        network_type=data.networkType,
        operator=data.operator,
        cell_id=data.cellId,
        physical_cell_id=data.physicalCellId,
        tracking_area_code=data.trackingAreaCode,
        country=data.country,
        city=data.city,
    )


def _reading_to_response(r: DeviceReading) -> NetworkDataResponse:
    return NetworkDataResponse(
        id=r.id,
        source=r.source,
        timestamp=r.timestamp.isoformat(),
        latitude=r.latitude,
        longitude=r.longitude,
        altitude=r.altitude,
        level=r.level,
        asu=r.asu,
        rsrp=r.rsrp,
        rssi=r.rssi,
        rsrq=r.rsrq,
        network_type=r.network_type,
        operator=r.operator,
        cell_id=r.cell_id,
        physical_cell_id=r.physical_cell_id,
        tracking_area_code=r.tracking_area_code,
        country=r.country,
        city=r.city,
        created_at=r.created_at.isoformat(),
    )


def _device_reading_to_response(r: DeviceReading) -> DeviceReadingResponse:
    return DeviceReadingResponse(
        device_id=r.source,
        timestamp=r.timestamp.isoformat(),
        latitude=r.latitude,
        longitude=r.longitude,
        altitude=r.altitude,
        level=r.level,
        asu=r.asu,
        rsrp=r.rsrp,
        rssi=r.rssi,
        rsrq=r.rsrq,
        network_type=r.network_type,
        operator=r.operator,
        cell_id=r.cell_id,
        physical_cell_id=r.physical_cell_id,
        tracking_area_code=r.tracking_area_code,
        country=r.country,
        city=r.city,
        created_at=r.created_at.isoformat(),
    )
# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
