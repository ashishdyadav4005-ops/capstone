"""FastAPI Backend Application Entrypoint for BDS-39 Dynamic Pricing Engine."""

from __future__ import annotations

import time
import uuid

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src import __version__
from src.api.routers import audit, auth, governance, health, monitoring, pricing, recommendations
from src.common.config import get_config
from src.common.logger import get_logger, setup_logging
from src.monitoring.metrics import PROMETHEUS_CONTENT_TYPE, get_metrics_registry

config = get_config()
setup_logging(log_level=config.logging.level, json_format=(config.logging.format == "json"))
logger = get_logger("src.api.main")

app = FastAPI(
    title="BDS-39 Counterfactual Dynamic Pricing API",
    description=(
        "Production-grade decision-support REST API for Counterfactual Dynamic Pricing "
        "with Double Machine Learning (DML) elasticities, SciPy SLSQP optimization, "
        "and fairness/margin/capacity guardrails."
    ),
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
)

# 1. Configure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.server.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 2. Request ID & Logging Middleware
@app.middleware("http")
async def request_correlation_middleware(request: Request, call_next):
    """Inject or propagate X-Request-ID and track request latency."""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    start_time = time.perf_counter()

    # Process request
    response = await call_next(request)

    process_time_ms = (time.perf_counter() - start_time) * 1000.0
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = f"{process_time_ms:.2f}"

    # Track Prometheus metrics
    try:
        registry = get_metrics_registry()
        registry.pricing_requests_total.inc(
            endpoint=request.url.path,
            status=str(response.status_code),
            role="api",
        )
    except Exception:
        pass

    logger.info(
        f"{request.method} {request.url.path} completed in {process_time_ms:.2f}ms with status {response.status_code}",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "latency_ms": round(process_time_ms, 2),
        },
    )
    return response


# 3. Global Exception Handlers
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle validation and business rule violations."""
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc), "error_type": "ValueError"},
    )


@app.exception_handler(KeyError)
async def key_error_handler(request: Request, exc: KeyError):
    """Handle missing entity lookups."""
    return JSONResponse(
        status_code=404,
        content={"detail": str(exc), "error_type": "KeyError"},
    )


# 4. Mount Routers
app.include_router(auth.router)
app.include_router(health.router)
app.include_router(pricing.router)
app.include_router(recommendations.router)
app.include_router(audit.router)
app.include_router(governance.router)
app.include_router(monitoring.router)


@app.get("/metrics")
def root_metrics() -> Response:
    """Root-level Prometheus metrics exposition endpoint."""
    registry = get_metrics_registry()
    content = registry.generate_prometheus_text()
    return Response(content=content, media_type=PROMETHEUS_CONTENT_TYPE)


@app.get("/")
def read_root():
    """Root endpoint providing system status and version info."""
    return {
        "service": "BDS-39 Counterfactual Dynamic Pricing API",
        "version": __version__,
        "status": "online",
        "environment": config.app.environment,
        "docs": "/docs",
    }
