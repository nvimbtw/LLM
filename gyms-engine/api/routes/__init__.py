from fastapi import APIRouter

from api.routes.gym import router as gym_router

ROUTERS = [gym_router]
