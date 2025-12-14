from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.routes.dp_routes import train_model_route, predict_prices_route
from app.routes.recommendation_routes import get_cart_recommendations
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger


def daily_job():
    train_model_route("train")
    print("Cron Report: Model Trained ")
    predict_prices_route()
    print("Cron Report: Prices Predicted")
    get_cart_recommendations()
    print("Cron Report: Cart Recommendations Updated")


scheduler = BackgroundScheduler()
trigger = CronTrigger(hour=2, minute=0)
scheduler.add_job(daily_job, trigger=trigger)


@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler.add_job(daily_job, IntervalTrigger(minutes=0.1))
    scheduler.start()
    yield
    scheduler.shutdown()
