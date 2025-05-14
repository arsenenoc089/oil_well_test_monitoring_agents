import asyncio
from pydantic import BaseModel
from agents import Agent, Runner, function_tool
from loguru import logger
from agents.utils import utils

class Weather(BaseModel):
    city: str
    temperature_range: str
    conditions: str


@function_tool
def get_weather(city: str) -> Weather:
    print("[debug] get_weather called")
    return Weather(city=city, temperature_range="14-20C", conditions="Sunny with wind.")


agent = Agent(
    name="Hello world",
    instructions="You are a helpful agent.",
    tools=[get_weather],
)


async def main():
    logger.info("Starting to load data")
    df = utils.load_data('data/RMO_Agentic AI_train_test.xlsx')
    # The weather in Tokyo is sunny.


if __name__ == "__main__":
    asyncio.run(main())