import asyncio
from pydantic import BaseModel
from agents import Agent, Runner, function_tool
from loguru import logger
from utils import utils
from dotenv import load_dotenv

load_dotenv()
#Prompts
def make_prompt(threshold=0.1):
    ANOMALY_DETECTOR_PROMPT = (f'You are anomaly detection agent for well test data in an Oil and Gas company.'
        '----------------------'
        'You will be given some well test data (one at a time) for a specific well. '
        'Look for anomalies in the oil rate, liquid rate or WCT (water cut). The data has some log of changes already computed '
       f'log_diff_oil, log_diff_liq and log_diff_wct = ln(value/previous value). The thresold is + or - {threshold}. '
        '----------------------'
        'Anomalies are defined as below:'
        '1 - Zonal test: you can detect a zonal test event (anomaly) if only two of the log_diff_z1bhp_meanbhp, log_diff_z2bhp_meanbhp'
        f', log_diff_z3bhp_meanbhp are greater than {threshold} at any given time and the third one is below -{threshold}.'
    )

    return ANOMALY_DETECTOR_PROMPT

ANOMALY_DETECTOR_PROMPT = make_prompt(threshold=0.1)

#Define the input schema
class WellTestContext(BaseModel):
    WellName: str
    WTLIQ: float
    WTOil: float
    WTTHP: float
    WTWCT: float
    Z1BHP: float
    Z2BHP: float
    Z3BHP: float
    mean_bhp: float
    log_diff_z1bhp_meanbhp: float
    log_diff_z2bhp_meanbhp: float
    log_diff_z3bhp_meanbhp: float
    log_diff_oil: float
    log_diff_liq: float
    log_diff_thp: float
    log_diff_wct: float

class AnomalyInsights(BaseModel):
    short_summary: str
    """A short 2-3 sentence of your findings after analysing the data looking for anomalies"""

    follow_up_questions: list[str]
    """Suggested follow-up questions for analysis."""

def get_test(df: dict) -> WellTestContext:
    # Convert the input dictionary to a WellTestInput object
    well_test_input = WellTestContext(**df)
    return well_test_input


anomaly_detection_agent = Agent(
    name="wt_anomaly_detector",
    instructions=ANOMALY_DETECTOR_PROMPT,
    output_type=AnomalyInsights,

)


async def main():
    logger.info("Starting to load data")
    df = utils.load_transform_welltest_data('data/RMO_Agentic AI_train_test.xlsx')
    df = df[['Date', 'WellName', 'WTLIQ', 'WTOil', 'WTTHP', 'WTWCT', 'Z1BHP',
            'Z2BHP', 'Z3BHP', 'mean_bhp', 'log_diff_z1bhp_meanbhp',
            'log_diff_z2bhp_meanbhp', 'log_diff_z3bhp_meanbhp', 'log_diff_oil',
            'log_diff_liq', 'log_diff_thp', 'log_diff_wct']].copy()
    
    df = df.iloc[9:]

    df_iterator = df.iterrows()
    next(df_iterator)
    idx, serie = next(df_iterator)
    df = serie.to_dict()
    context = get_test(df)

    well_test_input = WellTestContext(**df)
    logger.info(f"well test input {well_test_input}")
    result = await Runner.run(anomaly_detection_agent, input=f' Here are the well test data {well_test_input.model_dump()}' , context=context)
    print(result.final_output)



if __name__ == "__main__":
    asyncio.run(main())