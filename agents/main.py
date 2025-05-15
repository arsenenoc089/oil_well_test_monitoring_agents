import asyncio
from pydantic import BaseModel
from agents import Agent, Runner, function_tool, trace
from loguru import logger
from utils import utils
from tools.prompts import make_prompt, CONTEXT_PROMPT
from dotenv import load_dotenv


#Load environment variables
load_dotenv()

#Prompts
ANOMALY_DETECTOR_PROMPT, MEMORY_SAVER_PROMPT, INTERPRETATOR_PROMPT = make_prompt(threshold=0.1)

#Agents output types
class WellTestContext(BaseModel):
    Date: str
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

class ZonalTestMemory(BaseModel):
    Date: str
    WellName: str
    Anomaly: bool
    AnomalyType: str
    WTLIQ: float
    WTOil: float
    WTTHP: float
    WTWCT: float
    Z1Status: str
    Z2Status: str
    Z3Status: str
    Z1BHP: float
    Z2BHP: float
    Z3BHP: float

class WellTestInterpretation(BaseModel):
    Date: str
    WellName: str
    ZonalConfiguration: str
    Interpretation: str
    EngineerAction: str
    InsightsSummary: str
    """A short 2-3 sentence of your findings after analysing the data looking for anomalies"""

class AnomalyInsights(BaseModel):
    short_summary: str
    """A short 2-3 sentence of your findings after analysing the data looking for anomalies"""


# Tools
@function_tool
def save_test_memory(welldata: WellTestContext, file_path: str, 
                    Anomaly: bool,
                    AnomalyType: str,
                    Z1Status: str,
                    Z2Status: str,
                    Z3Status: str) -> None:
    """
        Save the well test data record as a comma separated line in a text file.
        Args:
            Welldata: The well test data record to save.
            file_path: The path to the file where the data will be saved.
    """

    file_path =  'agents/mem.txt'
    logger.info(f"Saving well test data in memory")
    fields = [
        welldata.Date,
        welldata.WellName,
        Anomaly,
        AnomalyType,
        welldata.WTLIQ,
        welldata.WTOil,
        welldata.WTTHP,
        welldata.WTWCT,
        Z1Status,
        Z2Status,
        Z3Status,
        welldata.Z1BHP,
        welldata.Z2BHP,
        welldata.Z3BHP        
    ]
    
    with open(file_path, 'a') as f:
        f.write(', '.join(map(str, fields)) + '\n')


# Agent team
anomaly_detection_agent = Agent(
    name="wt_anomaly_detector",
    instructions=ANOMALY_DETECTOR_PROMPT,
    output_type=AnomalyInsights,
)

MEMORY_SAVER_AGENT = Agent(
    name="WT_memory_saver",
    instructions=MEMORY_SAVER_PROMPT,
    output_type=ZonalTestMemory,
    tools=[save_test_memory],
)

INTERPRETATOR_AGENT = Agent(
    name="wt_interpretator",
    instructions=INTERPRETATOR_PROMPT,
    output_type=WellTestInterpretation,
)


#Function to run the agents
async def main():
    logger.info("Starting to load data")
    df = utils.load_transform_welltest_data('data/RMO_Agentic AI_train_test.xlsx')
    df = df[['Date', 'WellName', 'WTLIQ', 'WTOil', 'WTTHP', 'WTWCT', 'Z1BHP',
            'Z2BHP', 'Z3BHP', 'mean_bhp', 'log_diff_z1bhp_meanbhp',
            'log_diff_z2bhp_meanbhp', 'log_diff_z3bhp_meanbhp', 'log_diff_oil',
            'log_diff_liq', 'log_diff_thp', 'log_diff_wct']].copy()
    df['Date'] = df['Date'].astype(str)
    
    df = df.iloc[13:]

    df_iterator = df.iterrows()
    next(df_iterator)
    idx, serie = next(df_iterator)
    df = serie.to_dict()

    #Define the context
    context = CONTEXT_PROMPT

    well_test_input = WellTestContext(**df)
    logger.info(f"well test input {well_test_input}")

    # Ensure the entire workflow is a single trace
    with trace("Deterministic story flow"):

        # Run the anomaly detection agent
        result_anomaly = await Runner.run(anomaly_detection_agent, input=f' Here are the well test data {well_test_input.model_dump()}' , context=context)
        print(result_anomaly.final_output)

        # Run the memory saver agent
        result_memory = await Runner.run(MEMORY_SAVER_AGENT, input=f' Here are the well test data {well_test_input.model_dump()} and this is what the anomaly analysis result is {result_anomaly.final_output}' , context=context)
        print(result_memory.final_output)

        # Run the interpretator agent
        result_interpretator = await Runner.run(INTERPRETATOR_AGENT, input=f' Here are the well test data {well_test_input.model_dump()} and this is what the anomaly analysis result is {result_anomaly.final_output}' , context=context)
        print(result_memory.final_output)

if __name__ == "__main__":
    asyncio.run(main())