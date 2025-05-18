import asyncio
from pydantic import BaseModel
from agents import Agent, Runner, trace
from loguru import logger
from utils import utils
from tools.prompts import make_prompt, CONTEXT_PROMPT
from tools.output import WellTestContext, ZonalTestMemory, WellTestInterpretation, AnomalyInsights
from tools.tools import save_test_memory
from dotenv import load_dotenv
import streamlit as st
from st_aggrid import AgGrid
import pandas as pd

st.set_page_config(layout="wide")

#Load environment variables
load_dotenv()

#Prompts
ANOMALY_DETECTOR_PROMPT, MEMORY_SAVER_PROMPT, INTERPRETATOR_PROMPT = make_prompt(threshold=0.1)


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

    #Sidebar
    with st.sidebar:
        st.title("WellWatch")
        st.subheader("Oil Well Test Monitoring with Agentic AI")
        st.write("This is a demo of Colab work on Agentic AI for oil well test monitoring.")
        st.write("The platform uses a combination of statistics and RE expertise to enable the AI agents to detect anomalies in well test data and provide insights.")

    logger.info("Starting to load data")
    df = utils.load_transform_welltest_data('data/RMO_Agentic AI_train_test.xlsx')
    df = df[['Date', 'WellName', 'WTLIQ', 'WTOil', 'WTTHP', 'WTWCT', 'Z1BHP',
            'Z2BHP', 'Z3BHP', 'mean_bhp', 'log_diff_z1bhp_meanbhp',
            'log_diff_z2bhp_meanbhp', 'log_diff_z3bhp_meanbhp', 'log_diff_oil',
            'log_diff_liq', 'log_diff_thp', 'log_diff_wct']].copy()
    
    df['Date'] = df['Date'].astype(str)

    #Dates list
    dates = df['Date'].unique().tolist()
    dates.sort()
    #Select date
    selected_date = st.sidebar.selectbox("Select a well test date", dates)
    
    if selected_date:
        #select the data up until the selected date
        logger.info(f"Filtering data for date {selected_date}")
        # Filter the dataframe to include only rows with the selected dateand all rows before it
        df = df.loc[df['Date'] <= selected_date]
        logger.info(f"Done filtering data for date {selected_date}")
        df_aggrid = df.copy().round(0)
        df_aggrid['Date'] = pd.to_datetime(df_aggrid['Date'])
        df_aggrid = df_aggrid[['Date', 'WellName', 'WTLIQ', 'WTOil', 'WTTHP', 'WTWCT', 'Z1BHP',
            'Z2BHP', 'Z3BHP']]
        grid_return = AgGrid(df_aggrid, editable=True, width = 800, height=300, fit_columns_on_grid_load=True)

    
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
    button = st.button("Run well test AI Agents workflow")
    if button:
        # Run the agents in a deterministic story flow
        with trace("Deterministic story flow"):

            # Run the anomaly detection agent
            result_anomaly = await Runner.run(anomaly_detection_agent, input=f' Here are the well test data {well_test_input.model_dump()}' , context=context)
            print(result_anomaly.final_output)

            # Run the memory saver agent
            result_memory = await Runner.run(MEMORY_SAVER_AGENT, input=f' Here are the well test data {well_test_input.model_dump()} and this is what the anomaly analysis result is {result_anomaly.final_output}' , context=context)
            print(result_memory.final_output)

            # Run the interpretator agent
            result_interpretator = await Runner.run(INTERPRETATOR_AGENT, input=f' Here are the well test data {well_test_input.model_dump()} and this is what the anomaly analysis result is {result_anomaly.final_output}' , context=context)
            print(result_interpretator.final_output)

if __name__ == "__main__":
    asyncio.run(main())