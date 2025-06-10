import asyncio
from pydantic import BaseModel
from agents import Agent, Runner, trace
from openai import OpenAI
from loguru import logger
from utils import utils
from tools.prompts import make_prompt, CONTEXT_PROMPT
from tools.output import WellTestContext, ZonalTestMemory, WellTestInterpretation, AnomalyInsights
from tools.tools import save_test_memory, load_well_memory_data
from dotenv import load_dotenv
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
import streamlit_shadcn_ui as ui
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os


base_path = os.path.dirname(__file__)
mem_file_path = os.path.join(base_path, "mem.txt")
st.set_page_config(layout="wide")

#Load environment variables
try:
    # Access nested values
    openai_key = st.secrets["api_keys"]["openai"]
    os.environ["OPENAI_API_KEY"] = openai_key
except:
    load_dotenv()


#Prompts
ANOMALY_DETECTOR_PROMPT, MEMORY_SAVER_PROMPT, INTERPRETATOR_PROMPT = make_prompt(threshold=0.1, file_path = mem_file_path)


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

    with st.sidebar:
        st.divider()
        st.markdown('Select a well below:')
        chosen_well = ui.select(options=['cheetah-90', 'cheetah-20', 'cheetah-10'], key="well_select")

        if chosen_well:
            logger.info("Starting to load data")
            df_raw = utils.load_transform_welltest_data('data/RMO_Agentic AI_train_test.xlsx', well_name=chosen_well, threshold=0.1)
            df = df_raw[['Date', 'WellName', 'WTLIQ', 'WTOil', 'WTTHP', 'WTWCT', 'Z1BHP',
                    'Z2BHP', 'Z3BHP', 'mean_bhp', 'log_diff_oil',
                    'log_diff_liq', 'log_diff_thp', 'log_diff_wct', 'log_diff_z1bhp',
                    'log_diff_z2bhp', 'log_diff_z3bhp', 'zone1_status', 'zone2_status',
                    'zone3_status']]
        
            df['Date'] = df['Date'].astype(str)
            #Dates list
            dates = df['Date'].unique().tolist()
            dates.sort()


        logger.info(f"Data loaded for well {chosen_well} with {len(df)} rows")
        st.sidebar.success(f"Data loaded for well {chosen_well} with {len(df)} rows")

        st.markdown('Select a well test date below:')
        selected_date = ui.select(options=dates)

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

        value = ui.tabs(options=['Well test chart', 'Well test datatable'], default_value='PyGWalker', key="kanaries")

        if value == "Well test chart":
            #Charts
            # Option 1: Single plot with all lines
            fig = go.Figure()

            # Add each column as a separate line
            for column in ['WTLIQ', 'WTOil', 'WTWCT', 'Z1BHP','Z2BHP', 'Z3BHP']:  # Skip the Date column
                fig.add_trace(go.Scatter(
                    x=df['Date'],
                    y=df[column],
                    mode='lines',
                    name=column
                ))

            fig.update_layout(
                title="Well test data: WT OIL, WT WCT, WT Liq, Z1 BHP, Z2 BHP & Z3 BHP",
                xaxis_title="Date",
                yaxis_title="Value",
                legend_title="Columns",
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)
        elif value == "Well test datatable":
            # Build grid options
            gb = GridOptionsBuilder.from_dataframe(df_aggrid)

            # Cell styling for the 'Value' column based on threshold
            cellstyle_jscode = JsCode("""
            function(params) {
                if (params.value < 5000) {
                    return {
                        'color': 'white',
                        'backgroundColor': 'salmon'
                    }
                } else {
                    return {
                        'color': 'black',
                        'backgroundColor': 'white'
                    }
                }
            };
            """)

            # Apply conditional formatting to multiple columns
            columns_to_style = ['Z1BHP','Z2BHP', 'Z3BHP']
            for col in columns_to_style:
                gb.configure_column(col, cellStyle=cellstyle_jscode)
            # Build the grid options
            grid_options = gb.build()
            grid_return = AgGrid(df_aggrid, gridOptions=grid_options,editable=True, allow_unsafe_jscode=True, width = 800, height=300, fit_columns_on_grid_load=True)


    # Ensure the entire workflow is a single trace
    agentic_ai_button = ui.button("Run AI Agents", key="agentic_ai")

    #add image of agentic flow
    image_path = os.path.join(base_path, "agentic_ai.png")
    st.image(image_path)
    st.divider()


    if agentic_ai_button:
        # Run the agents in a deterministic story flow
        df = df.tail(1)
        df_iterator = df.iterrows()
        idx, serie = next(df_iterator)
        df = serie.to_dict()

        #Define the context
        context = CONTEXT_PROMPT

        #load memory data\
        try:
            logger.info(f"Loading memory data from {mem_file_path}")
            memory_data = load_well_memory_data(mem_file_path)
            logger.info(f"Memory data loaded successfully")
        except Exception as e:
            logger.error(f"Error loading memory data: {e}")
            memory_data = []

        well_test_input = WellTestContext(**df)
        logger.info(f"well test input {well_test_input}")
        with trace("Deterministic story flow"):

            # Run the anomaly detection agent
            logger.info(f"The anomaly detector agent is at work...")
            result_anomaly = await Runner.run(anomaly_detection_agent, input=f' Here are the well test data {well_test_input.model_dump()}' , context=context)
            logger.info(f"The anomaly detector agent has completed its work...")

            st.write("Agentic AI workflow has been triggered - See the results in the card below")
            with ui.card(key="card1"):
                ui.element("h3", children=["Anomaly detector agent:"], className="font-bold ")
                with ui.element("div", className="flex justify-between bg-stone-200 rounded-sm "):
                    ui.element("span", children=[str(result_anomaly.final_output.Short_summary)], className="font-Medium")

            # Run the memory saver agent
            logger.info(f"Now the memory savor agent is at work...")
            result_memory = await Runner.run(MEMORY_SAVER_AGENT, input=f' Here are the well test data {well_test_input.model_dump()} and this is what the anomaly analysis result is {result_anomaly.final_output}' , context=context)
            logger.info(f"The memory savor agent has completed its work...")

            # Run the interpretator agent
            logger.info(f"Now the insights interpreter agent is at work...")
            result_interpretator = await Runner.run(INTERPRETATOR_AGENT, 
                                                    input=f'Here are the well test data {well_test_input.model_dump()} and this is what the anomaly analysis result for this well test is {result_anomaly.final_output} - The memory data has past welltest {memory_data}', context=context)
            logger.info(f"The insights interpreter agent has completed its work...")
            
            with ui.card(key="card2"):
                ui.element("h3", children=["Insights Interpretation agent:"], className="font-bold ")
                with ui.element("div", className="flex bg-stone-200 rounded-sm"):
                    ui.element("h4", children=["Zonal Config: "], className="font-semibold")
                    ui.element("span", children=[str(result_interpretator.final_output.ZonalConfiguration)], className="font-Medium")
                with ui.element("div", className="flex bg-stone-200 rounded-sm"):
                    ui.element("h4", children=["Interpretation: "], className="font-semibold")
                    ui.element("span", children=[str(result_interpretator.final_output.Interpretation)], className="font-Medium inline-block")
                with ui.element("div", className="flex bg-stone-200 rounded-sm"):
                    ui.element("h4", children=["EngineerAction: "], className="font-semibold")
                    ui.element("span", children=[str(result_interpretator.final_output.EngineerAction)], className="font-Medium")
                with ui.element("div", className="flex bg-stone-200 rounded-sm"):
                    ui.element("h4", children=["InsightsSummary: "], className="font-semibold")
                    ui.element("span", children=[str(result_interpretator.final_output.InsightsSummary)], className="font-Medium")
            
            with open(mem_file_path, 'r') as f:
                logger.info(f"This has been saved in the memory file: {f.read()}")
            
            
  
if __name__ == "__main__":
    asyncio.run(main())