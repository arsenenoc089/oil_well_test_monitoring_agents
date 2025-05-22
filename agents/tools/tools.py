from agents import function_tool
from tools.output import WellTestContext, ZonalTestMemory, WellTestInterpretation, AnomalyInsights
from pathlib import Path
from loguru import logger

this_dir = Path(__file__).parent
file_path = this_dir.parent / "mem.txt"

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

    #file_path =  'agents/mem.txt'
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