from agents import function_tool
from tools.output import WellTestContext, ZonalTestMemory, WellTestInterpretation, AnomalyInsights
from pathlib import Path
from loguru import logger
from typing import List, Literal
from datetime import datetime
import csv
from pydantic import BaseModel

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
    
class WellTestData(BaseModel):
    Date: datetime
    WellName: str
    Anomaly: bool
    AnomalyType: str
    WTLIQ: float
    WTOil: float
    WTTHP: float
    WTWCT: float
    Z1Status: Literal["Open", "Closed"]
    Z2Status: Literal["Open", "Closed"]
    Z3Status: Literal["Open", "Closed"]
    Z1BHP: float
    Z2BHP: float
    Z3BHP: float


def load_well_memory_data(filepath: str) -> List[WellTestData]:
    well_data = []
    
    with open(filepath, 'r') as file:
        # Skip the header row
        next(file)
        
        # Create CSV reader
        csv_reader = csv.reader(file)
        
        for row in csv_reader:
            # Clean up whitespace and create dictionary
            cleaned_row = [item.strip() for item in row]
            data_dict = {
                'Date': datetime.strptime(cleaned_row[0], '%Y-%m-%d'),
                'WellName': cleaned_row[1],
                'Anomaly': cleaned_row[2].lower() == 'true',
                'AnomalyType': cleaned_row[3],
                'WTLIQ': float(cleaned_row[4]),
                'WTOil': float(cleaned_row[5]),
                'WTTHP': float(cleaned_row[6]),
                'WTWCT': float(cleaned_row[7]),
                'Z1Status': cleaned_row[8],
                'Z2Status': cleaned_row[9],
                'Z3Status': cleaned_row[10],
                'Z1BHP': float(cleaned_row[11]),
                'Z2BHP': float(cleaned_row[12]),
                'Z3BHP': float(cleaned_row[13])
            }
            well_data.append(WellTestData(**data_dict))
    
    return well_data