from pydantic import BaseModel
from typing import Literal

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
    zone1_status: Literal["Open", "Closed"]
    zone2_status: Literal["Open", "Closed"]
    zone3_status: Literal["Open", "Closed"]
    log_diff_oil: float
    log_diff_liq: float
    log_diff_thp: float
    log_diff_wct: float
    log_diff_z1bhp: float
    log_diff_z2bhp: float
    log_diff_z3bhp: float

class ZonalTestMemory(BaseModel):
    Date: str
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

class WellTestInterpretation(BaseModel):
    Date: str
    WellName: str
    ZonalConfiguration: str
    Interpretation: str
    EngineerAction: str
    InsightsSummary: str
    """A short 2-3 sentence of your findings after analysing the data looking for anomalies"""

class AnomalyInsights(BaseModel):
    Short_summary: str
    """A short 2-3 sentence of your findings after analysing the data looking for anomalies"""