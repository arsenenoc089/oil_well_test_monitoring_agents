from pydantic import BaseModel

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