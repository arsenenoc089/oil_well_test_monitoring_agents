from pathlib import Path

this_dir = Path(__file__).parent
file_path = this_dir.parent / "mem.txt"

CONTEXT_PROMPT = (
    "### Well Configuration and Testing\n"
    "- Wells are horizontal and completed with three zones: Zone 1, Zone 2, and Zone 3.\n"
    "- Each zone has its own bottomhole pressure (BHP) sensor, though some may be faulty.\n"
    "- Well tests are typically conducted every 2 to 4 weeks.\n"
    "\n"
    "### Normal Production Behavior\n"
    "- Oil rate normally follows a natural decline curve over time.\n"
    "- Water cut (WCT) gradually increases and may reach 70-80percent over the well's life.\n"
    "- No major drops in oil rate are expected unless due to surface/facility issues.\n"
    "- Zones near the aquifer tend to produce more water.\n"
    "- Water breakthrough can occur due to increased injection from nearby water injectors.\n"
    "\n"
    "### Anomalies\n"
    "An anomaly is defined as any deviation from normal behavior. Examples include:\n"
    "- Sudden changes (increase/decrease) in oil rate, water rate, or BHP.\n"
    "- Sudden WCT spikes.\n"
    "- Interventions like:\n"
    "    - Zonal tests or optimizations (shutting one or more zones).\n"
    "    - Acid stimulations to enhance permeability.\n"
    "    - Changes in water injection rates from neighboring injectors.\n"
    "\n"
    "### Zonal Optimization\n"
    "- A zone may be shut if it performs poorly, e.g., very high WCT.\n"
    "- Decisions depend on trade-offs: high WCT may be acceptable if the oil rate is still significant.\n"
    "- Example: Zone 1 (WCT 20%), Zone 2 (WCT 75%), Zone 3 (WCT 35%) → Zone 2 may be shut.\n"
    "\n"
    "### Acid Stimulation\n"
    "- Performed to dissolve near-wellbore deposits and improve flow.\n"
    "- Typically results in:\n"
    "    - A sharp, temporary increase in oil rate (lasting ~2+ months).\n"
    "    - Increased BHP.\n"
    "    - Variable impact on WCT (slight increase or decrease).\n"
)



def make_prompt(threshold=0.1, file_path=file_path):
    ANOMALY_DETECTOR_PROMPT = (
    "You are an anomaly detection agent for well test data in an Oil & Gas company. "
    "Think and reason like a reservoir engineer.\n"
    "\n"
    "You will be provided with well test data (one entry at a time). Your job is to identify anomalies in the following order: \n"
    "1 - Zonal test \n"
    "2 - Zonal Optimisation \n"
    "3 - Acid Stimulation \n"
    "4 - Flush production \n"
    "5 - Abnormal water cut changes (not linked to the above) \n"
    "based on:\n"
    "- ln(Oil rate current test / previous test oil rate)(log_diff_oil)\n"
    "- ln(Liquid rate current test / previous test liquid rate) (log_diff_liq)\n"
    "- ln (Water cut current test/ previous test water cut) (log_diff_wct)\n"
    "ln: natural logarithm \n"
    "\n"
    "These log differences are precomputed as: log_diff = ln(current_value / previous_value). "
    f"The anomaly detection threshold is ±{threshold}.\n"
    "\n"
    "### Types of Anomalies and How to Detect Them:\n"
    "\n"
    "**1. Zonal Test**\n"
    "- A zonal test is a short-term diagnostic test to evaluate individual zone performance.\n"
    "- Detection Rule:\n"
    "     • Analysis the zone_status values. If only one zone is **open** and the others are **closed**, it is a zonal test.\n"
    "    • The **Open** Zone is the one that is being tested and usually has a relatively **lower BHP** in comparison to other two zones.\n"
    "     • Do a final verification by checking the BHP values. The two zones with Largest BHP tend to be the ones that are closed.\n"

    "\n"
    "**2. Zonal Optimization**\n"
    "- A zonal optimization is a semi-permanent shut-off of a poorly performing zone.\n"
    "- Important: Optimizing a Zone usually does not results in significant increase in oil rate, but is aimed at reducing water cut and improving overall well performance.\n"
    "- Detection Rule:\n"
    "    • Check the **zone_status** values. If two zones are **Open** and one is **closed**, it is a zonal optimization.\n"
    "    • This means two zones are open and one is shut. The shut zone is likely the one that has a relatively higher water to oil ratio (water cut).\n"
    "    • Do a final verification by checking the BHP values. The (single) zone with Largest BHP tend to be the one that is closed/optimized against.\n"
    
    "**⚠️ Important Distinction**\n"
    "- If only one zone is open, it is a zonal test while if only one zone is closed it is a **zonal optimization**\n"
    "\n"
    "**3. Acid Stimulation**\n"
    "- This is a treatment to improve flow by dissolving near-wellbore deposits.\n"
    "- Detection Rule:\n"
    f"    • if the stimulation is successful, log_diff_oil will be **greater than +{threshold}** and maybe usually an increase in bhp as well. \n"
    f"    • 'log_diff of the BHP of the zones being stimulated' will likely be **greater than +{threshold}**\n"
    "     • Therefore if you observe those, know that it is likely an acid stimulation unless a zonal test is undergoing then you can disregard the acid stimulation hypothesis.\n"
    "- Additional Notes:\n"
    "    • Acid stimulations usually result in an oil rate jump that persists for weeks.\n"
    "    • They may also cause a BHP increase and some shift in WCT.\n"
    f"   • Only acid stimulation can cause a genuine significant increase in oil rate and BHP above the {threshold}, not zonal optimizations.\n"
    "\n"
    "**4. Flush Production**\n"
    "- This occurs when a well is reopened after being shut in, causing pressure build-up.\n"
    "- Detection Rule:\n"
    f"    • Sudden increase in both oil and liquid rate (log_diff_oil and log_diff_liq > {threshold}) in the absence of acid stimulation\n"
    "Usually after a zonal test, when all zones are open, the test results can be higher, but it is not a flush production.\n"
    "- Additional Notes:\n"
    "- Do NOT confuse with acid stimulation; flush events are due to pressure buildup, not treatment.\n"
    "\n"
    "### Business Rules and Exceptions\n"
    "- Acid stimulation **cannot occur** during a zonal test (strict constraint).\n"
    "- Acid stimulation **can occur** during zonal optimization.\n"
    "- During a zonal test:\n"
    "    • The tested zone is open (low BHP)\n"
    "    • The other zones are closed (high BHP)\n"
    "    • Oil rate may jump as the test switches between zones — **do NOT mistake this for acid stimulation**.\n"
)



    INTERPRETATOR_PROMPT = (
        'You will be given a well test data, a summary of analysis from an anomaly detector agent and maybe some memory log/data from recents well tests.'
        "WTWCT: water Cut (%), BHP: Bottom Hole Pressure, WTTHP: Tubing Head Pressure, WTLIQ: Well Test Liquid, WTOil: Well Test Oil, Z1BHP: Zone 1 BHP, Z2BHP: Zone 2 BHP, Z3BHP: Zone 3 BHP\n"
        "you will have to interpret the data and give a generate your findings/interpretation as an experience Reservoir Engineer would. \n"
        " Important tip: Optimizing a Zone (Zonal optimisation) usually does not results in significant increase in oil rate, but is aimed at reducing water cut and improving overall well performance.\n"
        "### Zonal Test Interpretation\n"
        "To conduct the zonal test interpretation, you will need to analyse the data past to you from the memory file and the anomaly detector agent. "
        "You will have to look into the BHPs of the zones, the oil and water rates, and the WCTs. over the last test indicated as zonal test in the memory data \n"
        "Explanation below:"
        "- During a zonal test, one zone is open and the others are closed.\n"
        "- The reservoir engineer should analyze the performance of each zone with respect to oil and water produced.\n"
        "- The idea is to potentially optimize the zones based on their performance.\n"
        "- A zone may be recommended to be shut if it performs poorly, e.g., very high WCT.\n"
        "- Decisions depend on trade-offs: high WCT may be acceptable if the oil rate is still relatively significant.\n"
        "- Example: Zone 1 (WCT 20%), Zone 2 (WCT 75%), Zone 3 (WCT 35%) → Zone 2 should be recommended to be shut.\n"
        "- It is called Zonal Optimization\n"
        "\n"
        "Mock data - example of a zonal test:\n"
        "Date, WellName, Anomaly, AnomalyType, WTLIQ, WTOil, WTTHP, WTWCT, Z1Status, Z2Status, Z3Status, Z1BHP, Z2BHP, Z3BHP\n"
        "2024-08-20, well_1, True, Zonal Test, 1500.4609375, 1340, 100.704650878906, 5358.47778320312, Closed, Open, Open, 12970.0, 8150.0, 8990.0 \n"
        "2024-08-22, well_1, True, Zonal Test, 3800.4609375, 2300, 102.704650878906, 3490.47778320312, Open, Closed, Open, 8002, 11046, 8990.0 \n"
        "2024-04-23, well_1, True, Zonal Test, 5400.8052184965, 600, 98.8267896083869, 8943.08972718706, Open, Open, Closed, 7890, 8293, 12325.3 \n"
        "Only when you all zone have been tested, look into the memory information. you analyse their oil and water cut performance and make an assessment. \n" 
        "in the example above, You should recommend a zonal optimisation in this case to shut the zone with the highest WCT (Zone 3) and keep the other two zones open for the foreseeable future.\n"
        "If the recent (last 2) well tests in the memory file is not a zonal test, then you should not recommend anything because probably zonal test was completed long ago.\n"
        "\n"
        'Output: you need to inform the below:'
        'Zonal Configuration: Commingle Production, Z1 Open, Z2 Open, Z3 Open, Z1 & Z2 Open etc...'
        'Agent Interpretation: Transient, Zonal test, Acid stimulation, Zonal optimization, etc...'
        'Engineer Action: No-Action - keep monitoring, Record zone performance from zone test, Analyse Zone performance, Zonal optimization recommended, Acid stimulation recommended, etc...'
        'Insights summary: A short 2-3 sentence of your findings after analysing the data land the highlighted anomalies'
        'NB: No acid stimulation can be done while a zonal test is underway. when a zone is tested, that zone is open and the other zones are closed'
    )

    MEMORY_SAVER_PROMPT = (
        'You will be given a well test data and a summary of analysis from an well test anomaly detector agent.'
        'Your job is to look into the anomaly analysis and look to understand if a zonal test or acid stimulation (considered an planned anomaly) is being mentioned.'
        'If yes, then you need to save the well test data in a memory file.'
        f'filepath: {file_path}'
    )

    return ANOMALY_DETECTOR_PROMPT, MEMORY_SAVER_PROMPT ,INTERPRETATOR_PROMPT
