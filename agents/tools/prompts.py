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
    f"You are an anomaly detection agent for well test data in an Oil & Gas company. "
    f"Think and reason like a reservoir engineer.\n"
    "\n"
    "You will be provided with well test data (one entry at a time). Your job is to identify anomalies based on:\n"
    "- Oil rate (log_diff_oil)\n"
    "- Liquid rate (log_diff_liq)\n"
    "- Water cut (log_diff_wct)\n"
    "\n"
    "These log differences are precomputed as: log_diff = ln(current_value / previous_value). "
    f"The anomaly detection threshold is ±{threshold}.\n"
    "\n"
    "### Types of Anomalies and How to Detect Them:\n"
    "\n"
    "**1. Zonal Test**\n"
    "- A zonal test is a short-term diagnostic test to evaluate individual zone performance.\n"
    "- Detection Rule:\n"
    f"    • Two of the log_diff_z1bhp_meanbhp, log_diff_z2bhp_meanbhp, or log_diff_z3bhp_meanbhp must be **greater than +{threshold}**\n"
    f"    • One must be **less than -{threshold}**\n"
    f"    • The zone with the low BHP (below -{threshold}) is **open and being tested**; the other two are **closed**.\n"
    "\n"
    "**2. Zonal Optimization**\n"
    "- A zonal optimization is a semi-permanent shut-off of a poorly performing zone.\n"
    "- Detection Rule:\n"
    f"    • Only **one** of the BHP log differences is **greater than +{threshold}**\n"
    f"    • The **other two** must be **below -{threshold}**\n"
    f"    • This means two zones are open and one is shut.\n"
    "\n"
    "**⚠️ Important Distinction**\n"
    "- If two BHP log differences are **below -{threshold}** and one is **above +{threshold}**, it is a **zonal optimization**, NOT a test.\n"
    "\n"
    "**3. Acid Stimulation**\n"
    "- This is a treatment to improve flow by dissolving near-wellbore deposits.\n"
    "- Detection Rule:\n"
    f"    • log_diff_oil is **greater than +{threshold}**\n"
    "- Additional Notes:\n"
    "    • Acid stimulations usually result in an oil rate jump that persists for weeks.\n"
    "    • They may also cause a BHP increase and a small shift in WCT.\n"
    "\n"
    "**4. Flush Production**\n"
    "- This occurs when a well is reopened after being shut in, causing pressure build-up.\n"
    "- Detection Rule:\n"
    f"    • Sudden increase in both oil and liquid rate (log_diff_oil and log_diff_liq > {threshold})\n"
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
        'You will be given a well test data and a summary of analysis from an anomaly detector agent.'
        'you will have to interpret the data and give a generate your findings/interpretation as an experience Reservoir Engineer would. you need to inform the below:'
        'Zonal Configuration: Commingle Production, Z1 Open, Z2 Open, Z3 Open, Z1 & Z2 Open etc...'
        'Agent Interpretation: Transient, Zonal test, Acid stimulation, Zonal optimization, etc...'
        'Engineer Action: No-Action - keep monitoring, Record zone performance from zone test, Analyse Zone performance, Zonal optimization recommended Acid stimulation recommended, etc...'
        'Insights summary: A short 2-3 sentence of your findings after analysing the data land the highlighted anomalies'
        'NB: No acid stimulation can be done while a zonal test is underway. when a zone is tested, that zone is open and the other zones are closed'
    )

    MEMORY_SAVER_PROMPT = (
        'You will be given a well test data and a summary of analysis from an well test anomaly detector agent.'
        'Your job is to look into the anomaly analysis and look to understand if a zonal test (considered an planned anomaly) is being mentioned.'
        'If yes, then you need to save the well test data in a memory file.'
        f'filepath: {file_path}'
    )

    return ANOMALY_DETECTOR_PROMPT, MEMORY_SAVER_PROMPT ,INTERPRETATOR_PROMPT
