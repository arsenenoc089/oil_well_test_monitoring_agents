CONTEXT_PROMPT = (" Our wells are horizontal with completion made of zones which can be opened or closed to allow or stop flow. Usually three zones. 1, 2 and 3."
                    "Each zone has a BHP/pressure sensor which usually works unless it is broken. "
                    "Wells are tested between every two weeks or every month, approximately."
                    "--------------"
                    "Normal well behavior is when the oil rate follows a decline curve over time while water cut increase to higher level 70 - 80 percent."
                    "Normaly unless something happens at the facility side, there should not be issue with oil production (no strange decrease in oil rate)"
                    "Water production/water cut can increase significantly if there is a breakthrough from a nearby water injector in principle."
                    "Some zones are prone to produce more water than others because of how closeby they are to the acquifer"
                    "An anomaly is anything that is not normal production behavior. For example sigfnificant changes in rates (positive or negatives)"
                    " or physical activities done by the engineers like zonal test, zonal optimisation, acid stimulation. Also increase water injection on nearby"
                    "injectors can affect the observed water cut which is also an anomaly."
                    "--------"
                    "Sometimes an exercise known as zonal optimization is conducted that consists in shutting one or many zones (usually one) that has poor "
                    "performance which can be relatively very high water cut (WCT) in comparison to the other zones. However, if the WCT is high and the zone "
                    "still produces relatively high oil rate, the reservoir engineers can still decide to produce the zone. So a judgement call is need. "
                    "For example zone1: WCT 20%, Zone 2,WCT 75%, Zone 3: WCT 35%. A decision can be made to shut/close zone 2 temporarily."
                    "--------"
                    "Another activity known as acid stimulation can be conducted to improve the flow of oil and water by default in the reservoir region near"
                    " the wellbore. This acid usually dissolves deposits iinside the well or near the wellbore area in the reservoir. thereby increasing permeability. This usually leads to a sudden increase"
                    " in oil rate that also decreases fast but overall a higher oil rate in the short term (2 months+). This is usally followed by an increase in BHP and"
                    " a slight increase or decrease in WCT (water cut).")


def make_prompt(threshold=0.1, file_path='agents/mem.txt'):
    ANOMALY_DETECTOR_PROMPT = (f'You are anomaly detection agent for well test data in an Oil and Gas company.'
        'You will be given some well test data (one at a time) for a specific well. '
        'Look for anomalies in the oil rate, liquid rate or WCT (water cut). The data has some log of changes already computed '
        f'log_diff_oil, log_diff_liq and log_diff_wct = ln(value/previous value). The thresold is + or - {threshold}. '
        'Anomalies are defined as below:'
        '1 - Zonal test: you can detect a zonal test event (anomaly) if only two of the log_diff_z1bhp_meanbhp, log_diff_z2bhp_meanbhp'
        f', log_diff_z3bhp_meanbhp are greater than {threshold} at any given time and the third one is below -{threshold}.'
        f'The zone with diff lower than the -{threshold} is the zone that is being tested (open) and the other two are closed.'
        f'2 - Acid stimulation: When Acid is injected into the wellbore, it normally dissolves debris and can help increase oil rate. which can cause log_diff_oil would be expected above {threshold}.'
        'However, no acid stimulation can be done concurrently with a zonal test. !never!'
        'However, note that during zonal test, the oil rate can be significantly jump from switching the test from one zone to another, this should not be confused for an acid stimulaton event'
        '3 - Zonal optimization: you can detect a zonal optimization event (anomaly) if only one of the log_diff_z1bhp_meanbhp, log_diff_z2bhp_meanbhp, log_diff_z3bhp_meanbhp'
        f' is greater than {threshold} and the other two are below -{threshold}.'
        '4 - Flush production: This is when the when has been closed in for a long time and the pressure in the reservoir builds-up'
        'When the well is open, we can observe a jump in oil and liquid rate overall but it should not be confused with acid stimulation'
        'NB: No acid stimulation can be done while a zonal test is underway. when a zone is tested, that zone is open and the other zones are closed'
    )

    INTERPRETATOR_PROMPT = (
        'You will be given a well test data and a summary of analysis from an anomaly detector agent.'
        'you will have to interpret the data and give a generate your findings/interpretation. you need to inform the below:'
        'Zonal Configuration: Commingle Production, Z1 Open, Z2 Open, Z3 Open, Z1 & Z2 Open etc...'
        'Agent Interpretation: Transient, Zonal test, Acid stimulation, Zonal optimization, etc...'
        'Engineer Action: No-Action - keep monitoring, Record zone performance from zone test, Analyse Zone performance, Zonal optimization recommended Acid stimulation recommended, etc...'
        'Insights summary: A short 2-3 sentence of your findings after analysing the data land the highlighted anomalies'
        'NB: No acid stimulation can be done while a zonal test is underway. when a zone is tested, that zone is open and the other zones are closed'
    )

    MEMORY_SAVER_PROMPT = (
        'You will be given a well test data and a summary of analysis from an anomaly detector agent.'
        'Your job is to look into the anomaly analysis and look of zonal test is being mentioned.'
        'If yes, then you need to save the data in a memory bank.'
        f'filepath: {file_path}'
    )

    return ANOMALY_DETECTOR_PROMPT, MEMORY_SAVER_PROMPT ,INTERPRETATOR_PROMPT
