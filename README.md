<h2>💧 DBPFinder: Sensor Placement Optimization for Disinfection Byproducts in Water Distribution Networks</h2>
<p>DBPFinder is an interactive decision-support tool designed to aid water utilities and researchers in identifying optimal sensor locations for monitoring Disinfection Byproducts (DBPs), such as Trihalomethanes (THMs) and Haloacetic Acids (HAAs). It leverages environmental parameters, hydraulic simulations (via EPANET), and customizable risk models to support placement strategies under various performance objectives.</p>

<h4> Multi-objective optimization: Supports placement based on:</h4>
- Time of detection  </br>
- Normalized Concentration Score   </br>
- Regulatory Event Occurance   </br>
- Contract-based Risk Weighting  </br>

<h4> Flexible DBP modeling:</h4>
- Use built-in standard formation equations for trihalomethanes and haloacetic acids </br>
- Define your own custom formation equations 

<h4>Visualization:</h4>
- View sensor placements over the network </br>
- Node-specific score bars depending on selected performance objectives

<h4>Support for any water distribution network .inp file:</h4>
- Integration with the WNTR library for hydraulic and quality simulation

<h4>Data-driven configuration:</h4>
- Load Excel or CSV files with environmental conditions

<h2>📦Requirements</h2>
- Python 3.7+
<h5>Required packages:</h5>
-streamlit </br>
-wntr </br>
-matplotlib </br>
-pandas </br>
-numpy </br>
-networkx </br>
-openpyxl
<h5>Install them via:</h5>

```pip install -r requirements.txt```

<h2>Quick Start:</h2>

```git clone https://github.com/superworld-cyens/DBPFinder.git``` </br>
``` streamlit run DBPFinder.py ```

