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

<h2>📖Manual</h2>
<p>You can find the manual of the software [here](https://superworld.cyens.org.cy/projects/dbp_finder/DBPFinder_Documentation_v1.pdf).</p>

<h2>Examples Folder </h2>
<p>The "Examples" folder contains 3 different files to showcase what is required for the software to run the simulations correctly. It contains a file for the water distribution network, a sample of environmental data and how the excel file should be constructed as well as a contracts text file to tackle the performance objective of minmizing the mass consumption.</p>

<h2>📝Changelog</h2>
- v1.0 -> Release of DBPFinder. </br>
- v1.0.1 -> Bug fixes related to the "contracts" performance objective. </br>
- v1.0.2 -> Several bug fixes related to the performance objectives & new environmental data example. 


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
