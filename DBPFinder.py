import streamlit as st
import pandas as pd
import numpy as np
import wntr
import networkx as nx
import matplotlib.pyplot as plt
import io
import random
import tempfile
import base64

st.title("DBPFinder")

# Initialize session state for abbreviation toggle
if "show_abbreviations" not in st.session_state:
    st.session_state.show_abbreviations = False

# Toggle button for abbreviations
if st.button("Show/Hide Abbreviations"):
    st.session_state.show_abbreviations = not st.session_state.show_abbreviations

if st.session_state.show_abbreviations:
    st.info("""
    **Abbreviations:**
    - **TOC**: Total Organic Carbon
    - **Cl2**: Chlorine Concentration
    - **Br**: Bromide Concentration
    - **DOC**: Dissolved Organic Carbon
    - **Temp**: Temperature
    - **pH**: Acidity/Basicity Level
    - **DBP**: Disinfection Byproducts
    """)
    
    
# Display a pop-up box with information about file formatting
def show_popup():
    st.warning("""
    **File Formatting Instructions:**
    - The `.inp` file should be a valid EPANET input file.
    - The Excel file must contain environmental data, including parameter columns.
    - Ensure all necessary parameters for DBP calculations are present.
    - If "Contracts" is selected as a performance objective, please follow the example given below for formatting.
    """)
    
# Show the pop-up on startup
show_popup()
    
# Create a sample Excel file for download
sample_data = {
        "Node": ["N1", "N2", "N3"],
        "TOC": [2.5, 3.1, 4.0],
        "Cl2": [1.2, 1.5, 1.8],
        "Br": [0.05, 0.07, 0.06],
        "Temp": [20, 22, 24],
        "pH": [7.2, 7.4, 7.6]
    }
df = pd.DataFrame(sample_data)
output = io.BytesIO()
with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sample Data')
        writer.close()
st.download_button(
        label="Download Sample Excel File",
        data=output.getvalue(),
        file_name="sample_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    
# Sample Contracts Data for Download
contracts_data = """Node\tContracts
1_1000\t0
1_1001\t5
1_1002\t0
1_1003\t12.5
1_1004\t0
1_1005\t2.5
1_1006\t0
1_1007\t5
1_1009\t0
"""
st.download_button(
    label="Download Sample Contracts File",
    data=contracts_data,
    file_name="sample_contracts.txt",
    mime="text/plain"
)

# Load EPANET File
inp_file = st.file_uploader("Upload EPANET .inp File", type=["inp"])

# Load Excel File
data_file = st.file_uploader("Upload Environmental Data (Excel)", type=["xlsx", "csv"])

# Choose Mandatory DBP Equation
st.subheader("Select Mandatory DBP Equation:")
selected_dbps = st.multiselect("Choose DBP equations to use", ["THM Equation", "HAA Equation"], default=["THM Equation"])

# Equation library with formula explanations
thm_equations = {
    "Standard THM Formula": {
        "coeffs": [0.04121, 1.098, 0.152, -0.068, 0.609, 1.601, 0.263],
        "formula": "THM = 0.04121 * TOC^1.098 * Cl2^0.152 * Br^-0.068 * Temp^0.609 * pH^1.601 * Time^0.263"
    },
    "Alternative THM Formula": {
        "coeffs": [0.0501, 1.2, 0.18, 0, 0.5, 1.4, 0.22],
        "formula": "THM = 0.0501 * TOC^1.2 * Cl2^0.18 * Temp^0.5 * pH^1.4 * Time^0.22"
    },
    "Custom THM Formula": {
        "coeffs": [],
        "formula": "THM = a * TOC^b * Cl2^c * Br^d * Temp^e * pH^f * Time^g"
    }
}

haa_equations = {
    "Standard HAA Formula": {
        "coeffs": [30.0, 0.997, 0.278, -0.138, 0.341, -0.799, 0.169],
        "formula": "HAA = 30.0 * TOC^0.997 * Cl2^0.278 * Br^-0.138 * Temp^0.341 * pH^-0.799 * Time^0.169"
    },
    "Alternative HAA Formula": {
        "coeffs": [28.5, 1.05, 0.25, 0, 0.38, -0.75, 0.15],
        "formula": "HAA = 28.5 * TOC^1.05 * Cl2^0.25 * Temp^0.38 * pH^-0.75 * Time^0.15"
    },
    "Custom HAA Formula": {
        "coeffs": [],
        "formula": "HAA = a * TOC^b * Cl2^c * Br^d * Temp^e * pH^f * Time^g"
    }
}

selected_thm_eq = None
selected_haa_eq = None

if "THM Equation" in selected_dbps:
    selected_thm_label = st.selectbox("Select THM Equation", list(thm_equations.keys()), key = "select_thm_eq")
    if selected_thm_label == "Custom THM Formula":
        a = st.number_input("a (base multiplier)", key ="a_thm", value=0.04121)
        b = st.number_input("TOC exponent (b)",key="toc_thm", value=1.098)
        c = st.number_input("Cl2 exponent (c)", key="cl2_thm", value=0.152)
        d = st.number_input("Br exponent (d)", key="br_thm", value=-0.068)
        e = st.number_input("Temp exponent (e)", key="temp_thm", value=0.609)
        f = st.number_input("pH exponent (f)", key="ph_thm", value=1.601)
        g = st.number_input("Time exponent (g)", key="time_thm", value=0.263)
        selected_thm_eq = [a, b, c, d, e, f, g]
    else:
        selected_thm_eq = thm_equations[selected_thm_label]["coeffs"]
        st.caption(f"Equation used: {thm_equations[selected_thm_label]['formula']}")

if "HAA Equation" in selected_dbps:
    selected_haa_label = st.selectbox("Select HAA Equation", list(haa_equations.keys()), key = "select_haa_eq")
    if selected_haa_label == "Custom HAA Formula":
        a = st.number_input("a (base multiplier)", key="a_haa", value=30.0)
        b = st.number_input("TOC exponent (b)", key="toc_haa", value=0.997)
        c = st.number_input("Cl2 exponent (c)", key="cl2_haa", value=0.278)
        d = st.number_input("Br exponent (d)", key="br_haa", value=-0.138)
        e = st.number_input("Temp exponent (e)", key="temp_haa",value=0.341)
        f = st.number_input("pH exponent (f)", key="ph_haa", value=-0.799)
        g = st.number_input("Time exponent (g)", key="time_haa", value=0.169)
        selected_haa_eq = [a, b, c, d, e, f, g]
    else:
        selected_haa_eq = haa_equations[selected_haa_label]["coeffs"]
        st.caption(f"Equation used: {haa_equations[selected_haa_label]['formula']}")


# Disable performance objective conflict
disable_thm_event = "HAA Equation" in selected_dbps
disable_haa_event = "THM Equation" in selected_dbps


# Choose Performance Objectives
st.subheader("Select at least one performance objective:")
time_detection = st.checkbox("Time of Detection*", value=False)
normalized_score = st.checkbox("Normalized Score Placement*", value=False)
contracts = st.checkbox("Contracts (Optional)", help = "'Contracts' option requires a .txt file that has contracts for each node in the network.")
THM_events = st.checkbox("THM Events (Optional)", help = "An event is when the concentration of DBPs is higher than the regulatory limit.", value = False, disabled = disable_thm_event)
HAA_events = st.checkbox("HAA Events (Optional)", help = "An event is when the concentration of DBPs is higher than the regulatory limit.", value = False, disabled = disable_haa_event)

# Show contracts file uploader if selected
contracts_file = None
if contracts:
    contracts_file = st.file_uploader("Upload Contracts File (TXT)", type=["txt"])

# Adjust Weights
st.subheader("Adjust Weights of Disinfection Byproducts:")
thm_weight = st.slider("THM Weight", 0, 5, 3)
haa_weight = st.slider("HAA Weight", 0, 5, 1)

# Informative box about DBP regulatory thresholds
st.subheader("Disinfection Byproducts Regulatory Thresholds")
st.info(
    "- **THMs**: Regulated at 100 µg/L in Europe, 80 µg/L in the US.\n"
    "- **HAA5**: Regulated at 60 µg/L in the US.\n"
    "- Other DBPs may have different regulations depending on the country."
)

#Select number of sensors
st.subheader("Select number of sensors to place:")
sensor_number = st.slider("Sensors",1,100,5)
randomize_injection = st.checkbox("Randomize injection", value=False, help = "Chooses nodes from the network randomly and sets initial quality of chlorine.")

# Contact
st.subheader("Contact information")
st.info("For inquiries and potential bugs contact: a.magklis@cyens.org.cy")

# Setup temporary image saving function
image_paths = []
def save_plot_as_png(fig, filename):
    path = tempfile.NamedTemporaryFile(delete=False, suffix='.png').name
    fig.savefig(path, bbox_inches='tight')
    image_paths.append((filename, path))

# Run Simulation
if not (time_detection or normalized_score):
    st.error("Please select at least one mandatory performance objective (Time of Detection or Normalized Score Placement).")
elif st.button("Run Simulation & Show Results"):
    if inp_file is None or data_file is None:
        st.error("Both the EPANET file and environmental data file must be uploaded to run the simulation.")
    else:
        with open("temp_inp_file.inp", "wb") as f:
            f.write(inp_file.read())
        wn = wntr.network.WaterNetworkModel("temp_inp_file.inp")

        data = pd.read_excel(data_file) if data_file.name.endswith('.xlsx') else pd.read_csv(data_file)
        G = wn.to_graph()
        pos = nx.get_node_attributes(G, 'pos')

        if contracts_file is not None:
            contracts_df = pd.read_csv(contracts_file, sep="\t", header=None, names=["Node", "Contracts"])
            contracts_dict = dict(zip(contracts_df["Node"], contracts_df["Contracts"]))
            data["Contracts"] = data["Node"].map(contracts_dict).fillna(0)

        if randomize_injection:
            epanet_nodes = list(wn.junction_name_list)
            num_random_nodes = random.randint(1, len(epanet_nodes) // 2)
            random_nodes = random.sample(epanet_nodes, num_random_nodes)

        wn.options.quality.parameter == "TRACE"
        wn.options.hydraulic.demand_model = "PDD"
        sim = wntr.sim.EpanetSimulator(wn)
        results = sim.run_sim()

        node_quality = results.node["quality"]
        chlorine_threshold = 0.0000002
        detection_times = {}
        time_steps = node_quality.index
        for node in node_quality.columns:
            series = node_quality[node]
            idx = series[series >= chlorine_threshold].first_valid_index()
            detection_times[node] = (time_steps.get_loc(idx) * wn.options.time.hydraulic_timestep / 60) if idx else None
        data["Detection_Time"] = data["Node"].map(detection_times)

        required = ["TOC", "Cl2", "Br", "Temp", "pH"]
        if not all(col in data.columns for col in required):
            st.error("Missing one or more required parameters for DBP equation calculation.")
        else:
            data["Time"] = data["Node"].map(lambda n: results.node["quality"].iloc[-1].get(n, 72) / 3600)

            if "THM Equation" in selected_dbps and selected_thm_eq:
                a, b, c, d, e, f, g = selected_thm_eq
                data["THM_Concentration"] = a * data["TOC"]**b * data["Cl2"]**c * data["Br"]**d * data["Temp"]**e * data["pH"]**f * data["Time"]**g
                data["THM_Event"] = (data["THM_Concentration"] > 100).astype(int)

            if "HAA Equation" in selected_dbps and selected_haa_eq:
                a, b, c, d, e, f, g = selected_haa_eq
                data["HAA_Concentration"] = a * data["TOC"]**b * data["Cl2"]**c * data["Br"]**d * data["Temp"]**e * data["pH"]**f * data["Time"]**g
                data["HAA5_Event"] = (data["HAA_Concentration"] > 60).astype(int)

            concentration_cols = [c for c in ["THM_Concentration", "HAA_Concentration"] if c in data]
            data["Normalized Total Node Score"] = data[concentration_cols].sum(axis=1)
            min_score = data["Normalized Total Node Score"].min()
            max_score = data["Normalized Total Node Score"].max()
            data["Normalized Total Node Score"] = (data["Normalized Total Node Score"] - min_score) / (max_score - min_score)

            filtered_nodes = data[data["Normalized Total Node Score"] > 0.90].sort_values(by="Normalized Total Node Score", ascending=False)

            top_score_nodes = filtered_nodes.nlargest(sensor_number, "Normalized Total Node Score")["Node"]
            top_detection_time_nodes = filtered_nodes.nsmallest(sensor_number, "Detection_Time")["Node"]
            top_thm_event_nodes = filtered_nodes.nlargest(sensor_number, "THM_Event")["Node"] if "THM_Event" in data else []
            top_haa5_event_nodes = filtered_nodes.nlargest(sensor_number, "HAA5_Event")["Node"] if "HAA5_Event" in data else []

            def plot_sensor_placement(wn, sensor_nodes, title="Sensor Node Placement"):
                x_all, y_all = [], []
                for node_name in wn.node_name_list:
                    coord = wn.get_node(node_name).coordinates
                    if coord:
                        x_all.append(coord[0])
                        y_all.append(coord[1])

                x_sel, y_sel, sel_labels = [], [], []
                for node_name in sensor_nodes:
                    coord = wn.get_node(node_name).coordinates
                    if coord:
                        x_sel.append(coord[0])
                        y_sel.append(coord[1])
                        sel_labels.append(node_name)

                fig, ax = plt.subplots(figsize=(12, 10))
                ax.scatter(x_all, y_all, color='black', s=10, zorder=1)
                ax.scatter(x_sel, y_sel, color='red', edgecolors='black', s=120, zorder=2)

                for x, y, label in zip(x_sel, y_sel, sel_labels):
                    ax.text(x, y + 1.5, label, fontsize=9, ha='center', va='bottom', zorder=3,
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round'))

                ax.set_title(title, fontsize=16, pad=20)
                ax.axis('equal')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_facecolor('white')
                for spine in ax.spines.values():
                    spine.set_visible(False)

                for pipe_name in wn.pipe_name_list:
                    pipe = wn.get_link(pipe_name)
                    start = wn.get_node(pipe.start_node_name).coordinates
                    end = wn.get_node(pipe.end_node_name).coordinates
                    if start and end:
                        ax.plot([start[0], end[0]], [start[1], end[1]], color='black', linewidth=1, zorder=0)

                plt.tight_layout()
                st.pyplot(fig)


            def plot_bar_for_placement(df, node_column, value_column, selected_nodes, title, color='orange'):
                df[node_column] = df[node_column].astype(str).str.strip()
                selected_nodes = [str(n).strip() for n in selected_nodes]
                nodes = df[df[node_column].isin(selected_nodes)].copy()

                if nodes.empty:
                    print("⚠️ No matching nodes found for bar plot.")
                    return

                nodes = nodes.sort_values(by=value_column, ascending=True)

                plt.figure(figsize=(10, 6))
                plt.bar(nodes[node_column], nodes[value_column], color=color)
                plt.xlabel("Sensor Node")
                plt.ylabel(value_column)
                plt.title(title)
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                st.pyplot(plt.gcf())

            if normalized_score:
                plot_sensor_placement(wn, top_score_nodes, title="Placement By Total Concentration Score")
                plot_bar_for_placement(
                    df=filtered_nodes,
                    node_column="Node",
                    value_column="Normalized Total Node Score",
                    selected_nodes=top_score_nodes,
                    title="Normalized Score for Sensor Placement",
                    color="orange"
                )

            if time_detection:
                plot_sensor_placement(wn, top_detection_time_nodes, title="Placement By Time of Detection")
                plot_bar_for_placement(
                    df=filtered_nodes,
                    node_column="Node",
                    value_column="Detection_Time",
                    selected_nodes=top_detection_time_nodes,
                    title="Time of Detection for Sensor Placement",
                    color="orange"
                )

            if THM_events:
                plot_sensor_placement(wn, top_thm_event_nodes, title="Placement for THM detection")
                plot_bar_for_placement(
                    df=filtered_nodes,
                    node_column="Node",
                    value_column="THM_Event",
                    selected_nodes=top_thm_event_nodes,
                    title="THM occurrences for Sensor Placement",
                    color="orange"
                )

            if HAA_events:
                plot_sensor_placement(wn, top_haa5_event_nodes, title="Placement for HAA detection")
                plot_bar_for_placement(
                    df=filtered_nodes,
                    node_column="Node",
                    value_column="HAA5_Event",
                    selected_nodes=top_haa5_event_nodes,
                    title="HAA occurrences for Sensor Placement",
                    color="orange"
                )

            st.success("Simulation completed. Visualizations generated. You can now analyze placements and download results.")

            for name, path in image_paths:
                with open(path, "rb") as img_file:
                    b64 = base64.b64encode(img_file.read()).decode()
                    href = f'<a href="data:image/png;base64,{b64}" download="{name}">Download {name}</a>'
                    st.markdown(href, unsafe_allow_html=True)



