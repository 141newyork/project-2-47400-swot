import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def convert_to_json(data):
    json_data = {}

    for category, group in data.groupby('CATEGORY'):
        factors = []
        for index, row in group.iterrows():
            factor = {
                'factor_type': row['FACTOR TYPE'],
                'sl_number': row['Sl #'],
                'param_name': row['PARAM NAME'],
                'est_value': row['EST. VALUE IN CURRENCY'],
                'min_prob': row['MIN PROB  %'],
                'realistic_prob': row['REALISTIC PROB  %'],
                'max_prob': row['MAX PROB %']
            }
            factors.append(factor)

        json_data[category] = factors

    return json_data

def calculate_extra_data(data):
    for category in data:
        factors = data[category]
        for factor in factors:
            min_prob = factor['min_prob']
            realistic_prob = factor['realistic_prob']
            max_prob = factor['max_prob']
            est_value = factor['est_value']

            stats_prob_3point = round((min_prob + realistic_prob + max_prob) / 3, 1)
            factor['stats_prob_3point'] = stats_prob_3point

            stats_prob_pert = round((min_prob + (realistic_prob * 4) + max_prob) / 6, 1)
            factor['stats_prob_pert'] = stats_prob_pert

            min_prob_value = round(est_value * (min_prob / 100), 1)
            factor['min_prob_value'] = min_prob_value

            max_prob_value = round(est_value * (max_prob / 100), 1)
            factor['max_prob_value'] = max_prob_value

            avg_prob_value = round((min_prob_value + max_prob_value) / 2, 1)
            factor['avg_prob_value'] = avg_prob_value

            realistic_prob_value = round(est_value * (realistic_prob / 100), 1)
            factor['realistic_prob_value'] = realistic_prob_value

            stats_prob_3point_value = round(est_value * (stats_prob_3point / 100), 1)
            factor['stats_prob_3point_value'] = stats_prob_3point_value

            stats_prob_pert_value = round(est_value * (stats_prob_pert / 100), 1)
            factor['stats_prob_pert_value'] = stats_prob_pert_value

    return data

def add_data(df):
    st.header("Add Data")
    category = st.selectbox("Select Category", ["Strength", "Weakness", "Opportunity", "Threat"])
    factor_type = st.text_input("Factor Type")
    sl_number = st.number_input("Sl #", value=1, step=1)
    param_name = st.text_input("Param Name")
    est_value = st.number_input("Est. Value in Currency")
    min_prob = st.number_input("Min Prob %")
    realistic_prob = st.number_input("Realistic Prob %")
    max_prob = st.number_input("Max Prob %")

    if st.button("Add"):
        new_factor = {
            "CATEGORY": category,
            "FACTOR TYPE": factor_type,
            "Sl #": sl_number,
            "PARAM NAME": param_name,
            "EST. VALUE IN CURRENCY": est_value,
            "MIN PROB  %": min_prob,
            "REALISTIC PROB  %": realistic_prob,
            "MAX PROB %": max_prob
        }

        # Append new data to DataFrame
        new_row = pd.DataFrame([new_factor])
        df = pd.concat([df, new_row], ignore_index=True)

        # Save DataFrame to the uploaded file
        df.to_excel('uploaded_file.xlsx', index=False)

        st.success("Data added successfully!")

def generate_graph(category_data, swap_axes):
    plt.figure(figsize=(14, 10))

    data_sets = [
        'est_value',
        'min_prob_value',
        'avg_prob_value',
        'max_prob_value',
        'realistic_prob_value',
        'stats_prob_3point_value',
        'stats_prob_pert_value'
    ]

    param_names = [factor['param_name'] for factor in category_data]
    values = np.array([[factor[data_set] for data_set in data_sets] for factor in category_data])

    if swap_axes:
        param_names, data_sets = data_sets, param_names
        values = values.T

    bar_width = 0.2
    index = np.arange(len(data_sets))

    num_factors = len(category_data)
    num_data_sets = len(data_sets)

    for i in range(num_factors):
        color_map = plt.cm.get_cmap('tab10')  # Use tab10 color map
        color = color_map(i / num_factors)  # Assign a color based on factor index

        plt.bar(index + i * bar_width, values[i], bar_width, label=f'{param_names[i]}', color=color)
        plt.xticks(index + (num_factors - 1) * bar_width / 2, data_sets, rotation=45)

    plt.xlabel('Data Sets')
    plt.ylabel('Value')
    plt.title('Bar Graph')
    plt.xticks(index + bar_width * (len(category_data) - 1) / 2, data_sets, rotation=45)
    plt.legend()
    plt.tight_layout()

    return plt

def perform_monte_carlo_analysis(data):
    # Create a dictionary to store the simulation results
    monte_carlo_results = {}

    # Iterate over each category in the data
    for category, factors in data.items():
        monte_carlo_results[category] = []

        # Iterate over each factor within the category
        for factor in factors:
            est_value = factor['est_value']
            min_prob = factor['min_prob']
            max_prob = factor['max_prob']

            # Generate random samples based on the probability distribution
            samples = np.random.triangular(min_prob, (min_prob + max_prob) / 2, max_prob, size=1000)

            # Calculate the simulated values by multiplying the samples with the estimated value
            simulated_values = samples * est_value

            # Store the simulated values in the result dictionary
            monte_carlo_results[category].append({
                'factor': factor['param_name'],
                'simulated_values': simulated_values
            })

    return monte_carlo_results


def display_monte_carlo_plots(monte_carlo_results):
    # Display the Monte Carlo plots for each category
    for category, results in monte_carlo_results.items():
        st.subheader(f"Monte Carlo Analysis - {category}")

        for result in results:
            factor = result['factor']
            simulated_values = result['simulated_values']

            fig, ax = plt.subplots()
            sns.histplot(simulated_values, kde=True, ax=ax)
            ax.set_xlabel('Simulated Values')
            ax.set_ylabel('Frequency')
            ax.set_title(f"Monte Carlo Analysis - {category}: {factor}")
            st.pyplot(fig)

def main():
    st.title('CSC 47400 Project 2 - SWOT')

    # File upload
    uploaded_file = st.file_uploader('Upload Excel file', type=['xlsx'])

    if uploaded_file is not None:
        # Read Excel file into a DataFrame
        try:
            df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f'Error reading Excel file: {str(e)}')
            return

        # Convert DataFrame to JSON
        json_data = convert_to_json(df)

        # Calculate extra data
        data_with_extra_data = calculate_extra_data(json_data)

        # Select category
        desired_categories = ['Strength', 'Weakness', 'Opportunity', 'Threat']
        category_colors = {'Strength': 'green', 'Weakness': 'yellow', 'Opportunity': 'blue', 'Threat': 'red'}

        # Create two rows of buttons for each category
        col1, col2 = st.columns(2)

        with col1:
            for category in desired_categories[:2]:
                button_color = category_colors.get(category, "black")
                button_clicked = st.button(category, key=f"{category}_button", help=f"{category} Graph")
                if button_clicked:
                    category_data = data_with_extra_data.get(category, [])
                    if len(category_data) > 0:
                        plt = generate_graph(category_data, swap_axes=False)
                        st.markdown(
                            '<p align="center" style="color: {}; font-size: 24px;">{} Graph</p>'.format(
                                button_color, category),
                            unsafe_allow_html=True
                        )
                        st.pyplot(plt)

        with col2:
            for category in desired_categories[2:]:
                button_color = category_colors.get(category, "red")
                button_clicked = st.button(category, key=f"{category}_button", help=f"{category} Graph")
                if button_clicked:
                    category_data = data_with_extra_data.get(category, [])
                    if len(category_data) > 0:
                        plt = generate_graph(category_data, swap_axes=False)
                        st.markdown(
                            '<p align="center" style="color: {}; font-size: 24px;">{} Graph</p>'.format(
                                button_color, category),
                            unsafe_allow_html=True
                        )
                        st.pyplot(plt)

        # Perform Monte Carlo analysis
        monte_carlo_results = perform_monte_carlo_analysis(data_with_extra_data)

        # Display Monte Carlo plots
        display_monte_carlo_plots(monte_carlo_results)

        # Add Data
        add_data(df)

if __name__ == '__main__':
    main()
