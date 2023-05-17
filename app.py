import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


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

def generate_graph(category_data, swap_axes,data_sets):
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
    plt.xticks(index + (num_factors - 1) * bar_width / 2, data_sets, rotation=45)
    plt.legend()
    plt.tight_layout()

    return plt

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

       # st.write('Excel data:')
        #st.dataframe(df)

        # Convert DataFrame to JSON
        json_data = convert_to_json(df)

        # Calculate extra data
        data_with_extra_data = calculate_extra_data(json_data)

       # st.write('Data with extra calculations:')
       # st.json(data_with_extra_data)

        # Select category
        desired_categories = ['Strength', 'Weakness', 'Opportunity', 'Threat']

        # Generate graphs for each category
        for category in desired_categories:
            category_data = data_with_extra_data.get(category, [])
            if len(category_data) > 0:
                data_sets = [
            'New Name 1',
            'New Name 2',
            'New Name 3',
            'New Name 4',
            'New Name 5',
            'New Name 6',
            'New Name 7'
        ]
                plt = generate_graph(category_data, swap_axes=False,data_sets=data_sets)
        
                if category == "Strength":
                    text_color = "green"
                elif category == "Weakness":
                    text_color = "yellow"
                elif category == "Opportunity":
                    text_color = "blue"
                elif category == "Threat":
                    text_color = "red"
                else:
                    text_color = "black"   
                st.markdown(
    '<p align="center" style="color: {}; font-size: 24px;">{} Graph</p>'.format(text_color, category),
    unsafe_allow_html=True
)
                st.pyplot(plt)
if __name__ == '__main__':
    main()
