import streamlit as st
import os
import pandas as pd
import json

def check_packaging_issues(review_text):
    """Simulate checking for packaging issues based on review text."""
    # Define issue categories and their keywords
    issue_categories = {
        'Damaged Packaging': ['crushed', 'torn', 'broken', 'old', 'dirty', 'water-damaged', 'dent', 'smashed'],
        'Leakage / Spillage': ['liquids leaking', 'spilled', 'leak', 'spillage'],
        'Tampered / Opened Packaging': ['seal broken', 'opened', 'used', 'tampered']
    }
    review_text = review_text.lower()
    issues_found = []

    # Check each category for matching keywords
    for category, keywords in issue_categories.items():
        if any(keyword in review_text for keyword in keywords):
            issues_found.append(category)

    if issues_found:
        return f"Potential issues detected: {', '.join(issues_found)}"
    return "No issues detected."

def main():
    """Main function to run the packaging dashboard."""
    st.set_page_config(page_title="Packaging Dashboard", layout="wide")
    st.title("Packaging Review Dashboard")

    # Section for entering a new packaging review
    st.header("Enter a New Packaging Review")
    review_text = st.text_area("Type your packaging review here:", height=100)
    if st.button("Submit Review"):
        st.write("Review submitted!")
        issue_result = check_packaging_issues(review_text)
        st.write(issue_result)

    # Section for visualizations
    st.header("Visualizations")
    visual_files = [
        "cluster_visual_20250926_0122_interactive.html",
        "cluster_visual_20250926_0122.html",
        "cluster_visual_20250926_0122.png",
        "topic_hierarchy_20250926_0122.html",
        "packaging_issues_finetuned.png",
        "packaging_types_finetuned.png",
        "product_categories_finetuned.png"
    ]
    selected_visual = st.selectbox("Select a visualization:", visual_files)
    visual_path = os.path.join("reports", selected_visual)
    if os.path.exists(visual_path):
        if selected_visual.endswith('.html'):
            with open(visual_path, "r", encoding="utf-8") as file:
                html_content = file.read()
            styled_html = f'<div style="max-width: 800px; margin: 0 auto;">{html_content}</div>'
            st.components.v1.html(styled_html, height=500, scrolling=True)
        elif selected_visual.endswith('.png'):
            st.image(visual_path, caption=selected_visual, use_column_width=False, width=800)
    else:
        st.write(f"Visualization {selected_visual} not found in the reports folder.")

    # Section for queryable reports
    st.header("Queryable Reports")
    report_files = [
        "crosstab_packaging_20250926_0126.csv",
        "main_topic_info_20250926_0122.csv",
        "meta_categories_top_20250926_0126.csv",
        "summary_finetuned.json"
    ]
    selected_report = st.selectbox("Select a report to query:", report_files)
    report_path = os.path.join("reports", selected_report)

    if os.path.exists(report_path):
        if selected_report.endswith('.csv'):
            df = pd.read_csv(report_path)
            st.write("Report Contents:", df)
            query = st.text_input("Enter a query (e.g., column_name == 'value'):")
            if query:
                try:
                    filtered_df = df.query(query)
                    st.write("Query Result:", filtered_df)
                except Exception as e:
                    st.write(f"Error in query: {e}")
        elif selected_report.endswith('.json'):
            with open(report_path, "r", encoding="utf-8") as file:
                data = json.load(file)
            st.write("Report Contents:", data)
            query_key = st.text_input("Enter a key to filter JSON (e.g., 'key' for value):")
            if query_key:
                try:
                    filtered_data = {k: v for k, v in data.items() if query_key in k.lower()}
                    st.write("Query Result:", filtered_data if filtered_data else "No matching keys found.")
                except Exception as e:
                    st.write(f"Error in query: {e}")
    else:
        st.write(f"Report {selected_report} not found in the reports folder.")

if __name__ == "__main__":
    main()