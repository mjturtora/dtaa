# dtaa: Data Table Auto-Analyzer

1) Inventories 2D data table in either .csv or .xlsx format (hardwired switch)


2) CLI or hardwired input filenames


3) Outputs to responsive html file in "/Reports" and writes png plot files to a subfolder


5) Outputs to jinja2 html template with sections conditioned on data type presence


   * Table dimensions (Worksheet shape and name)


   * Hyperlinked Table of Contents


   * Tables:
     * Table of All Column Names with data types
     * Table of Numeric columns with summary stat's
     * Table of (Python) Object Variables with PanDAS string stat's
     * Lists any empty column names and drops them from further processing
     * Tests for missing values and if they exist, produces tabular and graphical analyses ( using ResidentMario's missingno). Custom grouped bar plots of non-zero, zero, an nan numbers. Multiple plots of maxbars each (hardwired for now).


   * Graphics:
     * Time Series plots for time columns (soon to come)
     * Histograms for numeric variables 
     * Horizontal Descending bar charts with all catogories if < 10,000 categories
     * Vertical descending bar charts of top (maxbars) categories of all categorical variables
