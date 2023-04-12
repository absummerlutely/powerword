# PowerWord-Research Keyword Analyzer
This repository contains code to analyze a dataset of research publications and identify the top 10 most impactful keywords. The code uses the YAKE keyword extraction library to extract keywords from publication abstracts.

# Getting Started
These instructions will guide you through setting up the project on your local machine.

# Prerequisites
publication data can be found on the free version of Dimensions: https://app.dimensions.ai/discover/publication

The following Python packages are required to run the code:
pandas
yake
You can install these packages using pip:
pip install pandas yake

# Usage
1.Clone the repository to your local machine:  
bash
git clone https://github.com/yourusername/research-keyword-analyzer.git  
Place your dataset of research publications in CSV format in the project folder. The dataset should contain columns: "title", "abstract".   
Name this file publications.csv.  

2.Run the main script:  
python powerword.py   

3.The top 10 keywords will be printed to the console, along with their impact scores.  

# Contributing  
If you'd like to contribute to this project, feel free to submit a pull request.  

# License  
This project is licensed under the MIT License - see the LICENSE file for details.  
