## Team Formation (2497455):
Prairie Insights is a sole-author team (Rayne Wilde), formed by
considering regional significance and thematic relevance to actionable
insights on labor market dynamics. The name "Prairie Insights"
symbolizes clarity, directness, and focus on the Midwest prairie states
that anchor the project's recurring case studies.

## 2025-02-17: Acquire Data Milestone

### Project Goals (2501542)
Our team aims to investigate the relationship between education levels and labor market dynamics in Iowa and the broader Midwest region. Currently, we are focusing on analyzing data from the Bureau of Labor Statistics (BLS) due to challenges with the K-12 education data. In particular, we are working to provide new graduates with insights about which areas in the Midwest offer the best opportunities for labor market growth. Using BLS data, we will highlight regions with strong economic development. Additionally, we are integrating a fine-tuned LLaMA v4 model to assist in answering questions and providing valuable insights for graduates seeking to navigate these growing labor markets. Our audience includes new graduates, policymakers, and labor market analysts who can benefit from this data to make informed decisions about career opportunities and economic development.

### Team Evaluation (2501544)
As the sole member of Prairie Insights, I am making steady progress toward achieving the end-of-semester goal of having a fully functioning dashboard. I have successfully acquired the BLS data and am in the process of pulling data for the entire Midwest region, which will be integrated into the dashboard. The dashboard prototypes are also being developed, and I am keeping up with the timelines set for both this project and my other ongoing commitments. While there are no major obstacles currently hindering progress, the main challenge may come from integrating the final dataset into the dashboard and ensuring smooth interaction for users. To overcome this, I plan to focus on iterative development and testing of the dashboard to address any potential issues as they arise, while ensuring the data integration is seamless and accessible for end-users.

### Technology Plan (2501538)
For the entirety of the data pipeline, from data collection through the creation of the dashboard, I plan to use a combination of technologies to ensure a modular, scalable, and user-friendly solution. Data will be processed and integrated using Spark, which will allow for efficient horizontal scaling. The backend will be built with Flask, while the frontend UI/UX will be developed using Dash and Plotly via Python, with integration of JavaScript and HTML/CSS where needed to enhance UI/UX clarity. The dashboards will include overall insights, EDA, and comparisons, with a focus on providing intuitive visualizations of the BLS data. The system will be deployed via GitLab, utilizing Docker and Kubernetes for containerized deployment, ensuring modularity and ease of scalability. I anticipate needing minimal support in deciding on the technologies, as I have already made selections based on the project requirements. However, I may seek guidance during the integration and deployment stages to ensure best practices are followed, particularly when working with Kubernetes and Spark for larger datasets.

### Data Wrangling (2501536)
The data collected thus far consists of Bureau of Labor Statistics (BLS) data for the Midwest region, which has been expanded to include data going back to 1995. Additional series have been incorporated, and the data has been pivoted to a wider format, making it more human-readable and understandable. This cleaned and structured data has been loaded into EDA scripts to perform basic analyses, including correlation, independence testing, and time series analysis. The main roadblocks encountered so far involve the complexity of ensuring that the dataset is fully integrated and aligned across different years and series, which can sometimes lead to inconsistencies or gaps in the data. However, the goal is to refine and validate the dataset through iterative testing to ensure its accuracy and reliability for future analyses.

## 2025-02-23: Project Goals Milestone

### Project Goals (2504009)
The dashboard targets new graduates and policymakers, offering insights into labor market growth across the Midwest. It helps identify high-growth regions and inform career and policy decisions by providing real-time predictions and visualizations. The goal is to answer key questions like "Which regions have the highest job growth?" and "How do economic factors impact labor markets?"

### Modeling Plan (2504012)
We will use regression models (e.g., linear regression, random forest) and time series models (e.g., ARIMA) to predict labor market growth, using variables like unemployment rate and income. These models will forecast labor market trends and guide regional predictions. The retrained LLaMA v4 model will provide additional insights, while original models will be benchmarked for comparison, integrating into the dashboard for real-time forecasting.

### Exploratory Analysis (2504010)
We are using Pandas, NumPy, Matplotlib, Seaborn, and Spark to explore and analyze BLS data for the Midwest region, focusing on variables like unemployment rate, average income, and labor force participation. Initial findings reveal regional variations in labor market growth, with some areas showing consistent upward trends, while others have cyclical patterns. Basic visualizations have been created and will be included in the final dashboard, and we are addressing missing data and outliers as part of the ongoing analysis.

### Project Goals Milestone: Project Progress (2504014)
As the sole team member, I’ve managed all aspects of the project, including data acquisition and GitHub commits, with the tech stack set up on my local machine. Next steps include refining the dashboard prototype, integrating additional data, and testing models. Communication is strong, and I have a clear plan to address data issues and improve the user experience.

## 2025-03-03: Exploratory Analysis Milestone

### Brainstorm Dashboard (2505224)
The dashboard will provide new graduates and policymakers with interactive visualizations of labor market trends in the Midwest, using Plotly Dash for dynamic filtering, trend analysis, and predictive modeling. Key features include region-based filtering, time-series sliders, and comparison charts to answer questions like "Which regions have the highest job growth?" and "How do economic factors impact labor markets?" Early visual drafts are promising, and data accuracy is being tested to ensure clear, actionable insights.

### Data Report (2505225)
The dataset is comprehensive but requires refinement, particularly in handling missing values and ensuring consistency across time. While it answers key labor market questions, additional datasets on regional economic indicators may improve insights. Some datasets will be excluded due to incomplete reporting, and assumptions about data completeness and economic stability will be considered for modeling.

### Project Progress (2505226)
As the sole team member, I have been managing data processing, model development, and dashboard prototyping, with all necessary tools set up on my local machine. The next steps involve refining data integration, enhancing dashboard interactivity, and validating predictive models. Communication is smooth, and I have a clear roadmap for progress. The main roadblock is handling missing data and ensuring consistency across datasets, which will be addressed through imputation and validation techniques.

### Exploratory (2505222)
As of December 2024, Iowa's unemployment rate increased to 3.2%, up from 3.0% in December 2023. The labor force participation rate rose to 66.4% during the same period. A time-series analysis from January 2020 to December 2024 shows a peak in unemployment in 2020, likely due to the pandemic, followed by a gradual decline stabilizing around 3.2% in recent months. These visualizations will be incorporated into our final dashboard to provide clear insights into employment trends and support data-driven decision-making

## 2025-03-09: Brainstorm Dashboard Milestone

### Brainstorm Dashboard (2507366)
Project Progress As the sole member of the team, I finalized the exploratory data analysis (EDA) dashboard, which visualizes labor trends in Iowa and neighboring Midwest states from 2000–2024 using BLS data. This dashboard includes interactive widgets for timeline and state selection, as well as cluster charts built with seaborn to highlight patterns in the workforce. Though the RAND K–5 education dataset has proven difficult to extract insights from, I am considering switching to a more straightforward education-related data source.

### Finalize Data Models (2507368)
Finalize Data Models My current modeling approach focuses on forecasting labor trends three to five years out using time-series techniques, with 95% confidence intervals for each prediction. The response variable is the labor demand within specific industries, which I plan to refine by incorporating new variables like unemployment rates or regional housing costs for improved accuracy. I will continue testing hyperparameters and evaluating different models to find the most robust method for predicting future workforce needs. 

### Enhance Predictive Insights (2507371)
I plan to introduce additional variables—such as local economic indicators and housing data—to bolster model accuracy and capture broader market forces impacting the labor supply. Outliers will be carefully assessed to determine whether they reflect true anomalies or data inconsistencies, ensuring they don’t unduly skew the predictions. Ultimately, my goal is to present these refined forecasts in the final dashboard alongside feature importance metrics, helping stakeholders easily interpret and act on the predictive findings.

### Project Progress (2507372)
The main purpose of the dashboard is to provide new graduates and government planners with an accessible view of historical and potential future labor trends in the Midwest. It features a home page with general takeaways and an Ollama-retrained chatbot for answering user queries about the data and methodology. This design allows users to seamlessly transition from exploring EDA insights to running forecasting scenarios for individual states or for the region as a whole. 

## 2025-03-16: Finalize Data Models Milestone

### Dashboard Sketch (2510671)
I am building a four-page dashboard: an Overview/Landing page (with BLUF conclusions), an EDA page (showing interactive labor insights), a Forecasting page (comparing Iowa to other Midwest states), and a Data page (raw data plus basic visualizations). A left-column Ollama chatbot on each page answers user queries in real time. This layout allows users to toggle states, explore trends, and dive into the raw data as needed.

### Spring Break Plans (2510677)
I plan to keep working on the dashboard over spring break, integrating the forecast model and enhancing chatbot functionality. By the end of the break, I aim to have a solid prototype ready for feedback, with all four pages linked and interactive. This will position me to finalize the project’s modeling and visualization elements soon after.

### Finalize Data Models (2510670)
I am refining two models: a time-series forecast using BLS data (2000–2024) to predict labor trends, and a location-suitability model that merges labor with housing data. Key assumptions include stationarity in the time-series and consistent data coverage across states. Both models aim to clarify which regions and job types are most promising in the Midwest.

### Project Progress (2510672)
As a single-person team, I have finished EDA for the BLS dataset and am now focusing on time-series modeling and housing data integration. Aligning data sources and timelines has been a challenge, but I am resolving it through careful cleaning and validation. Once done, these models will feed directly into the final dashboard.

## Finalize Dashboard Sketch (2513594)
A streamlined textual sketch of the dashboard was created to clearly illustrate the essential components, user interactions, and outputs. The dashboard structure below provides a concise overview of the layout, tabs, and interactive elements, serving as a clear reference for ongoing development:
```yaml
Labor Market Dashboard:
  Navbar: Dashboard Title
  Tabs:
    - EDA / Overview:
        User Inputs:
          - State Selector
          - Year Range Slider
        Outputs:
          - Interactive Graphs (Labor Trends Over Time)
    - LFP Forecast:
        User Inputs:
          - Years Ahead Slider
        Outputs:
          - Iowa vs. Midwest Labor Force Participation Forecast Graph
    - Supersector Forecast:
        User Inputs:
          - Supersector Dropdown
          - Years Ahead Slider
        Outputs:
          - Employment Forecast Graph by State and Sector
  Visualization Area:
    Dynamic graphs updating based on user input selections
```
This structure clearly outlines the interaction flow and component relationships, ensuring intuitive navigation and usability for users.

## 2025-04-06: Minimum Viable Dashboard (2518204)

### Dashboard Access
Deploy and view the Minimum Viable Dashboard using:
```sh
./deploy.sh
```
Access the dashboard at [http://localhost:8050](http://localhost:8050).

### User Interaction
Users can select states and adjust timeframes, immediately updating the displayed labor statistics dynamically.

### Project Progress
This milestone introduces an integrated dashboard combining CES and LAUS data with interactive visualizations. Future steps include refining predictive models and incorporating additional economic indicators to further support informed decision-making.

## 2025-04-13: Add Models to Dashboard (2518745)

### Project Progress

The interactive dashboard integrates predictive models using recurrent neural networks (RNNs) with Long Short-Term Memory (LSTM) layers to forecast labor force participation rates (LFPR) and industry-specific employment trends (supersector forecasts) for Iowa and the Midwest region. The response variables are the labor force participation rate for the LFPR model and sector-specific employment numbers for the supersector forecasts. The explanatory variables include historical monthly data from the CES and LAUS datasets provided by the Bureau of Labor Statistics (BLS). The goal of these models is to accurately predict future labor market dynamics, enabling stakeholders to better understand and prepare for shifts in employment trends.

Currently, the models have been successfully integrated into the dashboard, deployed locally using Docker and a robust deployment script. All dependencies and model environments are fully containerized, ensuring reproducibility and ease of deployment. The Docker deployment instructions and scripts have been clearly documented, allowing users to either deploy automatically using the provided script or manually following detailed Docker commands.

### Deployment Methods

#### Automated Deployment
The recommended method is using the provided deployment script:

```bash
chmod +x deploy.sh
./deploy.sh
```

This script handles Docker installation, image building, container deployment, and verification. After successful deployment, the dashboard will be accessible at [http://localhost:8050](http://localhost:8050).

#### Manual Deployment
If you prefer not to use the automated script, follow these manual deployment instructions:

1. Ensure Docker is installed and running on your system.
2. Build the Docker image:

```bash
docker build -t labor-dashboard .
```

3. Run the Docker container:

```bash
docker run -d -p 8050:8050 --name labor_dashboard labor-dashboard
```

The dashboard will then be available at [http://localhost:8050](http://localhost:8050).

## 2025-04-20: Dashboard Peer Review (2520772)
Contextual information, including an "About" page clearly stating the dashboard's purpose—providing clear, actionable insights into labor trends in the Midwest—and target audience (new graduates, policymakers, and labor analysts), has been started and will be completed by the end of the day on 2025-04-27. Definitions for technical terms and acronyms, such as CES, LAUS, LFPR, and LSTM, will be included to ensure clarity and ease of understanding for first-time dashboard users.

## 2025-04-20: Finalize Dashboard Milestone (2522823)
Over the last week, substantial progress has been made on integrating an LLM to provide natural language insights, enhancing the accessibility of the dashboard. Additional visualizations were added, and significant refinements were implemented in UI/UX design, ensuring a more intuitive, clear, and engaging user experience. Current tasks include finalizing contextual information such as an 'About' page, clearly stating goals, defining jargon, and preparing the dashboard for peer review, expected to be completed by EOD 2025-04-27. 