# STOCK PRICE FORECASTING

## Stock Price Forecasting Using Linear Regression

Stock price forecasting is a method used to predict the future value of a company's stock based on historical data. One common approach to this problem is using linear regression, a statistical method that models the relationship between a dependent variable and one or more independent variables.

In the context of stock price forecasting, the dependent variable is the stock price, and the independent variables could include factors such as historical prices, trading volume, and other financial indicators. Linear regression aims to find the best-fitting line through the data points that minimizes the sum of the squared differences between the observed values and the values predicted by the line.

By training a linear regression model on historical stock price data, we can make predictions about future stock prices. However, it's important to note that stock prices are influenced by a multitude of factors, and linear regression is a relatively simple model that may not capture all the complexities of the stock market. Therefore, while linear regression can provide some insights, it should be used with caution and in conjunction with other methods and analyses.

## Data Source and Web Scraping

We are obtaining the stock price data from Yahoo Finance. The data is collected using a web scraping model that extracts historical stock prices and other relevant financial information from the Yahoo Finance website. This data is then used to train and test the linear regression model for forecasting future stock prices.

The web scraping process involves sending HTTP requests to the Yahoo Finance website, parsing the HTML content, and extracting the necessary data fields. This approach ensures that we have up-to-date and accurate data for our analysis.

## Backend Framework

We are using Django, a high-level Python web framework, for the backend of this project. Django allows us to build robust and scalable web applications quickly and efficiently. It follows the model-template-views (MTV) architectural pattern and provides a range of built-in features such as an ORM, authentication, and an admin interface.With Django set up, you can now start building the backend logic for your stock price forecasting application.

## Description
This project folder contains the following files:

1. `main.py`: The main script to run the project.
2. `requirements.txt`: A file listing all the dependencies required to run the project.
3. `README.md`: This file, providing an overview and instructions.
4. `data/`: A directory containing the dataset used in the project.
5. `models/`: A directory containing pre-trained models or saved models.

## How to Run the Project

Follow these steps to set up and run the project:

1. **Clone the repository**:
    ```sh
    git clone https://github.com/Harisaikumarbathula/FINAL-YEAR-PROJECT.git
    cd FINAL YEAR PROJECT
    ```

2. **Set up a virtual environment** (optional but recommended):
    ```sh
    python -m venv venv

    ```sh
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
    ```sh
    source venv/bin/activate  # On Mac and Linux
    ```


3. **Install the required dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

4. **Run database migrations**:
    ```sh
    python manage.py migrate

5. **Run the main script**:
    ```sh
    python manage.py runserver
    ```


## Additional Information
- Ensure you have Python installed (preferably version 3.6 or higher).
- Make sure to activate the virtual environment each time you work on the project to avoid dependency conflicts.
