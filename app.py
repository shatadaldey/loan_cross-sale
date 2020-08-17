from flask import Flask, Markup, render_template
import os
from google.cloud import bigquery
from flask import request

# Initialize flask application
app_flask = Flask(__name__,static_url_path="/", 
    static_folder="./templates")

# Define API route
@app_flask.route("/")
def root():
    return app_flask.send_static_file("front_end_input.html")


@app_flask.route("/cin")
def fetch_cin_details(methods=['GET']):

    # Fetch query parameter
    query_params = request.args
    cin = query_params["cin"]

    # Fetch details from DB
    # 1. Establish credentials
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/shatadaldey/Desktop/pandas/my_key.json"

    # 2. Establish BQ client
    client = bigquery.Client()

    # 3. Query
    sql_query = """
        SELECT 
            A.*,
            "Mr. Sherlock Holmes" as name,
            "95%" as probability 
            
        FROM 
            `deft-effect-282902.loan_takeup.loan_base` as A
        WHERE 
            A.ID = {cin}
    """

    # 4. Fetch results
    result = list(client.query(sql_query.format(cin = cin)))

    print(result)
    # Return response to 
    
    return render_template("front_end.html",cin = result[0]['ID'],name = result[0]['name'],age = result[0]['Age'],
        experience = result[0]['Experience'],income = result[0]['Income'],zip = result[0]['ZIP_Code'],
        family = result[0]['Family'],mortgage = result[0]['Mortgage'],securities = result[0]['Securities_Account'],
        online = result[0]['Online'],creditcard = result[0]['CreditCard'],ccavg = result[0]['CCAvg'],
        probability = result[0]['probability'])
    
app_flask.run(host='0.0.0.0', port=8012)