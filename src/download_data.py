import requests
import json
import pandas as pd
import os


def download_nhtsa(year=2021, make="FORD"):
    
    # URL to fetch models
    url_models = f"https://api.nhtsa.gov/products/vehicle/models?modelYear={year}&make={make}&issueType=c"

    # Base URL to fetch vehicle complaint data
    url_complaints = "https://api.nhtsa.gov/complaints/complaintsByVehicle"

    # List to store all data
    all_data = []

    try:
        # Make the request to get the models
        response = requests.get(url_models)
        response.raise_for_status()  # Raise exceptions for HTTP errors
        models_data = response.json()
        
        if "results" in models_data:  # Using the correct key
            # Get the list of models
            models = models_data["results"]
            
            print(f"{len(models)} models found for the year {year} and make {make}.")
            
            for model_info in models:
                model = model_info.get("model")
                if model:
                    print(f"Fetching data for model: {model}")
                    
                    # Make the request to get complaints data
                    response_complaints = requests.get(
                        f"{url_complaints}?make={make}&model={model}&modelYear={year}"
                    )
                    response_complaints.raise_for_status()
                    complaints_data = response_complaints.json()
                    
                    # Add to the complete data set
                    all_data.append({
                        "year": year,
                        "make": make,
                        "model": model,
                        "complaints": complaints_data.get("results", [])  # Using the correct key
                    })
        else:
            print("No models found for the provided parameters.")
        
        # Save the data to a JSON file
        json_filename = f"data/raw/full_data_{year}_{make}.json"
        os.makedirs(os.path.dirname(json_filename), exist_ok=True)  # Create directories if they don't exist
        with open(json_filename, "w", encoding="utf-8") as json_file:
            json.dump(all_data, json_file, ensure_ascii=False, indent=4)
        print(f"All data saved to: {json_filename}")
        
        # Convert the data to a DataFrame and save it as CSV
        records = []
        for entry in all_data:
            year = entry["year"]
            make = entry["make"]
            model = entry["model"]
            for complaint in entry["complaints"]:
                complaint_data = {
                    "Year": year,
                    "Make": make,
                    "Model": model,
                    **complaint,  # Add all complaint fields
                }
                records.append(complaint_data)
        
        # Create the DataFrame
        df = pd.DataFrame(records)

        # Save the data to a CSV file
        csv_filename = f"data/raw/full_data_{year}_{make}.csv"
        df.to_csv(csv_filename, index=False, encoding="utf-8")
        print(f"All data saved to: {csv_filename}")

    except requests.exceptions.RequestException as e:
        print(f"Error making the request: {e}")
