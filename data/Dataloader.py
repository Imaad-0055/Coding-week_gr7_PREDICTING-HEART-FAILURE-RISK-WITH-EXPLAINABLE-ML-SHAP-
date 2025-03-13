def download_data():
  data_local_filename = 'heart_failure_clinical_records_dataset.csv'
    url = "https://archive.ics.uci.edu/dataset/519/heart%2Bfailure%2Bclinical%2Brecords"
    urllib.request.urlretrieve(url, data_local_filename)
    return delivery_data_local_filename
