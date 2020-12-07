from wwo_hist import retrieve_hist_data

import os
# os.chdir("./Data")

frequency=1
start_date = '11-DEC-2007'
end_date = '11-MAR-2008'
api_key = '2a7fe2227933484a873170950200411'
location_list = ['california']

hist_weather_data = retrieve_hist_data(api_key,
                                location_list,
                                start_date,
                                end_date,
                                frequency,
                                location_label = False,
                                export_csv = True,
                                store_df = True)