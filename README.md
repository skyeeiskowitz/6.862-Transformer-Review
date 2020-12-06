
# Transformer Architecture with Exogonous Inputs
Currently, we have set up benchmarking for TRMF and DeepAR on two data sets. 

## Datasets: 

#### Weather data
* Maintained by Iowa State University's [IEM](https://mesonet.agron.iastate.edu/request/download.phtml?network=ILASOS)


#### Solar Power Plant Data 
This [dataset](https://www.nrel.gov/grid/solar-power-data.html) contains the solar power production from 137 plants in Alabama in 2006 sampled every hour. 

#### Electricity Data 
This [dataset](https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014) consists of the electricity consumption of 370 customers in hour-long intervals from January 2011 to September of 2014
##Benchmarking
 We have included TRMF from [this repository](https://github.com/SemenovAlex/trmf) and DeepAR from [this repository](https://github.com/zhykoties/TimeSeries) in our code and left these untouched, wrapping our code around their library.
 ### To run TRMF benchmark:

1. run TRMF_Benchmark/Benchmark_electricity.py or TRMF_Benchmark/Benchmark_solar.py
2. Reuslts are printed in the console or the list of parameters for tuning are outputted to TRMF_Benchmark/best_params_history.txt

### To run DeepAR benchmark:

1. run DeepAR Benchmark/Benchmark_DeepAR_elct.py or DeepAR Benchmark/Benchmark_DeepAR_solar.py
2. Results will be plotted in DeepAR/model/figures

## Transformer Code
The package  `transformer_pkg` is modified from Li et al.'s code, described in this [paper](https://arxiv.org/abs/1907.00235)
Li, S., Jin, X., Xuan, Y., Zhou, X., Chen, W., Wang, Y.-X., and Yan, X.  Enhancing the Locality and Breakingthe Memory Bottleneck of Transformer on Time SeriesForecasting.  (NeurIPS), 2019.  ISSN 10495258




This code is not public and was taken out for the peer review portion of this project