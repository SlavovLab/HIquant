# HI*quant*
## Quantifying homologous proteins and proteoforms


![GitHub](https://img.shields.io/github/license/SlavovLab/DO-MS.svg)

* [Slavov Lab](https://slavovlab.net)
* [HIquant article](https://www.mcponline.org/article/S1535-9476(20)34108-6/fulltext)




The Python code in this repository implements the HI*quant*  algorythm for quantifying homologous proteins and proteoforms from bottom up mass-spec data. It is described and tested in:

Malioutov D., Chen T., Jaffe J., Airoldi E., Budnik B., Slavov N. [Quantifying homologous proteins and proteoforms Molecular & Cellular Proteomics](https://doi.org/10.1074/mcp.TIR118.000947), Volume 18, Issue 1, 162 - 168 doi: [10.1074/mcp.TIR118.000947](https://doi.org/10.1074/mcp.TIR118.000947)


## Running this code:
This python code is tested on Max OS X 10.15.6 and using Python 3.8.5.

The script can be called from the command line and has 3 required arguments:
1. The input file (see exampleInput.txt)
2. Output file 1 - this has the inferred proteoform abundance
3. Output file 2 - this has the confidence and error estimates for the inference

An example command line call is:
> ed$ python3 ~/Documents/GitHub/HIquant/HIquant_run.py '/Users/ed/Documents/GitHub/HIquant/TestData_fromMCP.txt' '/Users/ed/Documents/GitHub/HIquant/Results/MCP_Output_1.csv' '/Users/ed/Documents/GitHub/HIquant/Results/MCP_Output_2.csv'



## About the project

For more information, contact [Slavov Laboratory](https://slavovlab.net) or directly [Prof. Nikolai Slavov](https://coe.northeastern.edu/people/slavov-nikolai/)

### License

The HI*quant* code is distributed by an [MIT license](https://github.com/SlavovLab/DO-MS/blob/master/LICENSE).

### Contributing

Please feel free to contribute to this project by opening an issue or pull request.
