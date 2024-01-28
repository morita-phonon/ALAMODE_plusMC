# ALAMODE_plusMC
implementation of Monte-Carlo integration with important sampling[1,2]
original source : ALAMODE-1.4.2 (https://github.com/ttadano/alamode/tree/master)[3]

Input style:
put new variables in &general field: MC_METHOD and (SAMPLE or SAMPLE_DENSITY) is required
* use MC_METHOD to apply MC integration
  MC_METHOD = 0 # MC integration is not active, default
  MC_METHOD = 1 # simple MC integration (without important sampling) !! not implemented yet
  MC_METHOD = 2 # MC integration with important sampling (use SPS, recommended)
  MC_METHDO = 3 # MC integration with important sampling (use Bose-function weighted SPS)
* use SAMPLE to define the number of sample points (int nsample)
  SAMPLE = nsample
* use SAMPLE_DENSITY to define sample density in total scattering channels (double ratio, nsample/total)
  SAMPLE_DENSITY = ratio
