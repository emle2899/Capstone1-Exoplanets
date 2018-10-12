# Finding Other Worlds
### Identifying Exoplanets From Kepler Data
###### Proposal 1 | Emily Levis

#### Background
On March 6th, 2009, NASA launched the [Kepler Telescope](https://www.nasa.gov/mission_pages/kepler/launch/index.html) in order to detect exoplanets - planets orbiting stars outside of our own solar system. To do this, Kepler mapped the brightness of the stars over time, called the light curve. As an exoplanet passes in front of its star relative to Earth, called the transit, it causes a slight decrease in the star's light curve. This dip is called the transit depth, which is what Kepler uses to detect potential exoplanets. The data from Kepler can also be used to determine an exoplanet's size, orbital period, distance from star, and more. NASA has now [discovered](https://www.nasa.gov/kepler/discoveries) 2,327 confirmed exoplanets and 2,244 exoplanet candidates.

My goal is to find correlations between the data parameters and create predictive models from these correlations.

#### Project Goals
MVP:
* Test correlations between parameters
* Find distribution of parameters and parameter correlations
* Determine validity of models

MVP+:
* Create a predictive model that determines whether or not an object is an exoplanet candidate
* Classify known exoplanets by planetary type (Terrestrial Planet, Super-Earth, Gas Giant, Hot Jupiter, or Rogue Planet) and look at their relative frequencies
* Create a predictive model to determine if an exoplanet is potentially habitable based on its distance from its star

#### Data
I am using data from the [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=koi).

#### Preliminary Look at Data
Looking only at data marked with confirmed status (disposition)

![](./transit_depth.png)
![](./depth_vs_size.png)
