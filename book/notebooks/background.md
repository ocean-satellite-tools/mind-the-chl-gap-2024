# Background

## Data Sources

Most of our remote sensing data are sourced from the Copernicus program. Copernicus program is an European flagship program providing reliable and open satellite-based imagery, models, and in situ (nonspace) data, and is a coordinated effort between many organizations, including the European Commission, the European Space Agency, the European Centre for Medium-Range Weather Forecasts (ECMWF), and European Union Agencies (Skoda & Adam, 2020). Furthermore, we also blended data from the National Aeronautics and Space Administration (NASA) EarthData and the National Centers for Environmental Information (NCEI)’s databases. 

### ERA5

The primary data source we used in this assembled dataset product is the Copernicus ERA5 Global Reanalysis, the fifth generation of an atmospheric reanalysis project from Copernicus and ECMWF (Hersbach et al., 2020). As an ongoing project, once completed, it will embody a “detailed record of the global atmosphere, land surface, and ocean waves from 1950 onwards” (Hersbach et al., 2020). However, only 1979 data onwards is currently publicly available for download. The dataset’s high temporal and spatial resolution and ranges enables consistent, detailed, and concise detection and prediction tasks in our project. Comparing its performance to another popular reanalysis product, MERRA-2 by NASA, it overpowers the other in all aspects: resolution, time coverage, and accuracy (Olauson, 2018). Given its powerful capabilities, We collected five variables from this source, namely sea surface temperature, two-meter-high from surface level atmospheric temperature, and horizontal and vertical surface wind velocities. All of them are currently accessible using Amazon Simple Storage Service (S3) and updated regularly (Hersbach et al., 2020).

#### Sea surface temperature

The temperature of the ocean’s surface, known as Sea Sur- face Temperature (SST), serves as a crucial gauge for assessing the Earth’s climate system (Reynolds et al., 2002). Consequently, having precise information about SST is vital for monitoring, researching, and forecasting climate patterns. This also rings true in the case of coastal upwelling, as seasonally variable low SST compared to the average temperature at the same latitude may partially indicate an upwelling zone (Benazzouz et al., 2014; Alvarez et al., 2010; Izumo et al., 2008).

#### Atmospheric temperature

Our choice to include atmospheric temperature was more or less a secondary addition. It does not have a instant, direct correlation on upwelling as SST; however, researches showed that its strength may be influenced by preceding air temperature records in the preceding seasons before the upwelling season (Sun et al., 2022). This complements our project, especially on prediction tasks.

#### Vertical and horizontal components of the surface wind

One of the prominent coastal upwelling characteristics is a parallel wind direction along the coast (Lill, 1978). Strong winds can also cool the ocean surface, promoting the conditions and occurrence of upwelling (Kim et al., 2023). Longshore surface wind is also a major factor in mass transport and upwelling intensity, especially in the case of wind stress (Nigam et al., 2018). In their paper, Nigam et al. (2018) mentioned a formula, τy = ρd|W|v, addressing the relationship between the meridional wind stress (τy ), wind speed (W ), and v (meridional wind component, or more informally known as the vertical component of the wind).

### Global Ocean Physics Reanalysis

The Global Ocean Physics Reanalysis (product code name GLORYS12V1) is the first version of a Copernicus Marine Environment Monitoring Service (CMEMS)’s reanalysis product covering an 5000m elevation range from 1993, with models used for reanalysis simi- lar to ERA5’s (European Union-Copernicus Marine Service, 2018). It covers a wide range of variables such as sea surface temperature, salinity, or mixed layer thickness that is relevant to our project. The latter two are extracted from this data source into our composite dataset, whose data is resampled and interpolated using arithmetic mean to fit in our collection resolution from the original 1/12◦ horizontal resolution (Jean-Michel et al., 2021).

#### Salinity

We used in-situ salinity covered at the most shallow point on the elevation range at 0.49 meters below the surface with bias reduced using 3D-VAR scheme correction (Jean- Michel et al., 2021). During coastal upwelling in the West Indian Ocean, subsurface water, which is more salined, rises up to the surface, bringing additional salinity to the surface water which is diluted by heavy precipitation in the monsoon season (Awo et al., 2022; Sreenath et al., 2022). Note that this is not always the case. For example, the Coast of Bengal, where river discharges and rainfall combined created a thick barrier and shallow mixed layer, preventing salinity to reach the surface (Vinayachandran et al., 2002; Lahiri and Vissa, 2022). Despite that, coastal upwelling still occurs in the area with the aid from the impact of seasonal monsoon (Ray et al., 2022). Note that the Coast of Bengal is not covered in our region of interest, so any concerns about the outcome of this variable is to not confused with the inherit nature of the area.

#### Mixed Layer Thickness Defined by Sigma T

During our search of possible variables to add in our blended dataset product, we came accross Bessa et al. (2019)’s paper in which the author discovered that there is a month by month variability in the mixed layer depth that coincides with the trend of upwelling and downwelling in the Moroccan Atlantic coast. Further research found speculation towards the relationship between the two variables in the Arabian Sea of our region of interest. In Copernicus terms, it is defined as “the ocean depth at which sigma-theta has increase by 0.01km/m3 relative to the near surface value at 10m depth” (Mladek, 2019). Sigma-delta is defined as “water potential density (the density when moved adiabatically to a reference pressure) of water having the same temperature and salinity, minus 1000 kg m-3” (CF Conventions, n.d.).

### Global Ocean Colour (Copernicus-GlobColour)

The Ocean Colour Thematic Assembly Centre (OCTAC) currently provide global and regional high quality data products used by mostly intergovermnetal bodies and EU insti- tuitons, focusing on mostly ecosystem model assimilation and validation (European Union- Copernicus Marine Service, 2022). The Global Ocean Colour dataset (code name OCEAN-COLOUR GLO BGC L4 MY 009 104), based on data validated using the GlobColour processor owned by Copernicus, output daily and monthly data on a 4km × 4km spatial resolution covering data from September 1997. This wide temporal extent provide our machine learning tasks with plentiful data to train and validate with, and better cover the exten- sive range that the ERA5 variables do comparing to comparable datasets such as NASA’s MODIS-Aqua (NASA Ocean Biology Processing Group, 2015).

#### Gapfree chlorophyll-a concentration and uncertainty (Level 4)

When upwelling happens, we can also observe an increase in nutrient-rich near-surface waters (Benazzouz et al., 2014), in which wind (convective) mixing and upwad nutrient fluxes to the subsurface zone leads to phyto- plankton bloom and chlorophyll-a production (Lahiri and Vissa, 2022; Brock et al., 1991). Cold waters being mixed rise above the thermocline to the surface, promoting the growth of species in unfavorable environments, which also contains chlorophyll-a (Alvarez et al., 2010). Therefore, high chlorophyll-a concentrations at the sea surface level can imply whether up- welling is happening. However, based on empirical data processing, we noticed that there is a lot of missing data, which is also confirmed in Park et al. (2020)’s paper. Many factors are weighed in, including phytoplankton’ photosyntehtic parameters, seawater optical complexity, or flog and clouds peristence due to seasonal monsoon, leading to rain. S. Yu et al. (2022)’s dataset, while addressed this issue, does not issue a daily resolution dataset that we need to incorporate into our product.

### Global Ocean Gridded L4 Sea Surface Heights And Derived Variables Reprocessed 1993 Ongoing

The dataset (code name SEALEVEL GLOB PHY L4 MY 008 047) is part of the Sea Level Altimeter product family, providing multiyear records of sea surface height anomalies and derived variables for the whole global ocean (European Union-Copernicus Marine Ser- vice, 2021). It has a 0.25◦ × 0.25◦ spatial resolution, similar to the standard ERA5’s that we based on, and covered an impressive temporal range from 1993 to 2022.

#### Sea surface height above geoid and sea surface height above level (sea level height anomaly)

Wind components may be a good starting point to investigate the state of coastal upwelling, but comparing to sea surface height anomaly, the latter is more directly involved, through changes in the the thermocline and isothermal layer depth changes (Zhang and Mochizuki, 2022; L. Yu, 2003). We have been searching to no avail for public D20 (20◦ isothermal layer depth) or D20 anomaly dataset that satisfies our resolution and coverage requirements. Zhang and Mochizuki (2022) calculated this variable using monthly ocean temp data, but no specific formulae/method is disclosed. We resort to sea surface height anomalies data as the variable is somewhat related to the former variable itself, albeit not completely linear due to complex involvements of other variables in our blended data product, like salinity and temperature (L. Yu, 2003).

### Ocean Surface Current Analyses Real-time (OSCAR) Version 2.0

OSCAR uses sea surface height anomaly from the above dataset to compute and analyze the surface current components we are using in our product (Dohan, 2021). In fact, the Global Ocean Gridded L4 dataset also included those variables; however, since we discovered OSCAR first, we have already processed this data before knowing the existence of the other one.

#### Vertical and horizontal components of surface currents and geostrophic surface currents

There are many papers discussing the correlation of currents, especially surface currents, and coastal upwelling (Lentz and Chapman, 2004; Nigam et al., 2018). Rao et al. (2008) mentioned how certain surface current directions (in the case of West Indian Ocean, southernly) would be favourable to the phenomenon. Somalia currents, the surface current in the western Arabian Sea of our interested region, has positive influences to upwelling (Schott et al., 1990). Geostrophic currents, defined as currents balanced between the Coriolis effect forces and gradient pressure forces, are found to contribute greatly to upwelling in systems such as the California Current Systems or Arafura Sea in Indonesia (Ding et al., 2021; Umaroh et al., 2017). By including both non-geostrophic and geostrophic currents, we hope to identify how using the two difference types of currents may impact the model’s productivity, indicating whether it is better to use one variable or the other, or both.

### SRTM30+ Global 1-km Digital Elevation Model Version 11: Bathymetry

A product from the Scripts Institution of Oceanography of the University of California San Diego, the SRTM30 Plus’s bathymetry data is based on a satellite-gravity model with a heavily calibrated gravity-to-topography ratio from over two hundred millions soundings (Becker et al., 2009). Given the ultrahigh resolution and precision of the map, we subsetted and provided two bathymetry maps covering our region of interest with different resolu- tions, one with standard 0.25◦ × 0.25◦, and another one with finer resolution using as a basemap/map background due to any graphing of the variables. The former one is included in our final assembled product, and therefore easier to extract and use within the scope of the dataset (which means, we can extract the bathymetry value at a predetermined point right away without having to interpolate from an outside source import).

#### Bathymetry

There have been multiple studies on the relationship between coastal up- welling, such as Garvine (1973) or Lill (1978). They proposed that the ocean depth (and inherently the ocean floor shape or bathymetry) can determine the motion of the subsurface return flow, one of the two principle layers of the upwelling motion of homogeneous water. The topographic variation also influences the water circulation, such as disrupting or redi- recting flows along coasts, weakening them, or increasing their strength to enhance mixing (Pitcher et al., 2010).

## Zarr Format

Zarr, short for “zarr array,” is a storage format specifically designed for efficient, scalable, and parallelizable access to multi-dimensional typed arrays (tensors), making it an ideal choice for managing Earth observation data, especially for cloud-hosted data (cite gowen (Moore et al., 2023). It is developed as an open-source project by using referencible associated metadata and binary data called “chunks” stored in “formatted” directories, it leverages modern data storage technologies, such as chunked and compressed arrays, to optimize storage and retrieval, and reducing access latency.

Gobet and Lane (2012), in their “Encyclopedia of the Sciences of Learning,” referred to ”chunk” as a high-level, “meaningful unit of information built from smaller pieces of in- formation.” Therefore, chunking is the mechanism, or the process of creating those objects. While Miller (1956) introduced the term in his 1956 paper as a way of “breaking up long strings of information into units”, it is not until de Groot, in his study of chess experts and observed this phenomenon in nature, developed the concept as a mean of transforming information after observing their capabilities to retain precise brief information on presented chess positions (Frey & Adesman, 1976). Further research explores the concept as a mea- surement of human cognitive system, and how chunking can be interpreted as an automatic learning processing to “recode information in a more efficient way” (Gobet & Lane, 2012). This definition can somewhat explain the overall concept of chunking in informatics (the field of information), given how it essential divides large datasets into manageable blocks (comparable to “recode information” or more technically speaking, “encoding”). In the Xar- ray library we mostly used in this project, chunking, by default, is mostly handled by the underlying Dask arrays, where NumPy (or NumPy-like) arrays are broken down and rear- ranged to speed up certain algorithms (Hoyer et al., 2022; Dask Development Team, 2016). Resulted chunks then are mapped to metadata files in JSON (JavaScript Object Notation) so that the overall structure and content description can be read quickly (Vance et al., 2019; Miles et al., 2023).

Therefore, understanding chunking, as a mechanism, is crucial in understanding how Zarr works so that it can provide seamless access to very large datasets, often in the terabyte and petabyte range. Additionally, its support for chunked storage, data compression over deduplication, and lightweight metadata management optimizes storage, enables parallel processing, and minimizes access latency, all of which are critical when dealing with extensive datasets which we will incorporate from the above data sources (Miles et al., 2023).

Zarr’s special structure makes it ideal to be stored on the cloud (Pollack, 2023). As modern data storage technologies and strategies, including distributed storage systems and cloud-based solutions, play a crucial role in efficiently managing and accessing large datasets, cloud-based datasets can ensure researchers that they can work with these data resources effectively.

By consolidating our data sources into a Zarr file, we aimed to create an easy-to-use analysis-ready data cube, bringing together multiple diverse datasets onto the same grid for seamless integration. This preprocessing step not only harmonized the data but also ensured that it adhered to a consistent spatial and temporal framework, facilitating efficient and comprehensive analyses of Earth’s oceans. This approach aligns with the broader shift in the remote-sensing community towards providing researchers with analysis-ready datasets, thus enabling more accessible and reproducible scientific investigations.

## Data Blending and Processing

### Computed Variables

To alleviate the usage of the product, we also pre-computed absolute speed and direction using vertical and horizontal components of our wind and (near-)surface current variables. For speed, we utilized a simple Pythagorean theorem approach where
$$
v = \sqrt{v_x^2 + v_y^2}
$$

with $v$ as vector-less speed, and $v_x$ and $v_y$ as horizontal and vertical velocity components, respectively. For direction, we utilized NumPy’s `arctan2()` function and then convert radians to degrees using their `rad2deg()` function, with the latter chosen as degrees are more commonly used in meteorology than radians due to its unique conventions comparing to the standard mathematical Cartesian plane’s (Harris et al., 2020; “Meteorological Concentions”, 2022).

### Interpolation

Due to the mismatched spatial grid configurations between our original datasets, we also applied linear interpolation on, both spatially and temporally, on all of our data variables so that they all follow an average daily temporal, 0.25◦ × 0.25◦ spatial grid. While temporal ranges may vary among data sources, with an insistence on available data of at least twenty- one years from 2000-2020, we enforced a spatial range of −12◦S → 32◦N, 42◦E → 102◦E across all variables.


