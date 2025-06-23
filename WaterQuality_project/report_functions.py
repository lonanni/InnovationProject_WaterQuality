
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.gaussian_process.kernels import (RationalQuadratic, Exponentiation, RBF, ConstantKernel)

from sklearn.gaussian_process import GaussianProcessRegressor
from typing import List, Dict, Union, Optional, Any
from shapely.geometry import Point
import geopandas as gpd
import fiona
from geopandas import GeoDataFrame
from shapely.geometry import Polygon
from shapely.geometry import MultiPoint, Polygon
import shapely.ops as ops
import shapely
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class WaterQualityProcessor:
    """
    A collection of utility methods for processing and transforming water quality monitoring datasets.

    This class provides static methods for:
    - Pivoting water quality measurements into a wide format
    - Filtering frequently observed determinants
    - Outlier removal using quantile thresholds
    - Adding seasonal and temporal features
    - Filtering the dataset to include only the most frequently sampled water types

    The input DataFrames are expected to follow a consistent schema, such as those found in 
    water quality monitoring data (e.g., EA, DEFRA, or Water Framework Directive datasets).
    """

    def __init__(self, df):
        """
        Initialize the processor with a raw water quality DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            Input data containing water quality measurements.
        """
        self.df = df.copy()

    @staticmethod
    def pivot(df):
        """
        Filters and pivots dataset to retain only frequently observed determinants.

        This function filters out determinants (i.e., measurement types) that appear in fewer than 25% of the total 
        unique sampling times. The remaining data is pivoted to have determinants as columns and 
        sampling time and location information as the index.

        Parameters
        ----------
        df : pandas.DataFrame
            The input DataFrame containing water quality data. Expected to include at least the following columns:
            - 'determinand.definition'
            - 'sample.sampleDateTime'
            - 'result'
            - 'sample.samplingPoint.easting'
            - 'sample.samplingPoint.northing'
            - 'sample.sampledMaterialType.label'
            - '@id'

        Returns
        -------
        pandas.DataFrame
            A pivoted DataFrame where each row represents a sampling event and location, and each column 
            corresponds to a frequently measured determinant.
        """
        # Count how many times each determinand appears
        det_counts = df['determinand.definition'].value_counts()

        # Compute threshold
        threshold = len(set(df['sample.sampleDateTime'])) / 4

        # Filter determinands that exceed threshold
        frequent_determinants = det_counts[det_counts > threshold].index.tolist()

        df_sub = df[df['determinand.definition'].isin(frequent_determinants)]
        piv = df_sub.pivot(index=['sample.sampleDateTime', "sample.samplingPoint.easting","sample.samplingPoint.northing",  
                          "sample.sampledMaterialType.label", '@id'],\
                   columns=['determinand.definition'], values='result').reset_index()
        piv.columns.name = None

        return piv

    @staticmethod
    def quantiles(df):
        """
        Filters out extreme outliers in float64 columns by setting values outside the 1st and 99th percentiles to NaN.

        This function selects all `float64` columns in the input DataFrame and replaces values that fall 
        below the 1st percentile or above the 99th percentile with `NaN`.

        Parameters
        ----------
        df : pandas.DataFrame
            The input DataFrame, which may contain multiple data types. Only columns of type `float64` 
            are considered for quantile filtering.

        Returns
        -------
        pandas.DataFrame
            A new DataFrame where all `float64` columns have had extreme values (below the 1st percentile 
            or above the 99th percentile) replaced with `NaN`. Non-float columns remain unchanged.

        Notes
        -----
        - The 1st and 99th percentiles are computed per column.
        - This operation is non-destructive: the original DataFrame is not modified.
        - Useful for outlier mitigation in preprocessing pipelines.
        """
        quantities = df.select_dtypes(include=[np.float64]).columns
        q_low = df[quantities].quantile(0.01)
        q_hi = df[quantities].quantile(0.99)

        df_return = df.copy()
        mask = (df[quantities] >= q_low) & (df[quantities] <= q_hi)
        df_return[quantities] = df[quantities].where(mask)
        return df_return

    @staticmethod
    def encoding_season(df):
        """
        Adds seasonal, year, and month information to a DataFrame based on sample date.

        It extracts the calendar date, year, and month from the 'sample.sampleDateTime' column, and assigns a binary 'season' label:
        - 0 for summer months (May to August)
        - 1 for winter months (September to April)

        Parameters
        ----------
        df : pandas.DataFrame
            Input DataFrame that must include a 'sample.sampleDateTime' column with datetime-like strings or objects.

        Returns
        -------
        pandas.DataFrame
            The input DataFrame with additional columns:
            - 'Date'   : Extracted calendar date (YYYY-MM-DD)
            - 'year'   : Integer year from the date
            - 'month'  : Integer month from the date
            - 'season' : 0 for May–August (summer), 1 for rest (winter)

        Notes
        -----
        - The 'sample.sampleDateTime' column will be converted to datetime if not already.
        - The output DataFrame is sorted chronologically by the 'Date' column.
        """
        df = df.copy()
        df["Date"] = pd.to_datetime(df['sample.sampleDateTime']).dt.date
        df["year"] = pd.to_datetime(df["Date"]).dt.year
        df["month"] = pd.to_datetime(df["Date"]).dt.month
        df["season"] = np.where((df["month"] > 4) & (df["month"] < 9), 0, 1)
        return df.sort_values(by="Date", ascending=True)

    @staticmethod
    def filter_top_water_types(df):
        """
        Filters the dataset to retain only the top 4 most frequently sampled water material types.

        This function groups the input DataFrame by the sampled material type (e.g., types of water), 
        counts the number of samples per type, and selects the top 4 types with the highest sample count. 
        The DataFrame is then filtered to include only rows belonging to those top 4 types.

        Parameters
        ----------
        df : pandas.DataFrame
            The input DataFrame. Must contain the columns:
            - 'sample.sampledMaterialType.label'
            - 'sample.samplingPoint.easting'

        Returns
        -------
        pandas.DataFrame
            A filtered DataFrame containing only rows where the sampled material type is one of the 
            four most frequently observed types.
        """
        counts = df.groupby("sample.sampledMaterialType.label")["sample.samplingPoint.easting"].count()
        top_labels = counts.sort_values(ascending=False).head(5).index
        return df[df["sample.sampledMaterialType.label"].isin(top_labels)]

    def prepare(self):
        """
        Executes the full water quality processing pipeline using static utilities.

        Returns
        -------
        pandas.DataFrame
            Fully processed DataFrame.
        """
        df = self.df
        df = self.pivot(df)
        df = self.quantiles(df)
        df = self.encoding_season(df)
        df = self.filter_top_water_types(df)
        self.df = df
        return df

class WaterSystemPreprocessor:
    """
    A class to handle preprocessing of water system data including regions, rivers, 
    postcodes, and sample locations for geospatial analysis.
    """
    
    def __init__(self, 
                 regions_path: str,
                 district_path: str, 
                 river_dir: str,
                 df: pd.DataFrame,
                 labels: List[str],
                 water_type_map: Dict[str, str]):
        """
        Initialize the preprocessor with data paths and core datasets.
        
        Args:
            regions_path: Path to regions data
            district_path: Path to district data
            river_dir: Directory containing river data
            df: Main dataframe with sample data
            labels: Labels for water type classification
            water_type_map: Mapping for water types
        """
        self.regions_path = regions_path
        self.district_path = district_path
        self.river_dir = river_dir
        self.df = df
        self.labels = labels
        self.water_type_map = water_type_map
        
        # Storage for all results
        self.region_data = None
        self.regions_ll = None
        self.regions_en = None
        self.UK_ocean = None
        self.rivers = None
        self.rivers_course = None
        self.river_in_region_of_interest = None
        self.df_by_type_sub = None
        self.gdf_by_type_sub = None
        self.district_of_interest = None
        self.gdf_loc_all = None
        self.locations_of_interest = None
        self.point_of_interest = None
        self.polygon = None
        self.polygon_results = None
        self.mask_location = None
        self.mask_location_course = None
        self.df_of_interest = None
        self.sea = None
        self.buffer_results = None
        self.gdf_location_studied = None
        self.gdf_location_studied_buff = None
        self.gdf_water = None
        self.gdf_water_buff = None
        self.united_water = None
        self.join_rivers = None
        self.united_water_course = None
        self.samples_by_system = None
        
    def preprocess(self, 
                   target_regions: List[str],
                   postcode_of_interest: str = "PO",
                   postcode_numbers: Optional[np.ndarray] = None) -> 'WaterSystemPreprocessor':
        """
        Execute the complete preprocessing pipeline.
        
        Args:
            target_regions: List of target regions to load
            postcode_of_interest: Postcode prefix (default: "PO")
            postcode_numbers: Array of postcode numbers (default: concatenated [0-12, 33-35])
            
        Returns:
            Self for method chaining
        """
        if postcode_numbers is None:
            postcode_numbers = np.concatenate([range(13), range(33, 36)])
            
        # Load selected regions
        self.region_data = load_selected_regions(self.regions_path, self.district_path, target_regions)
        self.regions_ll = self.region_data["regions_ll"]
        self.regions_en = self.region_data["regions_en"]
        
        # Generate UK ocean boundary
        self.UK_ocean = generate_uk_ocean_boundary(self.regions_path)
        
        # Load river data
        self.rivers, self.rivers_course = load_river_data(self.river_dir)
        
        # Get rivers in region of interest
        self.river_in_region_of_interest = gpd.sjoin(
            self.regions_ll[0], self.rivers_course, how='inner', predicate='contains'
        )
        
        # Extract water type geodata
        self.df_by_type_sub, self.gdf_by_type_sub = extract_water_type_geodata(
            self.df, self.labels, self.water_type_map
        )
        
        # Read UK district data
        uk_district = gpd.read_file(self.district_path)
        
        # Get sample locations by postcode
        self.district_of_interest, self.gdf_loc_all, self.locations_of_interest = get_sample_locations_by_postcode(
            self.df,
            uk_district,
            postcode_prefix=postcode_of_interest,
            postcode_numbers=postcode_numbers
        )
        
        # Select points of interest from all locations using the index values
        self.point_of_interest = self.gdf_loc_all.iloc[self.locations_of_interest.index.values]
        
        # Build convex hull polygon
        self.polygon = build_convex_hull_polygon(self.point_of_interest)
        
        # Get data within polygon
        self.polygon_results = get_data_within_polygon(
            polygon=self.polygon,
            rivers=self.rivers,
            rivers_course=self.rivers_course,
            UK_ocean=self.UK_ocean,
            gdf_loc_all=self.gdf_loc_all,
            df=self.df,
            labels=self.labels,
            water_type_map=self.water_type_map
        )
        
        # Extract results from polygon analysis
        self.mask_location = self.polygon_results["mask_location"]
        self.mask_location_course = self.polygon_results["mask_location_course"]
        self.df_of_interest = self.polygon_results["df_of_interest"]
        self.gdf_by_type_sub = self.polygon_results["gdf_by_type_sub"]
        self.sea = self.polygon_results["sea"]
        
        # Build buffers and perform join
        self.buffer_results = build_buffers_and_perform_join(
            df_of_interest=self.df_of_interest,
            rivers_course=self.rivers_course,
            mask_location_course=self.mask_location_course,
            sea=self.sea
        )
        
        # Extract buffer results
        self.gdf_location_studied = self.buffer_results["gdf_location_studied"]
        self.gdf_location_studied_buff = self.buffer_results["gdf_location_studied_buff"]
        self.gdf_water = self.buffer_results["gdf_water"]
        self.gdf_water_buff = self.buffer_results["gdf_water_buff"]
        self.united_water = self.buffer_results["united_water"]
        self.join_rivers = self.buffer_results["join_rivers"]
        
        # Create united water course
        self.united_water_course = gpd.GeoDataFrame(
            geometry=gpd.GeoSeries(self.gdf_water_buff.unary_union)
        ).explode()
        
        # Extract samples by water system
        self.samples_by_system = extract_samples_by_water_system(self.df_of_interest, self.join_rivers)
        
        return self
    
    def get_dataset_of_interest(self) -> Dict[str, Any]:
        """
        Returns all the key datasets and results from preprocessing.
        
        Returns:
            Dictionary containing all processed datasets
        """
        return {
            'region_data': self.region_data,
            'regions_ll': self.regions_ll,
            'regions_en': self.regions_en,
            'UK_ocean': self.UK_ocean,
            'rivers': self.rivers,
            'rivers_course': self.rivers_course,
            'river_in_region_of_interest': self.river_in_region_of_interest,
            'df_by_type_sub': self.df_by_type_sub,
            'gdf_by_type_sub': self.gdf_by_type_sub,
            'district_of_interest': self.district_of_interest,
            'gdf_loc_all': self.gdf_loc_all,
            'locations_of_interest': self.locations_of_interest,
            'point_of_interest': self.point_of_interest,
            'polygon': self.polygon,
            'mask_location': self.mask_location,
            'mask_location_course': self.mask_location_course,
            'df_of_interest': self.df_of_interest,
            'sea': self.sea,
            'gdf_location_studied': self.gdf_location_studied,
            'gdf_location_studied_buff': self.gdf_location_studied_buff,
            'gdf_water': self.gdf_water,
            'gdf_water_buff': self.gdf_water_buff,
            'united_water': self.united_water,
            'join_rivers': self.join_rivers,
            'united_water_course': self.united_water_course,
            'samples_by_system': self.samples_by_system
        }     

def load_selected_regions(regions_path: str, district_path: str, target_regions: List[str]) -> dict:
    """
    Load UK regions and districts shapefiles and extract target regions.

    Parameters:
        regions_path (str): Path to the UK regions shapefile.
        district_path (str): Path to the UK districts shapefile.
        target_regions (List[str]): Region names to extract.

    Returns:
        dict: Dictionary with filtered and reprojected data.
    """
    # Read shapefiles
    uk_regions = gpd.read_file(regions_path)
    uk_districts = gpd.read_file(district_path)

    # Filter and reproject
    regions_en = [uk_regions[uk_regions['RGN21NM'] == region] for region in target_regions]
    regions_ll = [region.to_crs(epsg=4326) for region in regions_en]

    return {
        "regions_en": regions_en,
        "regions_ll": regions_ll,
        "all_regions": uk_regions,
        "districts": uk_districts
    }


    
def plot_water_types_in_region(
    gdf_by_type: dict,
    regions_ll: list,
    rivers_course: gpd.GeoDataFrame,
    river_index: pd.Index,
    style_by_type: dict,
    figsize=(15, 15),
    facecolor='#FCF6F5FF'
):
    """
    Plots water types and rivers over a base region.

    Parameters:
    - gdf_by_type: dict of GeoDataFrames keyed by water type label
    - regions_ll: list of GeoDataFrames for base regions in lat/lon (EPSG:4326)
    - rivers_course: GeoDataFrame of river geometries
    - river_index: Index of rivers within region of interest (e.g., river_SE.index_right)
    - style_by_type: dict mapping water type labels to style configs
    - figsize: tuple for figure size
    - facecolor: background color of the figure
    """

    fig, ax = plt.subplots(facecolor=facecolor)
    fig.set_size_inches(*figsize)

    # Plot base region (assumes first region is the one to plot)
    regions_ll[0].plot(ax=ax, color='white', edgecolor='black')

    # Plot rivers
    rivers_course.loc[river_index].plot(
        ax=ax,
        color='#563FED',
        markersize=10,
        alpha=0.5,
        label="rivers"
    )

    # Plot water types
    for label, gdf in gdf_by_type.items():
        style = style_by_type.get(label, {})
        gdf.plot(
            ax=ax,
            color=style["color"],
            marker=style["marker"],
            edgecolor=style.get("edgecolor", "black"),
            label=style["label"]
        )

    # Final plot formatting
    plt.legend(fontsize=24)
    ax.set_xlabel("Latitude")
    ax.set_ylabel("Longitude")
    plt.tight_layout()
    plt.show()


def get_sample_locations_by_postcode(df, uk_district_gdf, postcode_prefix="PO", postcode_numbers=range(12)):
    """
    Returns water sample locations within postcode areas of interest.

    Parameters:
    - df: DataFrame with water sample point easting/northing columns.
    - uk_district_gdf: GeoDataFrame of UK postcode districts.
    - postcode_prefix: Prefix of the postcode to filter (e.g., "PO").
    - postcode_numbers: array number of postcodes' numbers.

    Returns:
    - locations_of_interest: GeoDataFrame of sample points within selected postcode areas.
    """
    # 1. Build list of postcodes of interest
    postcodes_of_interest = [f"{postcode_prefix}{i}" for i in postcode_numbers]

    # 2. Filter districts and project to WGS84
    district_of_interest = uk_district_gdf[uk_district_gdf["name"].isin(postcodes_of_interest)].to_crs(4326)

    # 3. Convert sample data to GeoDataFrame in WGS84
    gdf_loc_all = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(df["sample.samplingPoint.easting"], df["sample.samplingPoint.northing"]),
        crs=27700
    ).to_crs(4326)

    # 4. Spatial join: find sample points within selected districts
    locations_of_interest = gpd.sjoin(gdf_loc_all, district_of_interest, how="inner", predicate="within")

    return district_of_interest, gdf_loc_all, locations_of_interest

def generate_uk_ocean_boundary(
    regions_path,
    crs_output=4326
):
    """
    Generate an ocean boundary buffer around the UK landmass.

    Parameters:
        regions_path (str): Path to the UK regions shapefile.
        crs_output (int): EPSG code for output CRS (default: 4326).

    Returns:
        GeoDataFrame: Ocean area as a buffered geometry in output CRS.
    """
    # Load regions and project to metric CRS

    uk_regions = gpd.read_file(regions_path)

    uk_diss = uk_regions.dissolve()
    UK_region = uk_diss.convex_hull
    UK_ocean = UK_region.difference(uk_diss).to_crs(crs_output)

    return UK_ocean

def load_river_data(river_dir, crs='epsg:4326'):
    """
    Loads and reprojects UK river shapefiles.

    Parameters:
        river_dir (str): Path to the directory containing river shapefiles.
        crs (str): Target coordinate reference system (default: 'epsg:4326').

    Returns:
        tuple: GeoDataFrames for river nodes and river courses.
    """
    rivers = gpd.read_file(f"{river_dir}/HydroNode.shp").to_crs(crs)
    rivers_course = gpd.read_file(f"{river_dir}/WatercourseLink.shp").to_crs(crs)
    return rivers, rivers_course

def get_rivers_in_region(region_gdf, rivers_course_gdf):
    """
    Returns river courses that fall within the given region.

    Parameters:
        region_gdf (GeoDataFrame): Region geometry (typically a single region).
        rivers_course_gdf (GeoDataFrame): Full set of UK river courses.

    Returns:
        GeoDataFrame: Subset of rivers within the region.
    """
    return gpd.sjoin(region_gdf, rivers_course_gdf, how='inner', predicate='contains')

def extract_water_type_geodata(df, labels, water_type_map):
    """
    Filters input DataFrame and creates corresponding GeoDataFrames by water type.

    Parameters:
        df (DataFrame): Full dataset containing water type descriptions and coordinates.
        labels (list): Simplified labels like 'RIVER', 'OCEAN', etc.
        water_type_map (dict): Maps simplified labels to full descriptions.

    Returns:
        tuple:
            - df_by_type (dict): Pandas subsets of df.
            - gdf_by_type (dict): Corresponding GeoDataFrames.
    """
    df_by_type = {}
    gdf_by_type = {}
    for label in labels:
        water_name = water_type_map[label]
        subset = df[df['sample.sampledMaterialType.label'] == water_name]
        df_by_type[label] = subset
        gdf_by_type[label] = gpd.GeoDataFrame(
            subset,
            geometry=gpd.points_from_xy(subset["sample.samplingPoint.easting"],
                                        subset["sample.samplingPoint.northing"]),
            crs=27700
        ).to_crs(4326)
    return df_by_type, gdf_by_type


def build_convex_hull_polygon(gdf_points):
    """
    Creates a convex hull polygon that encloses all points in a GeoDataFrame.

    Parameters:
        gdf_points (GeoDataFrame): GeoDataFrame containing point geometries in WGS84 (EPSG:4326).

    Returns:
        GeoDataFrame: Single polygon (convex hull) GeoDataFrame with same CRS.
    """
    # Combine points into MultiPoint geometry
    multi_point = MultiPoint(gdf_points.geometry.tolist())

    # Get convex hull polygon
    convex_hull_polygon = multi_point.convex_hull

    # Return as GeoDataFrame
    return gpd.GeoDataFrame(geometry=[convex_hull_polygon], crs=gdf_points.crs)

def get_data_within_polygon(polygon, rivers, rivers_course, UK_ocean, gdf_loc_all, df, labels, water_type_map):
    """
    Extracts rivers, river courses, ocean, and sample data within the specified polygon.

    Parameters:
        polygon (GeoDataFrame): Polygon defining the region of interest (EPSG:4326).
        rivers (GeoDataFrame): River node geometries (EPSG:4326).
        rivers_course (GeoDataFrame): River course geometries (EPSG:4326).
        UK_ocean (GeoDataFrame): Ocean geometries (EPSG:4326).
        gdf_loc_all (GeoDataFrame): All sample location points (EPSG:4326).
        df (DataFrame): Original sample dataframe.
        labels (list): List of water type labels.
        water_type_map (dict): Maps water type labels to names.

    Returns:
        dict: {
            'mask_location': indices of river nodes within polygon,
            'mask_location_course': indices of river courses intersecting polygon,
            'df_of_interest': subset of df within polygon,
            'gdf_by_type_sub': dict of GeoDataFrames by water type within polygon,
            'sea': GeoDataFrame of ocean area within polygon
        }
    """
    # Ensure we're using a single shapely Polygon
    if isinstance(polygon, (gpd.GeoDataFrame, gpd.GeoSeries)):
        polygon_geom = polygon.geometry.iloc[0]  # extract the polygon
    else:
        polygon_geom = polygon  # assume it's already a shapely Polygon
    # Mask for river nodes within the polygon
    mask_location = rivers[rivers.geometry.within(polygon_geom)].index

    # Mask for river courses intersecting the polygon
    mask_location_course = rivers_course[rivers_course.geometry.intersects(polygon_geom)].index

    # Create single-row GeoDataFrames for overlay
    poly1 = gpd.GeoDataFrame(geometry=gpd.GeoSeries(polygon_geom), crs="EPSG:4326")
    poly2 = gpd.GeoDataFrame(geometry=gpd.GeoSeries(UK_ocean), crs="EPSG:4326")

    # Filter sample locations within the polygon
    df_of_interest = df.iloc[np.where(gdf_loc_all.within(polygon_geom))[0]]
    # Extract water-type grouped GeoDataFrames
    df_by_type_sub, gdf_by_type_sub = extract_water_type_geodata(df_of_interest, labels, water_type_map)


    # Clip ocean data to polygon
    sea = gpd.overlay(poly2, poly1, how='intersection')

    return {
        "df_of_interest": df_of_interest,
        "df_by_type_sub": df_by_type_sub,
        "gdf_by_type_sub": gdf_by_type_sub,
        "sea": sea,
        "mask_location": mask_location,
        "mask_location_course": mask_location_course
    }
def plot_polygon_context(
    district_of_interest,
    gdf_loc_all,
    locations_of_interest,
    polygon,
    rivers,
    rivers_course,
    mask_location,
    mask_location_course,
    gdf_by_type_sub,
    sea,
    style_by_type,
    figsize=(15, 15),
    background_color='#FCF6F5FF'
):
    """
    Plots rivers, sample locations, and classified water types within the polygon of interest.

    Parameters:
        district_of_interest (GeoDataFrame): District boundary.
        gdf_loc_all (GeoDataFrame): All sample location points.
        locations_of_interest (DataFrame): Filtered df for PO postcodes.
        polygon (GeoDataFrame): Convex hull polygon of selected locations.
        rivers (GeoDataFrame): River node geometries.
        rivers_course (GeoDataFrame): River course geometries.
        mask_location (ndarray): Indices of river nodes within polygon.
        mask_location_course (ndarray): Indices of river courses intersecting polygon.
        gdf_by_type_sub (dict): GeoDataFrames of points grouped by water type.
        sea (GeoDataFrame): Overlayed sea polygon clipped to area.
        style_by_type (dict): Dict mapping labels to plot styles.
        figsize (tuple): Size of the figure.
        background_color (str): Background color of the figure.
    """

    fig, ax = plt.subplots(facecolor=background_color)
    fig.set_size_inches(*figsize)

    # Base layers
    district_of_interest.plot(ax=ax, color='white', edgecolor='black')
    gdf_loc_all.iloc[locations_of_interest.index.values].plot(
        ax=ax, color="red", marker="o", label="Location"
    )
    polygon.plot(ax=ax, color='None', edgecolor='grey', linewidth=2)

    # Rivers
    rivers_course.iloc[mask_location_course].plot(
        ax=ax, color='#009ACD', markersize=10, alpha=0.5, label="rivers"
    )
    rivers.iloc[mask_location].plot(
        ax=ax, color='white', edgecolor="black", markersize=40, alpha=1,
        marker="X", label="river nodes"
    )

    # Water types
    for label, gdf in gdf_by_type_sub.items():
        style = style_by_type[label]
        gdf.plot(
            ax=ax,
            color=style["color"],
            marker=style["marker"],
            edgecolor=style.get("edgecolor", "black"),
            label=style["label"]
        )

    # Sea overlay
    sea.plot(ax=ax, color="blue", alpha=0.3)

    ax.legend(fontsize=25)
    ax.set_xlabel("Latitude")
    ax.set_ylabel("Longitude")
    ax.set_title("Sample Locations and Water Systems in Region of Interest", fontsize=18)
    plt.show()


def build_buffers_and_perform_join(df_of_interest, rivers_course, mask_location_course, sea):
    """
    Reproduces Block 11: creates buffers around locations and rivers,
    performs spatial join between locations and water systems.

    Parameters:
        df_of_interest (DataFrame): Filtered DataFrame with relevant sample points.
        rivers_course (GeoDataFrame): All river course geometries.
        mask_location_course (Index or list): Indices of river segments intersecting the polygon.
        sea (GeoDataFrame): Sea polygons GeoDataFrame.

    Returns:
        dict: Dictionary with:
            - gdf_location_studied
            - gdf_location_studied_buff
            - gdf_water
            - gdf_water_buff
            - united_water
            - join_rivers
    """
    # Step 1: Merge river geometries within region of interest
    united_river_course = shapely.ops.unary_union(rivers_course.iloc[mask_location_course].geometry)

    # Step 2: Convert unified river geometry to GeoDataFrame
    gdf_united_river_course = gpd.GeoDataFrame(
        geometry=gpd.GeoSeries(united_river_course)
    ).explode(ignore_index=True)

    # Step 3: Combine sea and river geometries
    gdf_water = pd.concat([sea, gdf_united_river_course])

    # Step 4: Build sample location GeoDataFrame from coordinates (EPSG:27700)
    gdf_location_studied = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(
            df_of_interest["sample.samplingPoint.easting"],
            df_of_interest["sample.samplingPoint.northing"]
        ),
        crs=27700
    ).to_crs(4326)

    # Step 5: Create buffers
    gdf_location_studied_buff = gdf_location_studied.to_crs(4326).buffer(0.005)
    gdf_water_buff = gdf_water.to_crs(4326).buffer(0.0003)

    # Step 6: Union water buffers into one geometry per system
    united_water = gpd.GeoDataFrame(
        geometry=gpd.GeoSeries(gdf_water_buff.unary_union)
    ).explode(ignore_index=True)

    # Step 7: Spatial join between location buffers and unified water systems
    join_rivers = gpd.sjoin(
        gpd.GeoDataFrame(geometry=gpd.GeoSeries(gdf_location_studied_buff), crs="EPSG:4326"),
        united_water,
        how='inner',
        predicate='intersects'
    )

    # Step 8: Return all components
    return {
        "gdf_location_studied": gdf_location_studied,
        "gdf_location_studied_buff": gpd.GeoDataFrame(geometry=gdf_location_studied_buff, crs="EPSG:4326"),
        "gdf_water": gdf_water,
        "gdf_water_buff": gdf_water_buff,
        "united_water": united_water,
        "join_rivers": join_rivers
    }
def plot_water_system_locations(
    district_gdf,
    polygon_gdf,
    united_water_gdf,
    sample_gdf,
    join_df,
    systems_to_plot=None,  # list of `index_right` or None to plot all
    figsize=(10, 10),
    background_color='#FCF6F5FF',
    marker_style="#66CD00",
    marker_symbol="s"
):
    """
    Plots sample locations for selected water systems within a district and polygon of interest.

    Parameters:
        district_gdf (GeoDataFrame): District boundary.
        polygon_gdf (GeoDataFrame): Convex hull or bounding polygon.
        united_water_gdf (GeoDataFrame): Water system geometries.
        sample_gdf (GeoDataFrame): Sample locations with buffer.
        join_df (DataFrame): Output of spatial join (e.g., gpd.sjoin).
        systems_to_plot (list): List of `index_right` values to filter from `join_df`. If None, plots all.
        figsize (tuple): Figure size.
        background_color (str): Background color of the plot.
        marker_style (str): Color for sample location markers.
        marker_symbol (str): Marker symbol (e.g., "s", "^", "o").

    Returns:
        None (displays plot).
    """
    if systems_to_plot is None:
        systems_to_plot = join_df["index_right"].unique()

    for system_id in systems_to_plot:
        water_segment = united_water_gdf.loc[[system_id]] if system_id in united_water_gdf.index else None
        subset_idx = join_df[join_df["index_right"] == system_id].index
        point_subset = sample_gdf.iloc[subset_idx]

        # Skip if either geometry is missing or empty
        if water_segment is None or water_segment.empty or point_subset.empty:
            print(f"Skipping system {system_id}: no data.")
            continue

        fig, ax = plt.subplots(facecolor=background_color)
        fig.set_size_inches(*figsize)

        district_gdf.plot(ax=ax, color='white', edgecolor='black')
        polygon_gdf.plot(ax=ax, color='None', edgecolor='grey', linewidth=2)

        water_segment.plot(ax=ax, color='blue', markersize=10, alpha=1, label=f"water system {system_id}")
        point_subset.plot(ax=ax, color=marker_style, markersize=40, marker=marker_symbol, label="locations")

        ax.set_xlabel("Latitude")
        ax.set_ylabel("Longitude")
        ax.legend()
        ax.set_title(f"Water System {system_id}", fontsize=14)
        plt.show()




def plot_selected_locations(district_gdf, locations_gdf, background_color='#FCF6F5FF', figsize=(5, 5)):
    fig, ax = plt.subplots(facecolor=background_color)
    fig.set_size_inches(*figsize)

    district_gdf.plot(ax=ax, color='white', edgecolor='black')
    locations_gdf.plot(ax=ax, color="red", marker="o", label="Selected Locations")

    plt.legend(fontsize=10)
    ax.set_xlabel("Latitude")
    ax.set_ylabel("Longitude")
    plt.show()
def extract_samples_by_water_system(
    df_full,
    join_df,
    systems_to_extract=None  # list of `index_right` values; if None, extract for all
):
    """
    Returns a dictionary of DataFrames, each containing samples associated with a specific water system.

    Parameters:
        df_full (DataFrame): The full original DataFrame (e.g., df_southampton).
        join_df (DataFrame): Output of spatial join (e.g., gpd.sjoin).
        systems_to_extract (list or None): List of `index_right` values to extract. If None, extracts all.

    Returns:
        dict: Keys are `index_right` values; values are corresponding DataFrame subsets.
    """
    if systems_to_extract is None:
        systems_to_extract = join_df["index_right"].unique()

    return {
        system_id: df_full.iloc[join_df[join_df["index_right"] == system_id].index]
        for system_id in systems_to_extract
        if not join_df[join_df["index_right"] == system_id].empty
    }

def quantiles_check(df, nutrients):
    q_low = df[nutrients[0::]].quantile(0.01)
    q_hi  = df[nutrients[0::]].quantile(0.99)
    
    df_return = df.copy()
    df_return[nutrients[0::]] = np.where( ((df[nutrients[0::]] < q_hi) & (df[nutrients[0::]] > q_low)), df[nutrients[0::]], np.nan)
    return df_return


def sum_year(df, notation, nutrients_to_sum):
    if np.size(notation)==1:
        df = df.sort_values(by="Date", ascending=True).reset_index()
        date = df[(df["sample.samplingPoint.notation"]==notation)]["Date"]
        sum_year = df[(df["sample.samplingPoint.notation"]==notation)][nutrients_to_sum].sum(min_count=1, axis=1)
        
    else:
        df = df.sort_values(by="Date", ascending=True).reset_index()
        date = df[(df["sample.samplingPoint.notation"].isin(notation))]["Date"]
        sum_year = df[(df["sample.samplingPoint.notation"].isin(notation))][nutrients_to_sum].sum(min_count=1, axis=1)
        
        

    return(pd.concat([date, sum_year], axis=1, keys=['Date', 'sum']))

def sum_seasons(df, notation, nutrients_to_sum):
    df = df.sort_values(by="Date", ascending=True).reset_index()

    if np.size(notation)==1:
        date_summer = df[(df["sample.samplingPoint.notation"]==notation)&(df["season"]==0)]["Date"]
        dain_summer = df[(df["sample.samplingPoint.notation"]==notation)&(df["season"]==0)][nutrients_to_sum].sum(min_count=1, axis=1)

        date_winter = df[(df["sample.samplingPoint.notation"]==notation)&(df["season"]==1)]["Date"]
        dain_winter = df[(df["sample.samplingPoint.notation"]==notation)&(df["season"]==1)][nutrients_to_sum].sum(min_count=1, axis=1)
    else:

        date_summer = df[(df["sample.samplingPoint.notation"].isin(notation))&(df["season"]==0)]["Date"]
        dain_summer = df[(df["sample.samplingPoint.notation"].isin(notation))&(df["season"]==0)][nutrients_to_sum].sum(min_count=1, axis=1)

        date_winter = df[(df["sample.samplingPoint.notation"].isin(notation))&(df["season"]==1)]["Date"]
        dain_winter = df[(df["sample.samplingPoint.notation"].isin(notation))&(df["season"]==1)][nutrients_to_sum].sum(min_count=1, axis=1)
    return(pd.concat([date_summer, dain_summer], axis=1, keys=['Date', 'sum']), pd.concat([date_winter, dain_winter], axis=1, keys=['Date', 'sum']))

def plot_sum_summer(df, notation, nutrients_to_sum, ax=None, plt_kwargs={}, sct_kwargs={}, xlabel=None, ylabel=None, plot_title=None, label_str=None):
	df = df.sort_values(by="Date", ascending=True).reset_index()

	df_summer, df_winter = sum_seasons(df, notation, nutrients_to_sum)
		
	if ax is None:
		ax = plt.gca()		
	ax.scatter(df_summer["Date"], df_summer["sum"], zorder=2, **sct_kwargs, label=label_str)

	if ylabel != None:
		ax.set_ylabel(str(ylabel))
	if xlabel != None:
		ax.set_xlabel(str(xlabel))
	if label_str != None:
		ax.legend(fontsize=18)
	if plot_title != None:
		ax.set_title(label=str(plot_title), fontsize=22)
			
	return(ax)

def plot_sum_winter(df, notation, nutrients_to_sum, ax=None,  plt_kwargs={}, sct_kwargs={}, xlabel=None, ylabel=None, plot_title=None, label_str=None):

	df_summer, df_winter = sum_seasons(df, notation, nutrients_to_sum)
		
	if ax is None:
		ax = plt.gca()		
	ax.scatter(df_winter["Date"], df_winter["sum"], zorder=2, **sct_kwargs, label=label_str)

	if ylabel != None:
		ax.set_ylabel(str(ylabel))
	if xlabel != None:
		ax.set_xlabel(str(xlabel))
	if label_str != None:
		ax.legend(fontsize=18)
	if plot_title != None:
		ax.set_title(label=str(plot_title), fontsize=22)
			
	return(ax)
	
	
	
def pred_linreg(x,y, date):
    m, c = np.polyfit(x[(np.isfinite(x))&(np.isfinite(y))], y[(np.isfinite(x))&(np.isfinite(y))], 1)
    y_model = m * date + c
    return y_model

def date_to_sec(df1, df2):


    return(np.array([(df1.reset_index().loc[i, "Date"]-df2.reset_index()["Date"][0]).total_seconds()*10**-8 for i in range(len(df1["Date"]))]))

def LinearRegression_boot_summer(df, notation, nutrients_to_sum , ax=None):
    df_summer, df_winter = sum_seasons(df, notation, nutrients_to_sum)
    
    x_df_sec = date_to_sec(df_summer, df)
    date = date_to_sec(df, df)
    
    x_df = np.array(df_summer["Date"])
    y_df = np.array(df_summer["sum"])
    
    lr_boot = []
    for i in range(0, 500):
        sample_index = np.random.choice(range(0, len(y_df)), len(y_df))
        X_samples = x_df_sec[sample_index]
        y_samples = y_df[sample_index]
        
        lr_boot.append(pred_linreg(X_samples, y_samples, date))
    

    lr_boot = np.array(lr_boot)
    lr = pred_linreg(x_df_sec, y_df, date)
    q_16, q_84 = np.percentile(lr_boot, (16, 84), axis=0)
    
    if ax is None:
        ax = plt.gca()
    ax.fill_between(df["Date"], q_16, q_84, alpha=0.2, color="grey")

    ax.plot(df["Date"], lr, color='darkblue', zorder=5)
    
    return ax

def LinearRegression_boot_winter(df, notation, nutrients_to_sum , ax=None):
    df_summer, df_winter = sum_seasons(df, notation, nutrients_to_sum)
    
    x_df_sec = date_to_sec(df_winter, df)
    date = date_to_sec(df, df)
    
    x_df = np.array(df_winter["Date"])
    y_df = np.array(df_winter["sum"])
    
    lr_boot = []
    for i in range(0, 500):
        sample_index = np.random.choice(range(0, len(y_df)), len(y_df))
        X_samples = x_df_sec[sample_index]
        y_samples = y_df[sample_index]
        
        lr_boot.append(pred_linreg(X_samples, y_samples, date))
    

    lr_boot = np.array(lr_boot)
    lr = pred_linreg(x_df_sec, y_df, date)
    q_16, q_84 = np.percentile(lr_boot, (16, 84), axis=0)
    
    if ax is None:
        ax = plt.gca()
    ax.fill_between(df["Date"], q_16, q_84, alpha=0.2, color="grey")

    ax.plot(df["Date"], lr, color='darkblue', zorder=5)
    
    return ax
    
def GP_missing_date_Xy(df, notation, nutrients_to_sum):

    df_sum = sum_year(df, notation, nutrients_to_sum)
    df_sum = df_sum.dropna().reset_index()
    df_sum_sec = np.array([(df_sum.loc[i, "Date"]-df_sum["Date"][0]).total_seconds()*10**-8 for i in range(len(df_sum["Date"]))])

    fitting = np.polyfit(df_sum_sec, df_sum["sum"], 
                     deg=2)

    p = np.poly1d(fitting)

    y_mean = p(df_sum_sec)

    X = df_sum_sec.reshape(-1, 1)
    y = df_sum["sum"]

    return(X, y, y_mean, p)

def find_missing_years(df, notation, nutrients_to_sum):
    df_year = np.unique(df.sort_values(by="Date", ascending=True).reset_index()["Date"])

    missing_years = [y for y in df_year if y not in np.unique(sum_year(df, notation, nutrients_to_sum).dropna()["Date"])]
    
    return(missing_years)

def find_all_years(df, notation, nutrients_to_sum):
    df_year = np.unique(df.sort_values(by="Date", ascending=True).reset_index()["Date"])
    
    return(df_year)
    
def GP_imputing(df, notation, nutrients_to_sum, kernel):
    
    X, y, y_mean, p = GP_missing_date_Xy(df, notation, nutrients_to_sum)
    
    gaussian_process = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    gaussian_process.fit(X, y-y_mean)

    missing_years = find_missing_years(df, notation, nutrients_to_sum)
    
    dates_random = pd.to_datetime(missing_years) 
    df_sum = sum_year(df, notation, nutrients_to_sum).reset_index()

    X_test = np.array((dates_random-pd.to_datetime(df_sum["Date"][0])).total_seconds()*10**-8)

    mean_y_pred, std_y_pred = gaussian_process.predict(X_test.reshape(-1,1), return_std=True)
    
    
    new_values = np.random.normal(mean_y_pred, std_y_pred, len(missing_years))+p(X_test)
    new_values[new_values<0] = 0.

    return(missing_years, new_values)
    
def GP_imputing(df, notation, nutrients_to_sum, kernel):
    
    X, y, y_mean, p = GP_missing_date_Xy(df, notation, nutrients_to_sum)
    
    gaussian_process = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    gaussian_process.fit(X, y-y_mean)

    missing_years = find_missing_years(df, notation, nutrients_to_sum)
    
    dates_random = pd.to_datetime(missing_years) 
    df_sum = sum_year(df, notation, nutrients_to_sum).reset_index()

    X_test = np.array((dates_random-pd.to_datetime(df_sum["Date"][0])).total_seconds()*10**-8)

    mean_y_pred, std_y_pred = gaussian_process.predict(X_test.reshape(-1,1), return_std=True)
    
    
    new_values = np.random.normal(mean_y_pred, std_y_pred, len(missing_years))+p(X_test)
    new_values[new_values<0] = 0.

    return(missing_years, new_values)
    
def GP_kernel (noise_level_=5, length_scale_ = 0.25, alpha_ = 1):

    noise_kernel = WhiteKernel(noise_level=noise_level_, noise_level_bounds=(1e-10, 1e+3))


    irregularities_kernel = RationalQuadratic(length_scale=length_scale_, alpha=alpha_, length_scale_bounds=(1e-5, 1e5))

    kernel =   noise_kernel + irregularities_kernel

    return(kernel)
    

