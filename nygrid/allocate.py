"""
Code for allocating renewable generation resources to NY Grid buses.

The nearest neighbor algorithm is originally from:

https://autogis-site.readthedocs.io/en/2019/notebooks/L3/nearest-neighbor-faster.html
"""

from sklearn.neighbors import BallTree
import numpy as np
import geopandas as gpd
from typing import Union, Tuple, Dict


def get_nearest(src_points: np.array,
                candidates: np.array,
                k_neighbors: int = 1,
                metric: str = 'minkowski',
                leaf_size: int = 20) -> Tuple[np.array, np.array]:
    """
    Find nearest neighbors for all source points from a set of candidate points

    Parameters
    ----------
    src_points : np.ndarray
        A numpy array of shape (n, 2) representing the source points.
    candidates : np.ndarray
        A numpy array of shape (m, 2) representing the candidate points.
    k_neighbors : int
        Number of nearest neighbors to return.
    metric : str
        The distance metric to use. Default is 'minkowski'.
    leaf_size : int
        Leaf size passed to BallTree. Default is 20.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The indices of the k-nearest neighbors in the candidates array and the corresponding distances.
    """

    # Create tree from the candidate points
    tree = BallTree(candidates, leaf_size=leaf_size, metric=metric)

    # Find the closest points and distances
    distances, indices = tree.query(src_points, k=k_neighbors)

    # Transpose to get distances and indices into arrays
    distances = distances.transpose()
    indices = indices.transpose()

    # Get closest indices and distances (i.e. array at index 0)
    # note: for the second closest points, you would take index 1, etc.
    closest = indices[0]
    closest_dist = distances[0]

    # Return indices and distances
    return closest, closest_dist


def nearest_neighbor_lat_lon(left_gdf: gpd.GeoDataFrame,
                             right_gdf: gpd.GeoDataFrame,
                             return_dist: bool = False,
                             leaf_size: int = 20) -> Union[Dict, Tuple]:
    """
    For each point in left_gdf, find closest point in right GeoDataFrame and return them.

    NOTICE: Assumes that the input Points are in WGS84 projection (lat/lon).

    Parameters
    ----------
    left_gdf : gpd.GeoDataFrame
        A GeoDataFrame containing the origin points.
    right_gdf : gpd.GeoDataFrame
        A GeoDataFrame containing the candidate destination points.
    return_dist : bool
        If True, the distance between the nearest neighbors is returned.
    leaf_size : int
        Leaf size passed to BallTree. Default is 20.

    Returns
    -------
    closest_points: Union[Dict, Tuple]
        A dictionary or tuple containing the closest points and distances (if requested).
    """

    left_geom_col = left_gdf.geometry.name
    right_geom_col = right_gdf.geometry.name

    # Ensure that index in right gdf is formed of sequential numbers
    right = right_gdf.copy().reset_index(drop=True)

    # Parse coordinates from points and insert them into a numpy array as RADIANS
    left_radians = np.array(
        left_gdf[left_geom_col].apply(lambda geom: (geom.x * np.pi / 180, geom.y * np.pi / 180)).to_list())
    right_radians = np.array(
        right[right_geom_col].apply(lambda geom: (geom.x * np.pi / 180, geom.y * np.pi / 180)).to_list())

    # Find the nearest points
    # -----------------------
    # closest ==> index in right_gdf that corresponds to the closest point
    # dist ==> distance between the nearest neighbors (in meters)

    closest, dist = get_nearest(src_points=left_radians, candidates=right_radians,
                                metric='haversine', leaf_size=leaf_size)

    # Return points from right GeoDataFrame that are closest to points in left GeoDataFrame
    closest_points = right.loc[closest]

    # Ensure that the index corresponds the one in left_gdf
    closest_points = closest_points.reset_index(drop=True)

    # Add distance if requested
    if return_dist:
        # Convert to meters from radians
        earth_radius = 6371000  # meters
        closest_points['distance'] = dist * earth_radius

    return closest_points


def nearest_neighbor_meters(left_gdf: gpd.GeoDataFrame,
                            right_gdf: gpd.GeoDataFrame,
                            return_dist: bool = False,
                            leaf_size: int = 20) -> Union[Dict, Tuple]:
    """
    For each point in left_gdf, find the closest point in right GeoDataFrame and return them.

    NOTICE: Assumes that the input Points are in WGS84 projection (meters).

    Parameters
    ----------
    left_gdf : gpd.GeoDataFrame
        A GeoDataFrame containing the origin points.
    right_gdf : gpd.GeoDataFrame
        A GeoDataFrame containing the candidate destination points.
    return_dist : bool
        If True, the distance between the nearest neighbors is returned.
    leaf_size : int
        Leaf size passed to BallTree. Default is 20.

    Returns
    -------
    closest_points: Union[Dict, Tuple]
        A dictionary or tuple containing the closest points and distances (if requested).
    """

    left_geom_col = left_gdf.geometry.name
    right_geom_col = right_gdf.geometry.name

    # Ensure that index in right gdf is formed of sequential numbers
    right = right_gdf.copy().reset_index(drop=True)

    # Parse coordinates from points and insert them into a numpy array as METERS
    left_meters = np.array(left_gdf[left_geom_col].apply(
        lambda geom: (geom.x, geom.y)).to_list())
    right_meters = np.array(right[right_geom_col].apply(
        lambda geom: (geom.x, geom.y)).to_list())

    # Find the nearest points
    # -----------------------
    # closest ==> index in right_gdf that corresponds to the closest point
    # dist ==> distance between the nearest neighbors (in meters)

    closest, dist = get_nearest(
        src_points=left_meters, candidates=right_meters, leaf_size=leaf_size)

    # Return points from right GeoDataFrame that are closest to points in left GeoDataFrame
    closest_points = right.loc[closest]

    # Ensure that the index corresponds the one in left_gdf
    closest_points = closest_points.reset_index(drop=True)

    # Add distance if requested
    if return_dist:
        # Convert to meters from radians
        # earth_radius = 6371000  # meters
        # closest_points['distance'] = dist * earth_radius
        closest_points['distance'] = dist

    return closest_points
