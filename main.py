from os import path
from time import time
import pandas as pd
import numpy as np
import cv2
from skimage.draw import polygon, line

# ? defult matrix data type
DTYPE = np.uint8

# ? IRAN REGION
IRAN_TOP_LEFT = (36, 44)  # long lat
IRAN_TOP_RIGHT = (68, 44)  # long lat
IRAN_BOTTOM_LEFT = (36, 22)  # long lat
IRAN_BOTTOM_RIGHT = (68, 22)  # long lat

# ? set selected region
REGION_MIN = IRAN_BOTTOM_LEFT
REGION_MAX = IRAN_TOP_RIGHT

# ? set default scale and map size
SCALE = 1000
MAX_X = (REGION_MAX[0] - REGION_MIN[0]) * SCALE  # longitude
MAX_Y = (REGION_MAX[1] - REGION_MIN[1]) * SCALE  # latitude

# ? set default satellite parameters
SATELLITE_FOV = 0.87
SATELLITE_DISTANCE = 500000
ANGLE = 55
ANGLE_STEP = 2

# ? set default earth parameters
EARTH_RADIUS = 6371000  # meters
EARTH_CIRCUMFERENCE = 40075160
LAT_TO_METER = EARTH_CIRCUMFERENCE / 360
LON_TO_METER = EARTH_CIRCUMFERENCE / 360


def lat_to_meter(lat):
    return lat * LAT_TO_METER


def lon_to_meter(lon):
    return lon * LON_TO_METER


def meter_to_lat(meter):
    return meter / LAT_TO_METER


def meter_to_lon(meter):
    return meter / LON_TO_METER


def get_satellite_view_distance(
    satellite_fov=SATELLITE_FOV, satellite_distance=SATELLITE_DISTANCE
):
    satellite_fov_rad = satellite_fov * np.pi / 180  # satellite fov in radians
    satellite_view = 2 * np.tan(satellite_fov_rad) * satellite_distance
    return satellite_view


def get_point(point, distance, angle, action=1):
    distance = distance * np.sqrt(2)
    if angle <= 90:
        if action % 2 == 1:
            theta = (np.pi * (angle - 45)) / 180
            delta_y = np.sin(theta) * distance
            delta_x = np.cos(theta) * distance
            lat = meter_to_lat(delta_y)
            lon = meter_to_lon(delta_x)
            return (
                [point[0] + lon, point[1] + lat]
                if action == 1
                else [point[0] - lon, point[1] - lat]
            )
        elif action % 2 == 0:
            theta = (np.pi * (180 - angle - 45)) / 180
            delta_y = np.sin(theta) * distance
            delta_x = np.cos(theta) * distance
            lat = meter_to_lat(delta_y)
            lon = meter_to_lon(delta_x)
            return (
                [point[0] - lon, point[1] + lat]
                if action == 2
                else [point[0] + lon, point[1] - lat]
            )
    else:
        if action % 2 == 1:
            theta = (np.pi * (angle - 45)) / 180
            delta_y = np.sin(theta) * distance
            delta_x = np.cos(theta) * distance
            lat = meter_to_lat(delta_y)
            lon = meter_to_lon(delta_x)
            return (
                [point[0] + lon, point[1] + lat]
                if action == 1
                else [point[0] - lon, point[1] - lat]
            )
        elif action % 2 == 0:
            theta = (np.pi * (180 - angle - 45)) / 180
            delta_y = np.sin(theta) * distance
            delta_x = np.cos(theta) * distance
            lat = meter_to_lat(delta_y)
            lon = meter_to_lon(delta_x)
            return (
                [point[0] - lon, point[1] + lat]
                if action == 2
                else [point[0] + lon, point[1] - lat]
            )


def get_points(distance, point=(0, 0), angle=0):
    point1 = get_point(point, distance, angle, 1)
    point2 = get_point(point, distance, angle, 2)
    point3 = get_point(point, distance, angle, 3)
    point4 = get_point(point, distance, angle, 4)
    return [point1, point2, point3, point4]


def filter_region(df, region_min, region_max, limit=3):
    indexes = []
    for i,row in df.iterrows():
        point = [row["Longitude"], row["Latitude"]]
        if point[0] <= (region_min[0] + limit) or point[0] >= (region_max[0] - limit):
            indexes.append(i)
            continue
        if point[1] <= (region_min[1] + limit) or point[1] >= (region_max[1] - limit):
            indexes.append(i)
            continue
    df = df.drop(indexes)
    return df


def transform_point(point, scale, region_min, max_x, max_y):
    new_point = []
    new_point.insert(
        0, (int(point[0] * scale) + max_x / 2) - (((region_min[0]) * scale) + max_x / 2)
    )
    new_point.insert(
        1,
        ((int(point[1] * scale) + max_y / 2) - (((region_min[1]) * scale) + max_y / 2)),
    )
    return new_point


def transform_points(points, scale, region_min, max_x, max_y):
    new_points = []
    for point in points:
        new_points.append(transform_point(point, scale, region_min, max_x, max_y))
    return new_points


def draw_rectangle(earth, points, scale, region_min, max_x, max_y, step):
    points = transform_points(points, scale, region_min, max_x, max_y)
    points = np.array(points)
    rr, cc = polygon(points[:, 0], points[:, 1], earth.shape)
    earth[rr, cc] += step
    return earth


def draw_line(earth, point, next_point, scale, region_min, max_x, max_y, step):
    point = transform_point(point, scale, region_min, max_x, max_y)
    next_point = transform_point(next_point, scale, region_min, max_x, max_y)
    rr, cc = line(point[0], point[1], next_point[0], next_point[1])
    earth[rr, cc] += step
    return earth


def fill_earth(df, earth, distance, scale, region_min, angle, max_x, max_y, step=10):
    limit = distance / 2
    for i,row in df.iterrows():
        point = [row["Longitude"], row["Latitude"]]
        points = get_points(limit, point, angle)
        earth = draw_rectangle(earth, points, scale, region_min, max_x, max_y, step)
    return earth


def get_ragion_sumation(earth, points, scale, region_min, max_x, max_y):
    points = transform_points(points, scale, region_min, max_x, max_y)
    points = np.array(points)
    rr, cc = polygon(points[:, 0], points[:, 1], earth.shape)
    return earth[rr, cc].sum()


def get_shifted_points(point, distance, angle, step):
    if angle <= 90:
        angle = np.pi * (angle / 180)
        theta = (np.pi / 2) - angle
        delta_x = (np.cos(theta) * distance) * step
        delta_y = (np.sin(theta) * distance) * step
        lon = meter_to_lon(delta_x)
        lat = meter_to_lat(delta_y)
        return [[point[0] + lon, point[1] - lat], [point[0] - lon, point[1] + lat]]
    else:
        angle = np.pi * (angle / 180)
        theta = angle - (np.pi / 2)
        delta_x = (np.cos(theta) * distance) * step
        delta_y = (np.sin(theta) * distance) * step
        lon = meter_to_lon(delta_x)
        lat = meter_to_lat(delta_y)
        return [[point[0] + lon, point[1] + lat], [point[0] - lon, point[1] - lat]]


def get_rotation_point(
    earth, point, angle, angle_step, scale, region_min, max_x, max_y
):
    distance = get_satellite_view_distance(angle_step)
    point1, point2 = get_shifted_points(point, distance, angle, 1)
    point3, point4 = get_shifted_points(point, distance, angle, 2)
    all_points = np.array(
        [
            {
                "point": point3,
                "step": 2,
            },
            {
                "point": point1,
                "step": 1,
            },
            {
                "point": point,
                "step": 0,
            },
            {
                "point": point2,
                "step": -1,
            },
            {
                "point": point4,
                "step": -2,
            },
        ]
    )
    limit = get_satellite_view_distance() / 2
    min_sum = 10e12
    min_point = None
    for pnt in all_points:
        points = get_points(limit, pnt["point"], angle)
        sumation = get_ragion_sumation(earth, points, scale, region_min, max_x, max_y)
        if sumation < min_sum:
            min_sum = sumation
            min_point = pnt
    return min_sum, min_point


def analyze_map(map_mat):
    print("----- map analyzer -----")
    print("mean => ", map_mat.mean())
    print("max => ", map_mat.max())
    print("min => ", map_mat.min())
    print("sum => ", map_mat.sum())


def save_map(map_mat, name):
    try:
        img = np.rot90(map_mat, 1)
        img_path = f"./last-data/{name}.png"
        cv2.imwrite(img_path, img)
    except:
        print("error in save image")


def get_map(max_x, max_y, name, force_new_map):
    img_path = f"./last-data/{name}.png"
    is_exist = path.isfile(img_path)
    mat = None
    if force_new_map or not is_exist:
        mat = np.zeros((max_x, max_y), dtype=DTYPE)
    else:
        img = cv2.imread(img_path)
        mat = np.array(img, dtype=DTYPE)
        mat = np.rot90(mat, 3)
    return mat


def show_map(map_mat, name="map"):
    cv2.imshow(name, map_mat)
    key = cv2.waitKey()


def get_dataframe(df_path="satellite-data-6m.csv"):
    df_path = f"./data/{df_path}"
    df = pd.read_csv(df_path)
    df_columns_org = list(df.columns)
    df.drop(df_columns_org[3:], axis=1, inplace=True)
    new_columns_names = ["Time", "Latitude", "Longitude"]
    df.columns = new_columns_names
    return df


def update_map(
    df,
    map_mat,
    max_x,
    max_y,
    region_min,
    region_max,
    scale,
    angle,
    satellite_fov,
    satellite_distance,
    step=1,
):
    iran_df = filter_region(df.copy(), region_min, region_max, 1)
    distance = get_satellite_view_distance(satellite_fov, satellite_distance)
    new_map_mat = fill_earth(
        iran_df, map_mat.copy(), distance, scale, region_min, angle, max_x, max_y, step
    )
    return new_map_mat


def main():
    max_x, max_y = MAX_X, MAX_Y
    region_min, region_max = REGION_MIN, REGION_MAX
    scale = SCALE
    angle = ANGLE
    angle_step = ANGLE_STEP
    satellite_fov = SATELLITE_FOV
    satellite_distance = SATELLITE_DISTANCE
    map_image_name = "map_mat_image_1"
    dataframe_name = "data-2.csv"
    check_point = [42.635, 24.303]
    # ! ---------------------------------------------------------------------------
    # TODO => change this parameters
    load_dataframe = True  # * load new data from csv file
    force_new_map = True  # * dont use old saved image and create empty array
    # ! ---------------------------------------------------------------------------
    try:
        print("start")
        start_time = time()
        map_mat = get_map(max_x, max_y, map_image_name, force_new_map)
        if load_dataframe:
            print("in load dataframe")
            df = get_dataframe(dataframe_name)
            print("update map")
            map_mat = update_map(
                df,
                map_mat,
                max_x,
                max_y,
                region_min,
                region_max,
                scale,
                angle,
                satellite_fov,
                satellite_distance,
                10,
            )
        print("get best point")
        min_sum, point = get_rotation_point(
            map_mat.copy(),
            check_point,
            12,
            angle_step,
            scale,
            region_min,
            max_x,
            max_y,
        )
        end_time = time()
        print(point, min_sum)
        analyze_map(map_mat)
        print("time => ", (end_time - start_time))
        # show_map(map_mat)
        save_map(map_mat, map_image_name)
        print("end")
    except:
        print("error in main")


if __name__ == "__main__":
    main()
