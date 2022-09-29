from os import path
from time import time
import pandas as pd
import numpy as np
import cv2
from skimage.draw import polygon, line

# ? defult matrix data type
DTYPE = np.uint8

# ? IRAN REGION
IRAN_TOP_RIGHT = (68, 44)  # long lat
IRAN_BOTTOM_LEFT = (36, 22)  # long lat

BRAZIL_TOP_RIGHT = (-26, 2)  # long lat
BRAZIL_BOTTOM_LEFT = (-79, -30)  # long lat

USA_TOP_RIGHT = (-66, 50)
USA_BOTTOM_LEFT = (-124, 26)

PAKISTAN_TOP_RIGHT = (78, 38)
PAKISTAN_BOTTOM_LEFT = (61, 24)


# ? set selected region
REGION_MAX = IRAN_TOP_RIGHT
REGION_MIN = IRAN_BOTTOM_LEFT

# ? set default scale and map size
SCALE = 1000
MAX_X = (REGION_MAX[0] - REGION_MIN[0]) * SCALE  # longitude
MAX_Y = (REGION_MAX[1] - REGION_MIN[1]) * SCALE  # latitude

# ? set default satellite parameters
SATELLITE_FOV = 0.87
SATELLITE_DISTANCE = 500000
ANGLE = 55
ANGLE_STEP = 1
ANGLE_RANGE = 10  # ( -10 , +10 )

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
    for i, row in df.iterrows():
        point = [row["Longitude"], row["Latitude"]]
        # print("in filter => ", point, region_min, region_max)
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


def check_points_are_same(point1, point2, scale):
    value1 = int(point1[1] * (scale / 2000))
    value2 = int(point2[1] * (scale / 2000))
    return value1 == value2


# def tilt_check(earth, points, scale, region_min, max_x, max_y):
#     # region_is_filled = get_ragion_sumation(
#     #     earth, points, scale, region_min, max_x, max_y
#     # )
#     # print("region filled => ", region_is_filled)
#     # return region_is_filled > 10000
#     return True


def tilt_point(point, angle, angle_step, distance):
    point = get_shifted_point(point, distance, angle, angle_step, angle_step > 0)
    return point


def change_angle_step(
    earth,
    point,
    angle,
    angle_step,
    angle_range,
    old_step,
    scale,
    region_min,
    max_x,
    max_y,
):
    selected_point = get_rotation_point(
        earth, point, angle, angle_step, angle_range, scale, region_min, max_x, max_y
    )
    if selected_point is not None:
        return int(selected_point["step"])
    else:
        return old_step


def fill_point(earth, points, scale, region_min, max_x, max_y, step=10):
    earth = draw_rectangle(earth, points, scale, region_min, max_x, max_y, step)
    return earth


def fill_earth(
    df,
    earth,
    check_point,
    angle_step,
    angle_range,
    distance,
    scale,
    region_min,
    angle,
    max_x,
    max_y,
    step,
):
    limit = distance / 2
    selected_angle_step = 0
    distance_to_shift = get_satellite_view_distance(angle_step)
    is_debounce = False
    debounce = 0
    max_debounce = 100
    print("in fill earth => ", df.shape[0])
    for i in range(1, df.shape[0]):
        angle = float(df.iloc[i]["Angle"])
        org_point = [df.iloc[i]["Longitude"], df.iloc[i]["Latitude"]]
        prev_point = [df.iloc[i - 1]["Longitude"], df.iloc[i - 1]["Latitude"]]
        # angle = 55 if org_point[1] > prev_point[1] else 135
        if angle == 50:
            angle = angle if org_point[1] > prev_point[1] else 90 + angle
        if (
            True
            and not is_debounce
            and check_points_are_same(org_point, check_point, scale)
        ):
            selected_angle_step = change_angle_step(
                earth,
                org_point,
                angle,
                angle_step,
                angle_range,
                0,
                scale,
                region_min,
                max_x,
                max_y,
            )
            is_debounce = True
            # print("--------------------------")
            # print("point =>",org_point)
            # print("selected step => ",selected_angle_step)
        if is_debounce:
            debounce += 1
            if debounce >= max_debounce:
                debounce = 0
                is_debounce = False
        point = tilt_point(
            org_point,
            angle,
            selected_angle_step,
            distance_to_shift,
        )
        # print("tilted point => ", point, org_point)
        points = get_points(limit, point, angle)
        earth = fill_point(
            earth,
            points,
            scale,
            region_min,
            max_x,
            max_y,
            step,
        )
    return earth


def get_ragion_sumation(earth, points, scale, region_min, max_x, max_y):
    points = transform_points(points, scale, region_min, max_x, max_y)
    points = np.array(points)
    rr, cc = polygon(points[:, 0], points[:, 1], earth.shape)
    return earth[rr, cc].sum()


def get_shifted_points(point, distance, angle, step):
    if step == 0:
        return point
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


def get_shifted_point(point, distance, angle, step, right=True):
    step = step if step > 0 else -step
    if step == 0:
        return point
    if angle <= 90:
        angle = np.pi * (angle / 180)
        theta = (np.pi / 2) - angle
        delta_x = (np.cos(theta) * distance) * step
        delta_y = (np.sin(theta) * distance) * step
        lon = meter_to_lon(delta_x)
        lat = meter_to_lat(delta_y)
        return (
            [point[0] + lon, point[1] - lat]
            if right
            else [point[0] - lon, point[1] + lat]
        )
    else:
        angle = np.pi * (angle / 180)
        theta = angle - (np.pi / 2)
        delta_x = (np.cos(theta) * distance) * step
        delta_y = (np.sin(theta) * distance) * step
        lon = meter_to_lon(delta_x)
        lat = meter_to_lat(delta_y)
        return (
            [point[0] + lon, point[1] + lat]
            if right
            else [point[0] - lon, point[1] - lat]
        )


def get_rotation_point(
    earth, point, angle, angle_step, angle_range, scale, region_min, max_x, max_y
):
    distance = get_satellite_view_distance(angle_step)
    all_points = []
    all_points.append({"point": point, "step": 0})
    for i in range(1, angle_range + 1):
        point1, point2 = get_shifted_points(point, distance, angle, i)
        all_points.append({"point": point1, "step": i})
        all_points.append({"point": point2, "step": -i})
    limit = get_satellite_view_distance() / 2
    min_sum = 10e15
    min_point = None
    # print("#############################")
    for pnt in all_points:
        points = get_points(limit, pnt["point"], angle)
        # if int(angle) == 125:
        #     earth = draw_rectangle(earth,points,scale,region_min,max_x,max_y,100)
        sumation = get_ragion_sumation(earth, points, scale, region_min, max_x, max_y)
        if sumation < min_sum:
            min_sum = sumation
            min_point = pnt
    return min_point


def analyze_map(map_mat):
    print("----- map analyzer -----")
    print("mean => ", map_mat.mean())
    print("max => ", map_mat.max())
    print("min => ", map_mat.min())
    print("sum => ", map_mat.sum())
    print(
        "area covered => ",
        np.count_nonzero(map_mat) / (map_mat.shape[0] * map_mat.shape[1]),
    )
    # print("variance => ", np.var(map_mat))
    # print("std => ", np.std(map_mat))


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
    cv2.waitKey()


def get_dataframe(df_path):
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
    check_point,
    angle_step,
    angle_range,
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
    region_df = filter_region(df, region_min, region_max, 1)
    distance = get_satellite_view_distance(satellite_fov, satellite_distance)
    new_map_mat = fill_earth(
        region_df,
        map_mat,
        check_point,
        angle_step,
        angle_range,
        distance,
        scale,
        region_min,
        angle,
        max_x,
        max_y,
        step,
    )
    return new_map_mat


def main():
    max_x, max_y = MAX_X, MAX_Y
    region_min, region_max = REGION_MIN, REGION_MAX
    scale = SCALE
    angle = 100  # 97.4 # ANGLE
    angle_step = ANGLE_STEP
    angle_range = ANGLE_RANGE
    satellite_fov = SATELLITE_FOV
    satellite_distance = SATELLITE_DISTANCE
    map_image_name = "Pakistan_tilt"
    dataframe_name = "Pakistan.csv"
    check_point = [68.135, 29.059]  # [29.059371,68.135715]  # [42.635, 36.4] #

    # ! ---------------------------------------------------------------------------
    # TODO => change this parameters
    load_dataframe = True  # * load new data from csv file
    force_new_map = True  # * dont use old saved image and create empty array
    # ! ---------------------------------------------------------------------------

    print("start")
    start_time = time()
    map_mat = get_map(max_x, max_y, map_image_name, force_new_map)
    if load_dataframe:
        print("in load dataframe")
        # df = get_dataframe(dataframe_name)
        df1 = get_dataframe(dataframe_name)
        df1.insert(3, "Angle", 97.4)
        df = pd.concat([df1])
        print("df => ", df.shape)
        print("update map")
        map_mat = update_map(
            df,
            map_mat,
            check_point,
            angle_step,
            angle_range,
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
    end_time = time()
    analyze_map(map_mat)
    print("time => ", (end_time - start_time))
    save_map(map_mat, map_image_name)
    print("end")


if __name__ == "__main__":
    main()
