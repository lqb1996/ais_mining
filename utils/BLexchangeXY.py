import math

EARTH_RADIUS = 6.371229*1e6
PI = math.pi


def latitudeLongitudeDistEarth(sLat, sLng, eLat=0, eLng=0):
    x = 0
    y = 0
    out = 0
    x = (eLat-sLat) * PI * EARTH_RADIUS * math.cos(((sLng+eLng)/2) * PI / 180)/180
    y = (eLng-sLng) * PI * EARTH_RADIUS / 180
    out = math.hypot(x, y)
    return out


'''
经纬度转墨卡托
@param lng 经度
@param lat 纬度
@return wgs墨卡托[经度，纬度]
'''
def transformLonLatToMecator(lng, lat):
    earthRad = 6378137.0
    x = lng * math.pi / 180 * earthRad
    a = lat * math.pi / 180
    y = earthRad / 2 * math.log((1.0 + math.sin(a)) / (1.0 - math.sin(a)))
    return x, y


'''
墨卡托转经纬度
@param x
@param y
return 经纬度[经度，纬度]
'''
def transformMercatorToLngLat(x, y):
    lng = x / 20037508.34 * 180
    lat = y / 20037508.34 * 180
    lat = 180 / math.pi * (2 * math.atan(math.exp(lat * math.pi / 180)) - math.pi / 2)
    return lng, lat

