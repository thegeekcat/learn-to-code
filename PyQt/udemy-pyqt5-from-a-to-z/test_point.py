import Point as pt

pt1 = pt.Point(3, 12, 4326)
pt3 = pt.Point3D(6, 16, 2140, 4326, "Feet")
pt2 = pt.Point3D(5, 56, 2200, 4326)

print(pt1.srs)

print(pt1.distance_to(pt2))
print(pt2.distance_to(pt3))
print(pt2.distance3D_to(pt3))

print(pt1)
print(pt2)
print(pt3)


print(type(pt1))
print(type(pt2))
print(type(pt3))

pt3.__units
pt3.set_units("Cubics")
print(pt3.get_units())
