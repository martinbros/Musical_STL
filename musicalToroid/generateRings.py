import numpy as np
from matplotlib import pyplot as plt
from stl import mesh
import pretty_errors


def genXYCords(numPoints, magnitude):
	return np.arange(numPoints), np.full(numPoints, magnitude)


def normalizePoints(x, origRange, newRange):
	x = np.divide(np.subtract(x, origRange[0]), np.subtract(origRange[1], origRange[0]))  # Normalize from 0 to 1
	multi = newRange[1] - newRange[0]
	x = np.add(np.multiply(x, multi), newRange[0])

	return x


def pol2cart(rho, phi):
	x = rho * np.cos(phi)
	y = rho * np.sin(phi)
	return x, y


def generateRing(numSegments, numPeaks, ringRad):
	xVert = []
	yVert = []
	zVert = []
	faces = []

	for idx in range(numSegments):
		# Calculate the vertices
		x, y = genXYCords(numPeaks, 1)  # Get peaks and x coordinates
		phi = normalizePoints(x, [0, numPeaks - 1], [-np.pi / 2.0, np.pi / 2.0])  # Convert x coordinate to be within range -pi/2 to pi/2
		x, zCord = pol2cart(y, phi)  # make a semi-circle, peaks being the magnitude, phi being the rotation step
		x = np.add(x, ringRad)  # Add the ring radius
		rotation = [2.0 * np.pi * (idx / numSegments)]  # calculate the rotation radian about z-axis
		xCord, yCord = pol2cart(x, rotation)  # Calculate x and y coordinates of 3d scatter

		xVert.extend(xCord)
		yVert.extend(yCord)
		zVert.extend(zCord)

		# Calculate the surface verticies
		if idx + 1 < numSegments:
			nextIdx = idx + 1
		else:
			nextIdx = 0

		colA = np.arange(idx * numPeaks, idx * numPeaks + numPeaks - 1)
		colB = np.add(colA, 1)
		colC = np.arange(nextIdx * numPeaks, nextIdx * numPeaks + numPeaks - 1)
		colD = np.add(colC, 1)
		vertical = [[colA[0], colB[-1], colC[0]], [colB[-1], colC[0], colD[-1]]]

		colA = np.reshape(colA, (-1, 1))
		colB = np.reshape(colB, (-1, 1))
		colC = np.reshape(colC, (-1, 1))
		colD = np.reshape(colD, (-1, 1))

		faces.extend(np.concatenate([colA, colB, colC], axis=1))
		faces.extend(np.concatenate([colC, colD, colB], axis=1))
		faces.extend(vertical)

	xVert = np.reshape(xVert, (-1, 1))
	yVert = np.reshape(yVert, (-1, 1))
	zVert = np.reshape(zVert, (-1, 1))

	return np.concatenate([xVert, yVert, zVert], axis=1), np.array(faces)


def genVertices(x, y, xRange, rotation, ringRad):
	phi = normalizePoints(x, xRange, [-np.pi / 2.0, np.pi / 2.0])  # Convert x coordinate to be within range -pi/2 to pi/2
	x, zCord = pol2cart(y, phi)  # make a semi-circle, peaks being the magnitude, phi being the rotation step
	x = np.add(x, ringRad)  # Add the ring radius
	xCord, yCord = pol2cart(x, rotation)  # Calculate x and y coordinates of 3d scatter

	xCord = np.reshape(xCord, (-1, 1))
	yCord = np.reshape(yCord, (-1, 1))
	zCord = np.reshape(zCord, (-1, 1))

	return np.concatenate([xCord, yCord, zCord], axis=1)


def genFaces(idx, numSegments, numPeaks):

	faces = []

	if idx + 1 < numSegments:
		nextIdx = idx + 1
	else:
		nextIdx = 0

	colA = np.arange(idx * numPeaks, idx * numPeaks + numPeaks - 1)
	colB = np.add(colA, 1)
	colC = np.arange(nextIdx * numPeaks, nextIdx * numPeaks + numPeaks - 1)
	colD = np.add(colC, 1)
	vertical = [np.array([colA[0], colB[-1], colC[0]]), np.array([colB[-1], colC[0], colD[-1]])]

	colA = np.reshape(colA, (-1, 1))
	colB = np.reshape(colB, (-1, 1))
	colC = np.reshape(colC, (-1, 1))
	colD = np.reshape(colD, (-1, 1))

	faces.extend(np.concatenate([colA, colB, colC], axis=1))
	faces.extend(np.concatenate([colC, colD, colB], axis=1))
	faces.extend(vertical)

	return faces


def genSTL(vertices, faces, fileName):
	# Create the mesh
	ring = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
	for i, f in enumerate(faces):
		for j in range(3):
			ring.vectors[i][j] = vertices[f[j], :]

	# Write the mesh to file "ring.stl"
	ring.save('%s.stl' % fileName)

"""
print("gen ring 1")
vertOne, facOne = generateRing(3, 6, 1)
genSTL(vertOne, facOne, "ring1")


print("gen ring 2")
numPeaks = 6
numSegments = 3
vertTwo = []
facTwo = []
x, y = genXYCords(numPeaks, 1)  # Get peaks and x coordinates
for idx in range(numSegments):
	rotation = [2.0 * np.pi * (idx / numSegments)]  # calculate the rotation radian about z-axis
	vertTwo.extend(genVertices(x, y, [0, numPeaks - 1], rotation, ringRad=1))
	facTwo.extend(genFaces(idx, numSegments, numPeaks))

genSTL(np.array(vertTwo), np.array(facTwo), "ring2")
"""