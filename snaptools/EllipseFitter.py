import numpy as np

# License: BSD
#Original Author: David Mashburn

# Original Documentation:

'''/*
Best-fitting ellipse routines by:

  Bob Rodieck
  Department of Ophthalmology, RJ-10
  University of Washington,
  Seattle, WA, 98195

Notes on best-fitting ellipse:

  Consider some arbitrarily shaped closed profile, which we wish to
  characterize in a quantitative manner by a series of terms, each
  term providing a better approximation to the shape of the profile.
  Assume also that we wish to include the orientation of the profile
  (i.e. which way is up) in our characterization.

  One approach is to view the profile as formed by a series harmonic
  components, much in the same way that one can decompose a waveform
  over a fixed interval into a series of Fourier harmonics over that
  interval. From this perspective the first term is the mean radius,
  or some related value (i.e. the area).  The second term is the
  magnitude and phase of the first harmonic, which is equivalent to the
  best-fitting ellipse.

  What constitutes the best-fitting ellipse?  First, it should have the
  same area.  In statistics, the measure that attempts to characterize some
  two-dimensional distribution of data points is the 'ellipse of
  concentration' (see Cramer, Mathematical Methods of Statistics,
  Princeton Univ. Press, 945, page 283).  This measure equates the second
  order central moments of the ellipse to those of the distribution,
  and thereby effectively defines both the shape and size of the ellipse.

  This technique can be applied to a profile by assuming that it constitutes
  a uniform distribution of points bounded by the perimeter of the profile.
  For most 'blob-like' shapes the area of the ellipse is close to that
  of the profile, differing by no more than about 4%. We can then make
  a small adjustment to the size of the ellipse, so as to give it the
  same area as that of the profile.  This is what is done here, and
  therefore this is what we mean by 'best-fitting'.

  For a real pathologic case, consider a dumbell shape formed by two small
  circles separated by a thin line. Changing the distance between the
  circles alters the second order moments, and thus the size of the ellipse
  of concentration, without altering the area of the profile.

public class Ellipse_Fitter implements PlugInFilter {
	public int setup(String arg, ImagePlus imp) {
		return DOES_ALL;
	}
	public void run(ImageProcessor ip) {
		EllipseFitter ef = new EllipseFitter();
		ef.fit(ip);
		IJ.write(IJ.d2s(ef.major)+" "+IJ.d2s(ef.minor)+" "+IJ.d2s(ef.angle)+" "+IJ.d2s(ef.xCenter)+" "+IJ.d2s(ef.yCenter));
		ef.drawEllipse(ip);
	}
}
*/'''

# I decided just to gank the algorithm straight from ImageJ...
# I changed it to assume that you know the bounding rectangle,
# and have sliced the image array accordingly (this is arr)
def EllipseFitter(arr, usePrint=False):
    left = 0 # holdover from the ImageJ version
    top = 0 # holdover from the ImageJ version
    width = arr.shape[0]
    height = arr.shape[1]

    HALFPI = np.pi/2.

    nCoordinates = 0 #int # Initialized by makeRoi()
    bitCount = 0 #int

    sqrtPi = np.sqrt(np.pi)

    xsum = 0.0
    ysum = 0.0
    x2sum = 0.0
    y2sum = 0.0
    xysum = 0.0

    for y in range(height):
        bitcountOfLine = 0
        xSumOfLine = 0
        for x in range(width):
            if arr[x,y] != 0:
                bitcountOfLine+=1
                xSumOfLine += x
                x2sum += x * x

        xsum += xSumOfLine
        ysum += bitcountOfLine * y
        ye = y
        xe = xSumOfLine
        xysum += xe*ye
        y2sum += ye*ye*bitcountOfLine
        bitCount += bitcountOfLine

    # getMoments
    if bitCount != 0:
        x2sum += 0.08333333 * bitCount
        y2sum += 0.08333333 * bitCount
        n = bitCount
        x1 = xsum/n
        y1 = ysum / n
        x2 = x2sum / n
        y2 = y2sum / n
        xy = xysum / n
        xm = x1
        ym = y1
        u20 = x2 - (x1 * x1)
        u02 = y2 - (y1 * y1)
        u11 = xy - x1 * y1

    # rest...
    m4 = 4.0 * np.abs(u02 * u20 - u11 * u11)
    if m4 < 0.000001:
        m4 = 0.000001
    a11 = u02 / m4
    a12 = u11 / m4
    a22 = u20 / m4
    xoffset = xm
    yoffset = ym

    tmp = a11 - a22
    if tmp == 0.0:
        tmp = 0.000001
    theta = 0.5 * np.arctan(2.0 * a12 / tmp)
    if theta < 0.0:
        theta += HALFPI
    if a12 > 0.0:
        theta += HALFPI
    elif a12 == 0.0:
        if a22 > a11:
            theta = 0.0
            tmp = a22
            a22 = a11
            a11 = tmp
        elif a11 != a22:
            theta = HALFPI
    tmp = np.sin(theta)
    if tmp == 0.0:
        tmp = 0.000001
    z = a12 * np.cos(theta) / tmp
    major = np.sqrt (1.0 / np.abs(a22 + z))
    minor = np.sqrt (1.0 / np.abs(a11 - z))
    scale = np.sqrt (bitCount / (np.pi * major * minor)) #equalize areas
    major = major*scale*2.0
    minor = minor*scale*2.0
    angle = 180.0 * theta / np.pi
    if angle == 180.0:
        angle = 0.0
    if major < minor:
        tmp = major
        major = minor
        minor = tmp
    xCenter = left + xoffset + 0.5
    yCenter = top + yoffset + 0.5

    if usePrint:
        print(angle)
        print(major, minor)
        print(xCenter, yCenter)

    return angle, major, minor, xCenter, yCenter
