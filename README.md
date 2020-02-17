# IsochroneMapper
Code to compute isochrone map for a target location and surrounding region. Made for HSL Helsinki region.

In short, this code creates a HTML page that depicts minimum travel time between surrounding region and target location (aka isochrone map). This is achieved by brute-force polling of a given area (circle or rectangle) using predefined travel parameters. Results are then plotted as contours and points.
The code was made for Helsinki region using HSL API, but you could apply it other regions as well with some modifications to parameters and API service.
Below are couple of screenshots what you get (see result file "mapper_results.html"):

![Sample figure 1](https://raw.githubusercontent.com/kauttoj/IsochroneMapper/master/sample1.png)
![Sample figure 2](https://raw.githubusercontent.com/kauttoj/IsochroneMapper/master/sample2.png)

16.2.2020 Janne Kauttonen
