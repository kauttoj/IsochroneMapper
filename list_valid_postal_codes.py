import pandas
import pickle
import numpy

POLLINGFILE = "IsochroneMapper_results.pickle" # result file
RESULTFILE = "Suitable_areacodes.txt"
POSTALCODEFILE = "FI_postal_codes.txt" # postal codes from http://download.geonames.org/export/zip/

THRESHOLD = 60 # in minutes

results, points, TRANSPORT_PARAMETERS = pickle.load(open(POLLINGFILE, "rb"))
postalcodes = pandas.read_csv(POSTALCODEFILE,sep="\t",header=None,dtype=str)
# columns:
# 10 latitude
# 11 longitude
# 1 code
all_coords = numpy.array(postalcodes[[9,10]],dtype=numpy.float)
code = postalcodes[1]
arealabel = postalcodes[2]

ind = [i for i in range(0,len(results)) if len(results[i])>0]
results = [results[i] for i in ind]
points = [points[i] for i in ind]

coords = []
for k,res in enumerate(results):
    for j in range(len(res)):
        if res[j]['duration']/60<THRESHOLD:
            coords.append(numpy.array(list(points[k]) + [res[j]['duration']/60,]))
            break

areas = dict()
print("list of valid codes:")
for coord in coords:
    dist = numpy.sum(numpy.power(all_coords - coord[0:2],2),axis=1)
    k = numpy.argmin(dist)
    if code[k] not in areas:
        areas[code[k]] = [coord[2]]

areas = pandas.DataFrame.from_dict(areas,orient="index",columns=["duration"])
areas = areas.sort_values("duration")
areas = areas[areas["duration"]<THRESHOLD]
areas["duration"] = areas["duration"].round()
areas.to_csv(RESULTFILE)

print("\nAll done!")
