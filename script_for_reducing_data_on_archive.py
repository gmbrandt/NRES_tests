import os
import datetime
import time

TIMESTAMP_FORMAT = '%Y-%m-%dT%H:%M:%S'
#make a datetime loop using time deltas and such. Just set the min and max date each time, and the directory accordingly.
site = 'elp'
interval = 5 #days
number = 30
start = datetime.datetime(year=2019, month=3, day=20)
dates_to_reduce = [start - i * datetime.timedelta(days=interval) for i in range(number)][::-1]
instrument = {'lsc': 'nres01', 'elp': 'nres02', 'cpt': 'nres03', 'tlv': 'nres04'}[site]
camera = {'nres01': 'fa09', 'nres02': 'fa17', 'nres03': 'fa13', 'nres04': 'fa18'}[instrument]

for dateobject in dates_to_reduce:
    min_date = dateobject.strftime(TIMESTAMP_FORMAT)
    max_date = (dateobject + datetime.timedelta(hours=48)).strftime(TIMESTAMP_FORMAT)
    file_date = dateobject.strftime('%Y%m%d')
    raw_path = '/archive/engineering/{site}/{instrument}/{date}/raw'.format(site=site, instrument=instrument, date=file_date)
    exec_string = 'docker run --rm --name "banzai_nres_back_reduction_{site}" -l gtn.lco.logstash="yes" -v /archive:/archive --entrypoint="banzai_nres_reduce_night" docker.lco.global/banzai-nres:0.3.0-411-g6eabd0f --site {site} --instrument-name {inst} --camera {cam} --fpack --min-date {min_date} --max-date {max_date} --ignore-schedulability --raw-path {path} --db-address=postgresql://pipeline:pipeline@chanunpa.lco.gtn:5435/pipeline &>> "/tmp/banzai_nres_back_reduction_{site}.log" '.format(site=site, inst=instrument, cam=camera, path=raw_path, min_date=min_date, max_date=max_date)
    os.system(exec_string)
    #print(exec_string)
    time.sleep(1)
