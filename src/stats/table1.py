"""
Script for producing the statistics in Table 1 of the paper.
"""
import os
import csv
import sys
import json
import dateutil

import numpy as np
import pandas as pd
from tqdm import tqdm


CHUNK_SIZE = 100
PATH = "../../projects/nuclear/resources/backup/etch-a-cell-classifications.csv"
OUT_PATH = "../../projects/nuclear/resources/backup/stats.csv"
WORKFLOW = "Going Nuclear"
MAX_CLASSIFICATION_TIME = 2*3600


def read_relevant_data_from_dataframe_chunks(csv_chunks):
    """
    format...

    rois = {
        'ROI_1656-6756-329': {
            'images': {
                'image filename 1': [23, 34, 243, 19 ...]
            },
            'users': {}
        }
    }
    """
    rois = {}

    for chunk in tqdm(csv_chunks):
        for _, entry in chunk.iterrows():

            if entry['workflow_name'] == WORKFLOW:

                metadata = json.loads(entry['metadata'])
                subject_data = json.loads(entry['subject_data'])[str(entry['subject_ids'])]
                annotations = json.loads(entry['annotations'])[0]['value']

                started_at = dateutil.parser.parse(metadata['started_at']).timestamp()
                created_at = dateutil.parser.parse(metadata['finished_at']).timestamp()
                timedelta = created_at - started_at

                # note: batches of 5 have the same timestamp (though usually only the central one is actually annotated)
                # frames = {}
                # for annotation in annotations:
                #     frame = annotation['frame']
                #     if frame not in frames.keys():
                #         frames[frame] = 1
                
                # it was decided to just use the central frame for these statistics
                frames = {'2': 1}

                for frame, frame_val in frames.items():
                    if f'Image {frame}' in subject_data:
                        filename, ext = os.path.splitext(subject_data[f'Image {frame}'])
                        roi = filename.rsplit('_', 1)[0]

                        if roi not in rois.keys():
                            rois[roi] = {
                                'images': {},
                                'users': {}
                            }

                        if filename not in rois[roi]['images']:
                            rois[roi]['images'][filename] = [timedelta]
                        else:
                            rois[roi]['images'][filename].append(timedelta)
                        user_name = entry['user_name']
                        if user_name not in rois[roi]['users']:
                            rois[roi]['users'][user_name] = 1

                    else:
                        print('failure')

    return rois


def calculate_stats_from_rois(rois):
    stats = {}
    for roi, data in rois.items():
        stats[roi] = {
            'total_classifications': 0,
            'total_volunteer_time': 0,
            'median_classifications_per_subject': 0,  # subjects=image slices
            'median_classification_time': 0,
            'num_unique_volunteers': 0,
            'total_time_as_clock': ""
        }

        stats[roi]['num_unique_volunteers'] = len(data['users'])

        total_classifications = 0
        total_volunteer_time = 0
        classifications_per_subject = []
        classification_times = []
        for image, times in data['images'].items():
            num_times = len(times)
            total_classifications += num_times
            classifications_per_subject.append(num_times)
            filtered_times = list(filter(lambda x: 0 < x < MAX_CLASSIFICATION_TIME, times))
            subject_time_total = np.sum(filtered_times)
            total_volunteer_time += subject_time_total
            classification_times += times

        stats[roi]['total_classifications'] = int(round(total_classifications))
        stats[roi]['median_classifications_per_subject'] = int(round(np.median(classifications_per_subject)))
        stats[roi]['median_classification_time'] = int(round(np.median(classification_times)))
        stats[roi]['total_volunteer_time'] = int(round(total_volunteer_time/3600))  # hours

        volunteer_days = total_volunteer_time//(24*60*60)
        unaccounted_volunteer_time = total_volunteer_time - volunteer_days*(24*60*60)
        volunteer_hours = unaccounted_volunteer_time//(60*60)
        unaccounted_volunteer_time = unaccounted_volunteer_time - volunteer_hours*(60*60)
        volunteer_minutes = unaccounted_volunteer_time//60
        volunteer_seconds = unaccounted_volunteer_time - volunteer_minutes*60

        stats[roi]['total_time_as_clock'] = f"{int(volunteer_days)} {int(volunteer_hours)}:" \
                                             f"{int(volunteer_minutes)}:{int(volunteer_seconds)}"

    return stats


def write_stats_to_csv(stats):
    f = open(OUT_PATH, 'w')
    f.write(",".join(['ROI', 'Total classifications', 'Total volunteer time (hours)',
                      'Median classifications per subject', 'Median classification time (seconds)',
                      'No. unique volunteers', 'Total time dd hh:mm:ss']) + "\n")
    for roi, data in stats.items():
        f.write(','.join([roi, str(data['total_classifications']), str(data['total_volunteer_time']),
                          str(data['median_classifications_per_subject']), str(data['median_classification_time']),
                          str(data['num_unique_volunteers']), data['total_time_as_clock'], '\n']))
    f.close()


if __name__ == '__main__':
    csv.field_size_limit(sys.maxsize)
    zooniverse_csv_chunks = pd.read_csv(PATH,
                                        engine='c',
                                        error_bad_lines=False,
                                        chunksize=CHUNK_SIZE)

    rois = read_relevant_data_from_dataframe_chunks(zooniverse_csv_chunks)

    stats = calculate_stats_from_rois(rois)

    print('Writing output file...', end='')
    write_stats_to_csv(stats)
    print('DONE')


