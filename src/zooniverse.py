import os
import csv
import json
import shutil

import pandas as pd
from tqdm import tqdm


# how many csv rows should we load in to memory at one time
CHUNK_SIZE = 100


class ZooniverseCSVParser:
    """
    Breaks the zooniverse csv format apart in to smaller csvs, preserving the data we are particularly interested in.
    We load in chunks, so even very large csv files which wouldn't fit in memory can be converted.
    """

    def __init__(self, output_dir, workflow=None, tool_label=None):
        self.errors = dict(invalid_format=0, missing_ref_image=0)
        self.processed = 0
        self.output_dir = output_dir
        self.workflow = workflow
        self.tool_label = tool_label

    def prepare_output_dir(self):
        """
        Delete everything in the directory where we are writing output csvs. Then make sure the folder exists.
        """
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def convert(self, zooniverse_csv_path):
        # since we're appending to csvs, we need to clear first to avoid duplication
        self.prepare_output_dir()

        csv.field_size_limit(2**30)
        zooniverse_csv_chunks = pd.read_csv(zooniverse_csv_path,
                                            engine='c',
                                            error_bad_lines=False,
                                            chunksize=CHUNK_SIZE)

        for chunk in tqdm(zooniverse_csv_chunks,  unit=' rows', unit_scale=CHUNK_SIZE):
            chunk = chunk[chunk.workflow_name == self.workflow]

            for index, row in chunk.iterrows():
                try:
                    parsed_entries = self.parse_row(row)
                    self.write_parsed_entries(parsed_entries)
                except MissingRefImageError:
                    self.errors['missing_ref_image'] += 1
                except InvalidAnnotationFormatError:
                    self.errors['invalid_format'] += 1

            self.processed += len(chunk.index)

    def censor(self, zooniverse_csv_path, output_path):
        csv.field_size_limit(2**30)
        zooniverse_csv_chunks = pd.read_csv(zooniverse_csv_path,
                                            engine='c',
                                            error_bad_lines=False,
                                            chunksize=CHUNK_SIZE)

        header_done = False
        with open(output_path, mode='w', newline='') as censored_csv:
            csv_writer = csv.writer(censored_csv)
            for chunk in tqdm(zooniverse_csv_chunks,  unit=' rows', unit_scale=CHUNK_SIZE):
                for index, row in chunk.iterrows():
                    row['user_name'] = ''
                    row['user_id'] = ''
                    row['user_ip'] = ''
                    if not header_done:
                        csv_writer.writerow(chunk.columns)
                        header_done = True
                    csv_writer.writerow(row)

    def parse_row(self, row):
        """
        Parses one row of the zooniverse csv, and extracts the parameters we are interested in.
        """
        parsed_entries = []

        annotations = json.loads(row.annotations)
        annotations = annotations[0]  # seems to always be a list of only one element

        subject_data = json.loads(row.subject_data)[str(row.subject_ids)]

        description = subject_data['description'] if 'description' in subject_data.keys() else ''
        attribution = subject_data['attribution'] if 'attribution' in subject_data.keys() else ''
        microscope = subject_data['microscope'] if 'microscope' in subject_data.keys() else ''
        raw_z_res = subject_data['Raw Z resolution (nm)'] if 'Raw Z resolution (nm)' in subject_data.keys() else 5
        raw_xy_res = subject_data['Raw XY resolution (nm)'] if 'Raw XY resolution (nm)' in subject_data.keys() else 5
        jpeg_quality = subject_data['jpeg quality (%)'] if 'jpeg quality (%)' in subject_data.keys() else 100
        scaling_factor = subject_data['Scaling factor'] if 'Scaling factor' in subject_data.keys() else 1
        retired = "\"" + str(subject_data['retired']) + "\"" if 'retired' in subject_data.keys() else ''
        subject_id = subject_data['Subject ID'] if 'Subject ID' in subject_data.keys() else 0

        # i.e. the 5 slices from the zooniverse classification screen
        slices = self.slices_from_annotations(annotations, subject_data)

        # NOTE: this is where to modify the code if we don't want to use the other 4 slices from the zooniverse screen
        for z_slice in slices.values():
            slice_filename = z_slice['image']
            slice_filename_parts = slice_filename.rsplit('_', 1)

            filename, extension = os.path.splitext(z_slice['image'])

            # dictionaries preserve order by default as long as we have python 3.7+ (3.6+ for CPython)
            parsed_entry = {
                'classification id': row.classification_id,

                # from subject data
                'raw xy resolution (nm)': raw_xy_res,
                'raw z resolution (nm)': raw_z_res,
                'microscope': microscope,
                'jpeg quality (%)': jpeg_quality,
                'attribution': attribution,
                'subject id': subject_id,
                'scaling factor': scaling_factor,

                'ROI': slice_filename_parts[0],
                'slice z': slice_filename_parts[1][:-4],
                'image_filename': slice_filename,

                'filename': filename,
                'extension': extension,

                'annotations': "\"" + str(z_slice['annotations']) + "\"",

                # new for mitochondria csv
                'tool': z_slice['tool'],
                'tool_label': z_slice['tool_label'],
                'expert': row.expert is True,
                'workflow_id': row.workflow_id,
                'workflow_name': row.workflow_name
            }

            parsed_entries.append(parsed_entry)

        return parsed_entries

    def write_parsed_entries(self, parsed_entries):
        for parsed_entry in parsed_entries:
            filename = self.output_dir+parsed_entry['filename']+'.csv'
            exists = os.path.exists(filename)
            f = open(filename, 'a')
            if not exists:
                f.write(','.join(parsed_entry.keys()) + '\n')
            values = [str(entry) for entry in parsed_entry.values()]
            f.write(','.join(values) + '\n')
            f.close()

    def slices_from_annotations(self, annotations, subject_data):
        """
        The annotations are a list of lists (rough idea being one inner list corresponds to one mitochondria).

        Each of the inner lists also has a slice specified, since the user can perform classifications on up to 5 different
        slices we assign a list for each different slice, so that we can match up classifications to the correct image.
        We keep the inner lists separate, as they can be considered closed polygons with the shape tool.
        """
        slices = dict()

        for annotation in annotations['value']:
            if type(annotation) is dict:
                frame = annotation['frame']
                points = []
                for point in annotation['points']:
                    points.append((point['x'], point['y']))
                if f'Image {frame}' in subject_data.keys():
                    if frame not in slices:
                        slices[frame] = dict()
                        slices[frame]['annotations'] = []
                    slices[frame]['tool'] = annotation['tool']
                    slices[frame]['tool_label'] = annotation['tool_label']
                    slices[frame]['image'] = subject_data[f'Image {frame}']
                    slices[frame]['annotations'].append(points)
                else:
                    raise MissingRefImageError
            else:
                raise InvalidAnnotationFormatError

        return slices


class MissingRefImageError(Exception):
    pass


class InvalidAnnotationFormatError(Exception):
    pass

