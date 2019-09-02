# -*- coding: utf-8 -*
import sys

import os
import yaml


class Yaml:
    def __init__(self, path=os.path.join(sys.path[0], 'app.yaml')):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.load(f)
                self.data = data
        except FileNotFoundError:
            self.data = ""
            pass

    def data_config(self):
        if 'data' in self.data:
            return self.data['data']

    def server_config(self):
        if 'server' in self.data:
            return self.data['servers']

    def model_config(self):
        if 'model' in self.data:
            return self.data['model']

    def get_input_names(self):
        if 'model' in self.data:
            config = self.data['model']
            input = config['input']
            names = []
            for columns in input['columns']:
                names.append(columns['name'])
            return names

    def get_input_shape(self):
        config = self.data['model']
        input = config['input']
        return input['shape']

    def get_output_names(self):
        if 'model' in self.data:
            config = self.data['model']
            input = config['output']
            names = []
            for columns in input['columns']:
                names.append(columns['name'])
            return names

    def get_output_shape(self):
        config = self.data['model']
        input = config['output']
        return input['shape']

    def get_data_id(self):
        if 'data' in self.data:
            return self.data['data']['id']

    def get_servers(self):
        if 'servers' in self.data:
            return self.data['servers']

    def processor(self):
        if 'model' in self.data:
            processor = dict()
            processor['processor'] = self.data['model']['processor']
            processor['input_x'] = self.data['model']['input_x']
            processor['input_y'] = self.data['model']['input_y']
            try:
                output_x = self.data['model']['output_x']
            except:
                output_x = self.data['model']['input_x']
            processor['output_x'] = output_x
            processor['output_y'] = self.data['model']['output_y']
            return processor
