import json

import yaml


def get_dstc_service_name(service_name: str) -> str:
    i = service_name.find("_")
    if i == -1:
        return service_name
    return service_name[: service_name.find("_")]


def read_json(path: str):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def register_contants(class_ref, default_values_path, override_values):
    with open(default_values_path, "r") as f:
        default_values = yaml.safe_load(f)
    for k, v in default_values.items():
        if k in override_values:
            v = override_values[k]
        setattr(class_ref, k, v)


def get_list_of_attrs(class_ref):
    return [
        getattr(class_ref, a)
        for a in dir(class_ref)
        if not a.startswith("__") and not a.startswith("_") and not callable(getattr(class_ref, a))
    ]

def remove_underscore(item:str):
    return item.replace("_", " ")

def extract_characters(string):
  #Using filter to get the characters
    return "".join(filter(lambda x: x.isalpha(), string))