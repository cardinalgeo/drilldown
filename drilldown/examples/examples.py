import pandas as pd
import pyvista as pv
import os


def load_assay_data(data_path):
    data = pd.read_csv(data_path)

    return data


def load_collar_data(data_path):
    data = pd.read_csv(data_path)

    return data


def load_survey_data(data_path):
    data = pd.read_csv(data_path)

    return data


def load_topo_data(data_path):
    data = pv.read(data_path)

    return data


def load_tom_zone_macpass_project():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    assay_data = load_assay_data(
        os.path.join(dir_path, "data/macpass_project/tom_zone/assay.csv")
    )
    collar_data = load_collar_data(
        os.path.join(dir_path, "data/macpass_project/tom_zone/collar.csv")
    )
    survey_data = load_survey_data(
        os.path.join(dir_path, "data/macpass_project/tom_zone/survey.csv")
    )
    topo_data = load_topo_data(
        os.path.join(dir_path, "data/macpass_project/tom_zone/topo.ply")
    )

    return {
        "assay": assay_data,
        "collar": collar_data,
        "survey": survey_data,
        "topo": topo_data,
    }


def load_forrestania():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    assay_data = load_assay_data(os.path.join(dir_path, "data/forrestania/assay.csv"))
    collar_data = load_collar_data(
        os.path.join(dir_path, "data/forrestania/collar.csv")
    )
    survey_data = load_survey_data(
        os.path.join(dir_path, "data/forrestania/survey.csv")
    )

    return {"assay": assay_data, "collar": collar_data, "survey": survey_data}


def load_copper_creek():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    assay_data = load_assay_data(os.path.join(dir_path, "data/copper_creek/assay.csv"))
    collar_data = load_collar_data(
        os.path.join(dir_path, "data/copper_creek/collar.csv")
    )
    survey_data = load_survey_data(
        os.path.join(dir_path, "data/copper_creek/survey.csv")
    )

    return {"assay": assay_data, "collar": collar_data, "survey": survey_data}
