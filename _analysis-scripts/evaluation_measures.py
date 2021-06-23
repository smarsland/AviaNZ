# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Modified from the DCASE reference version by:
#########################################################################
# Initial software, Nicolas Turpault, Romain Serizel, Hamid Eghbal-zadeh, Ankit Parag Shah
# Copyright © INRIA, 2018, v1.0
# This software is distributed under the terms of the License MIT
#########################################################################

# Will print evaluation details.
## USAGE: python evaluation_measures.py REFFILE ESTFILE > OUTFILE

import sys
from dcase_util.data import ProbabilityEncoder
import sed_eval
import numpy
import pandas as pd

def get_event_list_current_file(df, fname):
    """
    Get list of events for a given filename
    :param df: pd.DataFrame, the dataframe to search on
    :param fname: the filename to extract the value from the dataframe
    :return: list of events (dictionaries) for the given filename
    """
    event_file = df[df["filename"].str.contains(fname)]
    if len(event_file) == 1:
        if pd.isna(event_file["event_label"].iloc[0]):
            event_list_for_current_file = [{"filename": fname}]
        else:
            event_list_for_current_file = event_file.to_dict('records')
    else:
        event_list_for_current_file = event_file.to_dict('records')

    return event_list_for_current_file

def event_based_evaluation_df(reffile, estfile, sound="Cat"):
    """
    Calculate EventBasedMetric given a reference and estimated dataframe
    :param reference: csv containing "filename" "onset" "offset" and "event_label" columns which describe the
    reference events
    :param estimated: csv containing "filename" "onset" "offset" and "event_label" columns which describe the
    estimated events to be compared with reference
    :param sound: string identifying which class to calculate this on
    :return: sed_eval.sound_event.EventBasedMetrics with the scores
    """
    reference = pd.read_csv(reffile, sep='\t')
    estimated = pd.read_csv(estfile, sep='\t')
    #print("reference events: %d, predicted: %d" %(len(reference), len(estimated)))
    # print("Selecting class %s:" % sound)
    # def get_class(df, cl):
    #     return df[df["event_label"].isnull() | df["event_label"].str.contains(cl)]
    #reference = get_class(reference, sound)
    #estimated = get_class(estimated, sound)
    print("reference events: %d, predicted: %d" %(len(reference), len(estimated)))

    evaluated_files = reference["filename"].unique()
    classes = []
    classes.extend(reference.event_label.dropna().unique())
    classes.extend(estimated.event_label.dropna().unique())
    event_based_metric = sed_eval.sound_event.EventBasedMetrics(
        event_label_list=classes,
        t_collar=0.200,
        percentage_of_length=0.2,
        empty_system_output_handling='zero_score'
    )

    for fname in evaluated_files:
        print("FILE", fname)
        reference_event_list_for_current_file = get_event_list_current_file(reference, fname)
        estimated_event_list_for_current_file = get_event_list_current_file(estimated, fname)

        event_based_metric.evaluate(
            reference_event_list=reference_event_list_for_current_file,
            estimated_event_list=estimated_event_list_for_current_file,
        )
    print(event_based_metric)
    return event_based_metric


REFFILE = sys.argv[1]
ESTFILE = sys.argv[2]
event_based_evaluation_df(REFFILE, ESTFILE, "Cat")
