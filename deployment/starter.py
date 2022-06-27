#!/usr/bin/env python
# coding: utf-8
import pickle
from pathlib import Path
from typing import Tuple

import pandas as pd


def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[_CATEGORICAL] = df[_CATEGORICAL].fillna(-1).astype('int').astype('str')
    
    return df


def load_models(model_path):
    with open(model_path, 'rb') as f_in:
        return pickle.load(f_in)


_CATEGORICAL = ['PUlocationID', 'DOlocationID']


def make_input_filename(year, month):
    return f"fhv_tripdata_{year:04d}-{month:02d}.parquet"


def predict_on_paths(
        input_year:int,
        input_month:int,
        input_dir="/Users/rgareev/data/ny-tlc/src",
        model_path='model.bin',
        output_path='./wrk/predictions.parquet'):
    dv, lr = load_models(model_path)
    input_path = Path(input_dir) / make_input_filename(input_year, input_month)
    df = read_data(input_path)

    dicts = df[_CATEGORICAL].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)
    print(y_pred.mean())

    df['predicted_duration'] = y_pred
    df['ride_id'] =  df.pickup_datetime.apply(lambda dt: f'{dt.year:04d}/{dt.month:02d}_') + df.index.astype('str')

    output_path = Path(output_path)
    if len(output_path.parts) > 1:
        output_path.parent.mkdir(exist_ok=True, parents=True)
    df[['ride_id', 'predicted_duration']].to_parquet(
        output_path,
        engine='pyarrow',
        compression=None,
        index=False
    )


if __name__=='__main__':
    from fire import Fire
    Fire(predict_on_paths)