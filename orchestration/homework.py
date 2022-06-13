from calendar import month
import datetime as dt
from dateutil.relativedelta import relativedelta
from pathlib import Path
import pickle

import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from prefect import flow, task, get_run_logger


def read_data(path):
    df = pd.read_parquet(path)
    return df


@task
def prepare_features(df, categorical, train=True):
    logger = get_run_logger()
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        logger.info(f"The mean duration of training is {mean_duration}")
    else:
        logger.info(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df


@task
def train_model(df, categorical):
    logger = get_run_logger()

    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    logger.info(f"The shape of X_train is {X_train.shape}")
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    logger.info(f"The MSE of training is: {mse}")
    return lr, dv


@task
def run_model(df, categorical, dv, lr):
    logger = get_run_logger()
    
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    logger.info(f"The MSE of validation is: {mse}")
    return


def _to_path(base_path, date: dt.date):
    return base_path / ("fhv_tripdata_%s-%02d.parquet" % (date.year, date.month))


@task
def get_paths(arg_date: dt.date, base_path: Path) -> tuple[str, str]:
    if not arg_date:
        arg_date = dt.date.today()
    train_date = arg_date - relativedelta(months=2)
    val_date = arg_date - relativedelta(months=1)
    train_path = _to_path(base_path, train_date)
    val_path = _to_path(base_path, val_date)
    get_run_logger().info("Train path: %s, val path: %s", train_path, val_path)
    return train_path, val_path


@flow(name="NYC taxi duration model training")
def main(base_path: str, date: dt.date = None):
    train_path, val_path = get_paths(date, Path(base_path)).result()

    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()
    date_str = date.strftime("%Y-%m-%d")
    # TODO
    output_dir = Path('models')
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"model-{date_str}.bin").open(mode='wb') as f_out:
        pickle.dump(lr, f_out)
    with (output_dir / f"dv-{date_str}.b").open(mode="wb") as f_out:
        pickle.dump(dv, f_out)
    run_model(df_val_processed, categorical, dv, lr)


def run_main_flow(base_path, date = None):
    main(base_path, date)


if __name__ == "__main__":
    import fire
    fire.Fire(run_main_flow)
else:
    from prefect.deployments import DeploymentSpec
    from prefect.orion.schemas.schedules import CronSchedule
    DeploymentSpec(
        name="NYC taxi duration model training on current date",
        flow=main,
        schedule=CronSchedule(cron="0 9 15 * *")
    )