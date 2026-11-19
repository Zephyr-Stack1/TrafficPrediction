from log_config import get_logger
from pathlib import Path
import numpy as np

MINUTES_PER_DAY = 24 * 60
DAY_PER_WEEK = 7
DIMENSION_NUM_3 = 3
MAX_SPEED_CHECK = 120

logger = get_logger(__name__)

class TrafficData:

    def __init__(self, x, y, validMask, timestamps, sensorIds, sampleId, features = None):
        self.x = x
        self.y = y
        self.validMask = validMask
        self.timestamps = timestamps
        self.sensorIds = sensorIds
        self.sampleId = sampleId

        self.normalizedX = None
        self.normalizedY = None

        if features is None:
            self.features = {}
        else:
            self.features = features


class TrafficNpzLoader:

    def __init__(self, dataDir = 'data'):
        self.dataDir = Path(dataDir).resolve().parent / dataDir

    def load(self, split) -> TrafficData:

        filePath = self.dataDir / (split + '.npz')
        
        logger.info(f"Loading data from {filePath}")
        
        if not filePath.is_file():
            raise FileNotFoundError('Data file not found: ' + str(filePath))

        # numpy.load can load the data to a N-dimesion array directly
        # And it also can load multiple array at same time
        with np.load(filePath, allow_pickle = False) as raw:

            required = ['X', 'valid_mask', 'timestamps', 'sensor_ids', 'sample_id']

            # To judge whether the split is test_features, 
            # we can check if the split is not equal to 'test_features'
            if split != 'test_features':
                required.append('Y')
            for key in required:
                if key not in raw.files:
                    raise ValueError('Missing NPZ key: ' + key)

            y = None
            if 'Y' in raw.files:
                y = raw['Y'].astype(np.float32)
            
            data = TrafficData(
                raw['X'].astype(np.float32),
                y,
                raw['valid_mask'],
                raw['timestamps'],
                raw['sensor_ids'],
                raw['sample_id'],
            )

        return data


class TrafficCleaner:

    def __init__(self, maxSpeed = None):
        if maxSpeed is not None:
            if not isinstance(maxSpeed, (int, float)) or isinstance(maxSpeed, bool):
                raise ValueError('maxSpeed must be a positive finite number')
            if not np.isfinite(maxSpeed) or maxSpeed <= 0:
                raise ValueError('maxSpeed must be a positive finite number')
        self.maxSpeed = maxSpeed

    def validate(self, data: TrafficData) -> None:

        x = data.x
        y = data.y

        if not isinstance(x, np.ndarray):
            raise ValueError('x must be a NumPy array')
        if x.ndim != DIMENSION_NUM_3 or min(x.shape) <= 0:
            raise ValueError('x must have shape [samples, history, sensors]')
        if x.dtype.kind not in 'fiu':
            raise ValueError('x must contain real numeric values')

        sampleCount = x.shape[0]
        sensorCount = x.shape[2]

        if not isinstance(data.validMask, np.ndarray):
            raise ValueError('validMask must be a NumPy array')
        if data.validMask.shape != (sampleCount, sensorCount):
            raise ValueError('validMask must have shape [samples, sensors]')
        if data.validMask.dtype != np.bool_:
            raise ValueError('validMask must contain boolean values')
        if not isinstance(data.sensorIds, np.ndarray):
            raise ValueError('sensorIds must be a NumPy array')
        if data.sensorIds.shape != (sensorCount,):
            raise ValueError('sensorIds has the wrong shape')
        # Unicode string integer unsigned interger
        if data.sensorIds.dtype.kind not in 'USiu':
            raise ValueError('Sensor IDs must be strings or integers')
        if np.any(np.char.strip(data.sensorIds.astype(str)) == ''):
            raise ValueError('Sensor IDs must not be empty')
        if not isinstance(data.sampleId, np.ndarray):
            raise ValueError('sampleId must be a NumPy array')
        if data.sampleId.shape != (sampleCount,):
            raise ValueError('sampleId has the wrong shape')
		# Unicode string integer unsigned interger
        if data.sampleId.dtype.kind not in 'USiu':
            raise ValueError('Sample IDs must be strings or integers')
        if np.any(np.char.strip(data.sampleId.astype(str)) == ''):
            raise ValueError('Sample IDs must not be empty')
        if not isinstance(data.timestamps, np.ndarray):
            raise ValueError('timestamps must be a NumPy array')
        if data.timestamps.shape != (sampleCount,):
            raise ValueError('timestamps has the wrong shape')
        if data.timestamps.dtype.kind != 'M' or np.isnat(data.timestamps).any():
            raise ValueError('timestamps must contain valid datetime64 values')
        if np.any(data.timestamps[1:] <= data.timestamps[:-1]):
            raise ValueError('Timestamps must be strictly increasing')
        if len(np.unique(data.sensorIds)) != sensorCount:
            raise ValueError('Sensor IDs must be unique')
        if len(np.unique(data.sampleId)) != sampleCount:
            raise ValueError('Sample IDs must be unique within each split')

        if y is not None:
            if not isinstance(y, np.ndarray):
                raise ValueError('y must be a NumPy array')
            if y.ndim != DIMENSION_NUM_3 or y.shape[0] != sampleCount:
                raise ValueError('y has the wrong sample dimension')
            if y.shape[1] <= 0 or y.shape[2] != sensorCount:
                raise ValueError('y has the wrong horizon or sensor dimension')
            if y.dtype.kind not in 'fiu':
                raise ValueError('y must contain real numeric values')

    def clean(self, data: TrafficData) -> TrafficData:

        x = data.x
        y = data.y

        validX = np.isfinite(x)
        validX[x < 0] = False

        if self.maxSpeed is not None:
            validX[x > self.maxSpeed] = False

        validHistory = validX.all(axis = 1)

        validMask = data.validMask.copy()
		
        validMask[validHistory == False] = False

        if y is not None:
            validY = np.isfinite(y)
            validY[y < 0] = False

            if self.maxSpeed is not None:
                validY[y > self.maxSpeed] = False

                validFuture = validY.all(axis = 1)
                validMask[validFuture == False] = False

        if not validMask.any():
            raise ValueError('No valid sample-sensor pairs remain after cleaning')

        data.x = np.where(validMask[:, None, :], x, 0)
        if y is not None:
            data.y = np.where(validMask[:, None, :], y, 0)
		
        data.validMask = validMask
        data.normalizedX = None
        data.normalizedY = None

        return data


class TrafficZScoreNormalizer:

    def __init__(self):
        self.mean = None
        self.std = None
        self.sensorIds = None

    def fit(self, data: TrafficData):

        sensorCount = data.x.shape[2]

        mean = np.empty(sensorCount, dtype = np.float32)
        std = np.empty(sensorCount, dtype = np.float32)

        for sensorIndex in range(sensorCount):
            validRows = data.validMask[:, sensorIndex]
            values = data.x[validRows, :, sensorIndex]

            if values.size == 0 or not np.isfinite(values).all():
                raise ValueError('No complete training data for sensor ' + str(sensorIndex))
            
            mean[sensorIndex] = np.mean(values, dtype = np.float64)
            std[sensorIndex] = np.std(values, dtype = np.float64)

            if std[sensorIndex] < 0.00000001:
                std[sensorIndex] = 1.0
        
        self.mean = mean
        self.std = std
        self.sensorIds = data.sensorIds.copy()

    def transform(self, data: TrafficData) -> TrafficData:

        if self.mean is None or self.std is None:
            raise ValueError('Normalize training data first')
        
        if not np.array_equal(data.sensorIds, self.sensorIds):
            raise ValueError('Sensor IDs or order differ from training data')

        sensorCount = data.x.shape[2]
        if self.mean.shape != (sensorCount,) or self.std.shape != (sensorCount,):
            raise ValueError('mean and std must have one value per sensor')
        
        if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all():
            raise ValueError('mean and std must contain finite values')
        
        if np.any(self.std <= 0):
            raise ValueError('std must contain positive values')

        normalizedX = (data.x - self.mean) / self.std
        normalizedY = None
        if data.y is not None:
            normalizedY = (data.y - self.mean) / self.std
        data.normalizedX = normalizedX
        data.normalizedY = normalizedY
        return data


class TrafficTimeFeatureBuilder:

    def __init__(self, stepMinutes = 5, enabled = True):
        if not isinstance(stepMinutes, int) or stepMinutes <= 0:
            raise ValueError('stepMinutes must be a positive integer')
        self.stepMinutes = stepMinutes
        self.enabled = enabled

    def build(self, data: TrafficData) -> TrafficData:
        features = {}
        if self.enabled:
            history = data.x.shape[1]
            steps = np.arange(1 - history, 1)
            offsets = steps * np.timedelta64(self.stepMinutes, 'm')
            times = data.timestamps[:, None] + offsets[None, :]
            minute = times.astype('datetime64[m]').astype(np.int64) % MINUTES_PER_DAY
            day = times.astype('datetime64[D]').astype(np.int64)

            weekday = (day + 3) % DAY_PER_WEEK
            angle = 2 * np.pi * minute / MINUTES_PER_DAY
            timeSin = np.sin(angle)
            timeCos = np.cos(angle)
            isWeekend = weekday >= 5
            features['time'] = np.stack(
                [timeSin, timeCos, isWeekend], axis=2
            ).astype(np.float32)
        data.features = features
        return data


class TrafficDataPreparer:

    def __init__(self, loader, cleaner, normalizer, featureBuilder):
        self.loader = loader
        self.cleaner = cleaner
        self.normalizer = normalizer
        self.featureBuilder = featureBuilder

    def prepare(self):
        splits = {}
        history = None
        horizon = None
        sensorIds = None

        for split in ['train', 'validation', 'test_features']:
            raw = self.loader.load(split)
            self.cleaner.validate(raw)
            clean = self.cleaner.clean(raw)

            if split == 'train':
                if clean.y is None:
                    raise ValueError('Training targets are required')
                history = clean.x.shape[1]
                horizon = clean.y.shape[1]
                sensorIds = clean.sensorIds.copy()
                self.normalizer.fit(clean)
                norm = self.normalizer.transform(clean)
            else:
                if not np.array_equal(clean.sensorIds, sensorIds):
                    raise ValueError('Sensor IDs must match across splits')
                if clean.x.shape[1] != history:
                    raise ValueError('History lengths must match across splits')
                if clean.y is not None and clean.y.shape[1] != horizon:
                    raise ValueError('Forecast horizons must match across splits')
                if raw.timestamps[0] <= splits['train'].timestamps[-1]:
                    raise ValueError('Held-out forecast origins must follow training')
                if split == 'test_features':
                    if raw.timestamps[0] <= splits['validation'].timestamps[-1]:
                        raise ValueError('Test forecast origins must follow validation')
                norm = self.normalizer.transform(clean)
            result = self.featureBuilder.build(norm)
            splits[split] = result

        return splits
    
    def saveTrafficData(self,splits, normalizer, outputDir):
        for split in ['train', 'validation', 'test_features']:
            if split not in splits:
                raise RuntimeError('Call prepare() before saveData()')
            data = splits[split]
            if data.normalizedX is None:
                raise RuntimeError('Normalized inputs are not available: ' + split)
            if split != 'test_features' and data.normalizedY is None:
                raise RuntimeError('Normalized targets are not available: ' + split)
        if normalizer.mean is None or normalizer.std is None:
            raise RuntimeError('Training normalization parameters are not available')

        outputDir = Path(outputDir)
        outputDir.mkdir(parents = True, exist_ok = True)

        for split in ['train', 'validation', 'test_features']:
            data = splits[split]
            arrays = {
                'normalizedX': data.normalizedX,
                'valid_mask': data.validMask,
                'timestamps': data.timestamps,
                'sensor_ids': data.sensorIds,
                'sample_id': data.sampleId,
            }
            if data.normalizedY is not None:
                arrays['normalizedY'] = data.normalizedY
            for name in data.features:
                arrays['feature_' + name] = data.features[name]

            np.savez_compressed(outputDir / (split + '.npz'), **arrays)

        np.savez_compressed(
        outputDir / 'norm_params.npz',
        mean = normalizer.mean,
        std = normalizer.std,
        sensor_ids = normalizer.sensorIds,
        method = np.array('zscore'),
    )


class TrafficWindowDataset:

    def __init__(self, splits):
        self.splits = splits
        self.sampleRows = {}
        self.sensorRows = {}

        for split in splits:
            data = splits[split]
            rows, sensors = np.nonzero(data.validMask)
            self.sampleRows[split] = rows
            self.sensorRows[split] = sensors

    def getBatchCount(self, split, batchSize=64):
        if not isinstance(batchSize, int) or batchSize <= 0:
            raise ValueError('batchSize must be a positive integer')
        if split not in self.splits:
            raise ValueError('Split is not prepared: ' + str(split))
        pairCount = len(self.sampleRows[split])
        return (pairCount + batchSize - 1) // batchSize

    def shuffleTraining(self, seed=42):
        if 'train' not in self.splits:
            raise RuntimeError('Call prepare() first')
        order = np.arange(len(self.sampleRows['train']))
        rng = np.random.default_rng(seed)
        rng.shuffle(order)
        self.sampleRows['train'] = self.sampleRows['train'][order]
        self.sensorRows['train'] = self.sensorRows['train'][order]

    def getBatch(self, split, batchIndex, batchSize=64):
        count = self.getBatchCount(split, batchSize)
        if not isinstance(batchIndex, int) or batchIndex < 0 or batchIndex >= count:
            raise IndexError('batchIndex is out of range')
        start = batchIndex * batchSize
        end = start + batchSize
        rows = self.sampleRows[split][start:end]
        sensors = self.sensorRows[split][start:end]
        data = self.splits[split]
        x = data.normalizedX[rows, :, sensors]
        x = np.expand_dims(x, axis=2)
        y = None
        if data.normalizedY is not None:
            y = data.normalizedY[rows, :, sensors]
            y = np.expand_dims(y, axis=2)
        features = {}
        for name in data.features:
            features[name] = data.features[name][rows]
        return {
            'x': x,
            'y': y,
            'features': features,
            'sampleIndex': rows.copy(),
            'sensorIndex': sensors.copy(),
            'sampleId': data.sampleId[rows],
            'sensorId': data.sensorIds[sensors],
            'timestamps': data.timestamps[rows],
        }


def main():
    dataDir = 'data'
    outputDir = 'processed'
    
    logger.info("Data preparation started")

    loader = TrafficNpzLoader(dataDir)
    cleaner = TrafficCleaner(MAX_SPEED_CHECK)
    normalizer = TrafficZScoreNormalizer()
    featureBuilder = TrafficTimeFeatureBuilder()

    preparer = TrafficDataPreparer(
        loader,
        cleaner,
        normalizer,
        featureBuilder,
    )

    splits = preparer.prepare()
    preparer.saveTrafficData(splits, normalizer, outputDir)

    print('Prepared files saved to:', outputDir)
    logger.info("Data preparation ended")


if __name__ == '__main__':
    main()
