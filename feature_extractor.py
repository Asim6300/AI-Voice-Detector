import librosa
import numpy as np

def extract_features(path):

    try:

        import soundfile as sf

        y,sr=sf.read(path)

        if len(y.shape)>1:
            y=np.mean(y,axis=1)

        y=librosa.resample(
            y,
            orig_sr=sr,
            target_sr=16000
        )

        sr=16000

    except:

        y,sr=librosa.load(
            path,
            sr=16000,
            mono=True
        )

    y,_=librosa.effects.trim(y)

    mfcc=np.mean(
        librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=20
        ),
        axis=1
    )

    zcr=np.mean(
        librosa.feature.zero_crossing_rate(y)
    )

    centroid=np.mean(
        librosa.feature.spectral_centroid(
            y=y,
            sr=sr
        )
    )

    rolloff=np.mean(
        librosa.feature.spectral_rolloff(
            y=y,
            sr=sr
        )
    )

    rms=np.mean(
        librosa.feature.rms(
            y=y
        )
    )

    pitches,_=librosa.piptrack(
        y=y,
        sr=sr
    )

    pitch=np.mean(
        pitches[pitches>0]
    ) if np.any(pitches>0) else 0

    features=np.concatenate([
        mfcc,
        [
            zcr,
            centroid,
            rolloff,
            rms,
            pitch
        ]
    ])

    return features