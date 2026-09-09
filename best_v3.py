"""
LOTO 7/39 — KOMPLETAN TEST PREPOZNAVANJA SKRIVENE ZAKONITOSTI

Automatsko rangiranje porodica generatora
→ simbolička regresija i modularne rekurencije
→ detekcija skrivenih režima
→ Hidden Markov / switching-state model
→ k-mer, suffix i minimizer pretraga
→ sekvencijalno poravnanje
→ spektralna, autokorelaciona i entropijska analiza
→ De Bruijn graf nastavaka
→ gradient-boosting i Extra Trees regresioni ansambl
→ vremenski Transformer
→ Bajesovo ponderisanje kandidata
→ nested walk-forward validacija
→ zaključani završni holdout
→ jedna NEXT predikcija

Kombinacije se tokom računanja čuvaju kao njihovi jedinstveni rangovi. 
Svaki rang jednoznačno predstavlja jednu Loto 7/39
kombinaciju i može se bez gubitka pretvoriti nazad u sedam brojeva.
"""

from __future__ import annotations

import math
import os
import random
import tempfile
import time
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from scipy.optimize import minimize
from scipy.special import logsumexp

from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

try:
    from hmmlearn.hmm import GaussianHMM
except ImportError as greska:
    raise SystemExit(
        "\nNedostaje hmmlearn.\n"
        "Instalacija:\n\n"
        "pip install hmmlearn\n"
    ) from greska

try:
    from numba import njit
except ImportError as greska:
    raise SystemExit(
        "\nNedostaje numba.\n"
        "Instalacija:\n\n"
        "pip install numba\n"
    ) from greska

try:
    import torch
    import torch.nn as nn
except ImportError as greska:
    raise SystemExit(
        "\nNedostaje PyTorch.\n"
        "Instalacija:\n\n"
        "pip install torch\n"
    ) from greska


# =============================================================================
# PODEŠAVANJA
# =============================================================================

SEED = 39

BROJ_KUGLICA = 39
BROJEVA_U_KOMBINACIJI = 7

BROJ_SVIH_KOMBINACIJA = math.comb(
    BROJ_KUGLICA,
    BROJEVA_U_KOMBINACIJI,
)


LOTO_CSV = (
    "/Users/4c/Desktop/GHQ/data/"
    "loto7_4682_k72.csv"
)

UDEO_RAZVOJ = 0.60
UDEO_VALIDACIJA = 0.20
UDEO_HOLDOUT = 0.20


# Podela 60% / 20% / 20%.
UDEO_RAZVOJ = 0.60
UDEO_VALIDACIJA = 0.20
UDEO_HOLDOUT = 0.20



# Uzorci za skupe modele. Zaključani simbolički model ipak se proverava
# na svim kombinacijama.
SIMBOLICKI_UZORAK = 350_000
HMM_UZORAK = 100_000
ML_UZORAK = 160_000
TRANSFORMER_UZORAK = 80_000

BROJ_REZIMA_MIN = 1
BROJ_REZIMA_MAX = 16

DUZINA_SEKVENCE = 8

KMER_DUZINA = 4
MINIMIZER_PROZOR = 8
BROJ_SLICNIH_REGIONA = 100

TRANSFORMER_DIMENZIJA = 32
TRANSFORMER_GLAVE = 4
TRANSFORMER_SLOJEVI = 2
TRANSFORMER_EPOHE = 4
TRANSFORMER_BATCH = 512

EPS = 1e-12

warnings.filterwarnings("ignore")
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)


# =============================================================================
# ISPIS
# =============================================================================

def naslov(tekst: str, znak: str = "=") -> None:
    print()
    print(znak * 78)
    print(tekst)
    print(znak * 78)


def status(naziv: str, prosao: bool, dodatak: str = "") -> None:
    oznaka = "PROŠLO" if prosao else "NIJE PROŠLO"

    if dodatak:
        print(f"{naziv:<48} {oznaka:<13} {dodatak}")
    else:
        print(f"{naziv:<48} {oznaka}")


# =============================================================================
# RANGIRANJE LOTO KOMBINACIJA
# =============================================================================

def kombinacija_u_rang(kombinacija: list[int]) -> int:
    kombinacija = sorted(kombinacija)

    if len(kombinacija) != BROJEVA_U_KOMBINACIJI:
        raise ValueError("Kombinacija mora sadržati sedam brojeva.")

    if len(set(kombinacija)) != BROJEVA_U_KOMBINACIJI:
        raise ValueError("Brojevi u kombinaciji moraju biti različiti.")

    if kombinacija[0] < 1 or kombinacija[-1] > BROJ_KUGLICA:
        raise ValueError("Brojevi moraju biti između 1 i 39.")

    rang = 0
    prethodni = 0

    for indeks, broj in enumerate(kombinacija):
        preostalo = BROJEVA_U_KOMBINACIJI - indeks - 1

        for kandidat in range(prethodni + 1, broj):
            rang += math.comb(
                BROJ_KUGLICA - kandidat,
                preostalo,
            )

        prethodni = broj

    return rang


def rang_u_kombinaciju(rang: int) -> list[int]:
    if rang < 0 or rang >= BROJ_SVIH_KOMBINACIJA:
        raise ValueError("Rang kombinacije nije u dozvoljenom opsegu.")

    rezultat = []
    pocetak = 1

    for preostalo in range(
        BROJEVA_U_KOMBINACIJI,
        0,
        -1,
    ):
        for broj in range(
            pocetak,
            BROJ_KUGLICA + 1,
        ):
            broj_nastavaka = math.comb(
                BROJ_KUGLICA - broj,
                preostalo - 1,
            )

            if rang < broj_nastavaka:
                rezultat.append(broj)
                pocetak = broj + 1
                break

            rang -= broj_nastavaka

    return rezultat


def formatiraj_kombinaciju(kombinacija: list[int]) -> str:
    return ", ".join(
        f"{broj:02d}"
        for broj in kombinacija
    )


def ucitaj_csv(putanja: str) -> np.ndarray:
    okvir = pd.read_csv(
        putanja,
        header=None,
    )

    okvir = okvir.apply(
        pd.to_numeric,
        errors="coerce",
    ).dropna()

    if okvir.shape[1] < BROJEVA_U_KOMBINACIJI:
        raise ValueError(
            f"CSV mora imati najmanje "
            f"{BROJEVA_U_KOMBINACIJI} kolona."
        )

    kombinacije = okvir.iloc[
        :,
        :BROJEVA_U_KOMBINACIJI,
    ].astype(int).to_numpy()

    rangovi = np.empty(
        len(kombinacije),
        dtype=np.int64,
    )

    for indeks, red in enumerate(kombinacije):
        brojevi = sorted(
            int(broj)
            for broj in red
        )

        if len(set(brojevi)) != BROJEVA_U_KOMBINACIJI:
            raise ValueError(
                f"Red {indeks + 1} sadrži ponovljene brojeve."
            )

        if brojevi[0] < 1 or brojevi[-1] > BROJ_KUGLICA:
            raise ValueError(
                f"Red {indeks + 1} sadrži broj van opsega 1–39."
            )

        rangovi[indeks] = kombinacija_u_rang(
            brojevi
        )

    if len(rangovi) < 100:
        raise ValueError(
            "CSV nema dovoljno istorijskih izvlačenja."
        )

    return rangovi


def granice_podele(
    broj_redova: int,
) -> tuple[int, int]:
    razvoj_kraj = int(
        broj_redova * UDEO_RAZVOJ
    )

    validacija_kraj = int(
        broj_redova
        * (UDEO_RAZVOJ + UDEO_VALIDACIJA)
    )

    razvoj_kraj = max(
        razvoj_kraj,
        DUZINA_SEKVENCE + 20,
    )

    validacija_kraj = max(
        validacija_kraj,
        razvoj_kraj + 10,
    )

    validacija_kraj = min(
        validacija_kraj,
        broj_redova - 1,
    )

    return razvoj_kraj, validacija_kraj



# =============================================================================
# SKRIVENI GENERATOR
# =============================================================================

# Pattern-recognition funkcije ne dobijaju ove koeficijente.
SKRIVENI_BROJ_REZIMA = 7

SKRIVENI_A = np.asarray(
    [
        10001,
        17011,
        23003,
        31013,
        47017,
        59009,
        71023,
    ],
    dtype=np.int64,
)

SKRIVENI_C = np.asarray(
    [
        104729,
        130363,
        169087,
        224737,
        275015,
        350377,
        425623,
    ],
    dtype=np.int64,
)


@njit
def generisi_skriveni_niz(
    broj_stanja: int,
    modul: int,
    pocetno_stanje: int,
    a: np.ndarray,
    c: np.ndarray,
) -> np.ndarray:
    niz = np.empty(
        broj_stanja,
        dtype=np.int64,
    )

    niz[0] = pocetno_stanje
    broj_rezima = len(a)

    for i in range(1, broj_stanja):
        prethodno = niz[i - 1]
        rezim = prethodno % broj_rezima

        niz[i] = (
            a[rezim] * prethodno
            + c[rezim]
        ) % modul

    return niz


# =============================================================================
# SIMBOLIČKA REGRESIJA I MODULARNE REKURENCIJE
# =============================================================================

@dataclass
class ModularniModel:
    naziv: str
    broj_rezima: int
    a: np.ndarray
    c: np.ndarray

    def predvidi_jedan(self, stanje: int) -> int:
        rezim = stanje % self.broj_rezima

        return int(
            (
                int(self.a[rezim]) * int(stanje)
                + int(self.c[rezim])
            ) % BROJ_SVIH_KOMBINACIJA
        )

    def predvidi(self, stanja: np.ndarray) -> np.ndarray:
        rezimi = stanja % self.broj_rezima

        return (
            self.a[rezimi] * stanja
            + self.c[rezimi]
        ) % BROJ_SVIH_KOMBINACIJA


def pronadji_modularni_par(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[int, int] | None:
    """
    Iz dva odgovarajuća prelaza rešava:

        y = a*x + c mod M

    a zatim proverava rešenje na svim prosleđenim prelazima.
    """

    maksimalno = min(len(x), 500)

    for prvi in range(maksimalno):
        x1 = int(x[prvi])
        y1 = int(y[prvi])

        for drugi in range(prvi + 1, maksimalno):
            x2 = int(x[drugi])
            y2 = int(y[drugi])

            razlika_x = (
                x2 - x1
            ) % BROJ_SVIH_KOMBINACIJA

            if math.gcd(
                razlika_x,
                BROJ_SVIH_KOMBINACIJA,
            ) != 1:
                continue

            inverz = pow(
                razlika_x,
                -1,
                BROJ_SVIH_KOMBINACIJA,
            )

            a = (
                (y2 - y1) * inverz
            ) % BROJ_SVIH_KOMBINACIJA

            c = (
                y1 - a * x1
            ) % BROJ_SVIH_KOMBINACIJA

            predikcija = (
                a * x + c
            ) % BROJ_SVIH_KOMBINACIJA

            if np.array_equal(predikcija, y):
                return int(a), int(c)

    return None


def prilagodi_modularni_model(
    niz: np.ndarray,
    broj_rezima: int,
    kraj_obuke: int,
) -> ModularniModel | None:
    dostupni_kraj = min(
        kraj_obuke,
        SIMBOLICKI_UZORAK,
        len(niz) - 1,
    )

    prethodna = niz[:dostupni_kraj]
    naredna = niz[1:dostupni_kraj + 1]

    koeficijenti_a = []
    koeficijenti_c = []

    for rezim in range(broj_rezima):
        maska = (
            prethodna % broj_rezima
        ) == rezim

        x = prethodna[maska]
        y = naredna[maska]

        if len(x) < 3:
            return None

        resenje = pronadji_modularni_par(
            x,
            y,
        )

        if resenje is None:
            return None

        a, c = resenje
        koeficijenti_a.append(a)
        koeficijenti_c.append(c)

    return ModularniModel(
        naziv=f"Modularni model sa {broj_rezima} režima",
        broj_rezima=broj_rezima,
        a=np.asarray(
            koeficijenti_a,
            dtype=np.int64,
        ),
        c=np.asarray(
            koeficijenti_c,
            dtype=np.int64,
        ),
    )


def rangiraj_porodice_generatora(
    niz: np.ndarray,
    razvoj_kraj: int,
    validacija_kraj: int,
) -> tuple[ModularniModel | None, list[dict]]:
    kandidati = []

    for broj_rezima in range(
        BROJ_REZIMA_MIN,
        BROJ_REZIMA_MAX + 1,
    ):
        model = prilagodi_modularni_model(
            niz=niz,
            broj_rezima=broj_rezima,
            kraj_obuke=razvoj_kraj,
        )

        if model is None:
            continue

        kraj_provere = min(
            validacija_kraj,
            len(niz) - 1,
        )

        validacioni_x = niz[
            razvoj_kraj:kraj_provere
        ]

        validacioni_y = niz[
            razvoj_kraj + 1:kraj_provere + 1
        ]

        if len(validacioni_y) == 0:
            continue

        predikcije = model.predvidi(
            validacioni_x
        )

        razlika = np.abs(
            predikcije.astype(np.int64)
            - validacioni_y.astype(np.int64)
        )

        broj_gresaka = int(
            np.sum(predikcije != validacioni_y)
        )

        srednja_greska = float(
            np.mean(
                np.minimum(
                    razlika,
                    BROJ_SVIH_KOMBINACIJA - razlika,
                )
            )
        )

        kandidati.append(
            {
                "model": model,
                "greske": broj_gresaka,
                "srednja_greska": srednja_greska,
                "slozenost": broj_rezima,
            }
        )

    kandidati.sort(
        key=lambda rezultat: (
            rezultat["greske"],
            rezultat["srednja_greska"],
            rezultat["slozenost"],
        )
    )

    if not kandidati:
        return None, []

    return kandidati[0]["model"], kandidati



# =============================================================================
# SKRIVENI REŽIMI I HIDDEN MARKOV MODEL
# =============================================================================

def napravi_rezimske_osobine(
    niz: np.ndarray,
    maksimum: int,
) -> np.ndarray:
    deo = niz[:maksimum].astype(np.float64)

    return np.column_stack(
        [
            deo / BROJ_SVIH_KOMBINACIJA,
            (deo % 7) / 7.0,
            (deo % 11) / 11.0,
            (deo % 13) / 13.0,
            (deo % 17) / 17.0,
        ]
    )


def detektuj_rezime(
    niz: np.ndarray,
    razvoj_kraj: int,
) -> dict:
    osobine = napravi_rezimske_osobine(
        niz,
        min(
            HMM_UZORAK,
            razvoj_kraj,
        ),
    )

    rezultati = []

    najveci_broj_rezima = min(
        10,
        max(2, len(osobine) // 20),
    )

    for broj_rezima in range(
        2,
        najveci_broj_rezima + 1,
    ):
        model = GaussianMixture(
            n_components=broj_rezima,
            covariance_type="diag",
            n_init=3,
            random_state=SEED,
        )

        model.fit(osobine)

        rezultati.append(
            (
                float(model.bic(osobine)),
                broj_rezima,
                model,
            )
        )

    rezultati.sort(
        key=lambda rezultat: rezultat[0]
    )

    _, izabrani_broj, izabrani_model = rezultati[0]

    return {
        "broj_rezima": izabrani_broj,
        "model": izabrani_model,
        "osobine": osobine,
    }


def prilagodi_hmm(
    osobine: np.ndarray,
    broj_rezima: int,
) -> dict:
    hmm = GaussianHMM(
        n_components=broj_rezima,
        covariance_type="diag",
        n_iter=150,
        tol=1e-4,
        random_state=SEED,
        min_covar=1e-5,
    )

    hmm.fit(osobine)
    stanja = hmm.predict(osobine)

    return {
        "model": hmm,
        "stanja": stanja,
        "broj_stanja": len(np.unique(stanja)),
    }


# =============================================================================
# K-MER, SUFFIX I MINIMIZER
# =============================================================================

def stabilni_hash(vrednosti: tuple[int, ...]) -> int:
    rezultat = 1469598103934665603

    for vrednost in vrednosti:
        rezultat ^= int(vrednost) + 0x9E3779B97F4A7C15
        rezultat *= 1099511628211
        rezultat &= (1 << 64) - 1

    return rezultat


def napravi_kmer_indeks(
    niz: np.ndarray,
    kraj: int,
) -> dict:
    simboli = (
        niz[:kraj] % 4096
    ).astype(np.int32)

    indeks = defaultdict(list)

    for pozicija in range(
        KMER_DUZINA,
        len(simboli),
    ):
        kmer = tuple(
            int(x)
            for x in simboli[
                pozicija - KMER_DUZINA:
                pozicija
            ]
        )

        indeks[kmer].append(pozicija)

    return {
        "simboli": simboli,
        "indeks": indeks,
    }


def napravi_minimizer_indeks(
    simboli: np.ndarray,
) -> dict[int, list[int]]:
    rezultat = defaultdict(list)

    if len(simboli) < MINIMIZER_PROZOR:
        return rezultat

    for kraj in range(
        MINIMIZER_PROZOR,
        len(simboli) + 1,
    ):
        deo = simboli[
            kraj - MINIMIZER_PROZOR:kraj
        ]

        hash_vrednosti = []

        for pocetak in range(
            0,
            MINIMIZER_PROZOR - KMER_DUZINA + 1,
        ):
            kmer = tuple(
                int(x)
                for x in deo[
                    pocetak:
                    pocetak + KMER_DUZINA
                ]
            )

            hash_vrednosti.append(
                stabilni_hash(kmer)
            )

        minimizer = min(hash_vrednosti)
        rezultat[minimizer].append(kraj)

    return rezultat


def suffix_pretraga(
    simboli: np.ndarray,
    upit: np.ndarray,
) -> list[int]:
    kandidati = []

    for duzina in range(
        len(upit),
        0,
        -1,
    ):
        suffix = tuple(
            int(x)
            for x in upit[-duzina:]
        )

        for kraj in range(
            duzina,
            len(simboli),
        ):
            kandidat = tuple(
                int(x)
                for x in simboli[
                    kraj - duzina:kraj
                ]
            )

            if kandidat == suffix:
                kandidati.append(kraj)

        if kandidati:
            break

    return kandidati


# =============================================================================
# SEKVENCIJALNO PORAVNANJE
# =============================================================================

def sekvencijalno_poravnanje(
    upit: np.ndarray,
    kandidat: np.ndarray,
) -> float:
    if len(upit) != len(kandidat):
        raise ValueError(
            "Sekvence moraju imati istu dužinu."
        )

    tezine = np.linspace(
        0.5,
        1.5,
        len(upit),
    )

    tezine /= tezine.sum()

    maksimalna_razlika = float(
        BROJ_SVIH_KOMBINACIJA
    )

    razlike = np.minimum(
        np.abs(
            upit.astype(np.int64)
            - kandidat.astype(np.int64)
        ),
        BROJ_SVIH_KOMBINACIJA
        - np.abs(
            upit.astype(np.int64)
            - kandidat.astype(np.int64)
        ),
    )

    lokalni_skor = (
        1.0 - razlike / maksimalna_razlika
    )

    return float(
        np.dot(tezine, lokalni_skor)
    )


def pronadji_slicne_regione(
    niz: np.ndarray,
    kraj: int,
) -> list[tuple[float, int]]:
    duzina = DUZINA_SEKVENCE
    upit = niz[kraj - duzina:kraj]

    kandidati = np.linspace(
        duzina,
        kraj - 1,
        min(10_000, kraj - duzina),
        dtype=int,
    )

    rezultati = []

    for kandidat_kraj in kandidati:
        kandidat = niz[
            kandidat_kraj - duzina:
            kandidat_kraj
        ]

        skor = sekvencijalno_poravnanje(
            upit,
            kandidat,
        )

        rezultati.append(
            (skor, int(kandidat_kraj))
        )

    rezultati.sort(
        key=lambda rezultat: (
            -rezultat[0],
            -rezultat[1],
        )
    )

    return rezultati[:BROJ_SLICNIH_REGIONA]


# =============================================================================
# SPEKTAR, AUTOKORELACIJA I ENTROPIJA
# =============================================================================

def spektralna_analiza(
    niz: np.ndarray,
    maksimum: int,
) -> dict:
    deo = niz[:maksimum].astype(np.float64)
    normalizovan = (
        deo / BROJ_SVIH_KOMBINACIJA
    )

    normalizovan -= normalizovan.mean()

    spektar = np.abs(
        np.fft.rfft(normalizovan)
    )

    spektar[0] = 0.0

    dominantni_indeks = int(
        np.argmax(spektar)
    )

    dominantna_jacina = float(
        spektar[dominantni_indeks]
        / max(len(normalizovan), 1)
    )

    autokorelacije = {}

    for pomeraj in (
        1,
        2,
        3,
        5,
        7,
        11,
        13,
        17,
        23,
        31,
    ):
        if len(normalizovan) <= pomeraj:
            continue

        autokorelacije[pomeraj] = float(
            np.corrcoef(
                normalizovan[:-pomeraj],
                normalizovan[pomeraj:],
            )[0, 1]
        )

    simboli = (
        deo.astype(np.int64) % 4096
    )

    brojanja = np.bincount(
        simboli,
        minlength=4096,
    )

    verovatnoce = (
        brojanja[brojanja > 0]
        / brojanja.sum()
    )

    entropija = float(
        -np.sum(
            verovatnoce
            * np.log2(verovatnoce)
        )
    )

    return {
        "dominantni_indeks": dominantni_indeks,
        "dominantna_jacina": dominantna_jacina,
        "autokorelacije": autokorelacije,
        "entropija": entropija,
    }


# =============================================================================
# DE BRUIJN GRAF
# =============================================================================

def napravi_de_bruijn_graf(
    simboli: np.ndarray,
    red: int = 4,
) -> dict:
    graf = defaultdict(Counter)

    for kraj in range(
        red,
        len(simboli),
    ):
        stanje = tuple(
            int(x)
            for x in simboli[
                kraj - red:kraj
            ]
        )

        nastavak = int(simboli[kraj])
        graf[stanje][nastavak] += 1

    return graf


def de_bruijn_predikcija(
    graf: dict,
    poslednji_simboli: np.ndarray,
) -> int | None:
    stanje = tuple(
        int(x)
        for x in poslednji_simboli[-4:]
    )

    if stanje not in graf:
        return None

    return int(
        graf[stanje].most_common(1)[0][0]
    )


# =============================================================================
# OSOBINE ZA REGRESORE
# =============================================================================

def napravi_ml_podatke(
    niz: np.ndarray,
    pocetak: int,
    kraj: int,
) -> tuple[np.ndarray, np.ndarray]:
    osobine = []
    mete = []

    for indeks in range(
        max(DUZINA_SEKVENCE, pocetak),
        kraj,
    ):
        istorija = niz[
            indeks - DUZINA_SEKVENCE:
            indeks
        ].astype(np.float64)

        normalizovano = (
            istorija
            / BROJ_SVIH_KOMBINACIJA
        )

        razlike = np.diff(normalizovano)

        fft = np.abs(
            np.fft.rfft(
                normalizovano
                - normalizovano.mean()
            )
        )

        rezimi = np.asarray(
            [
                istorija[-1] % 7,
                istorija[-1] % 11,
                istorija[-1] % 13,
                istorija[-1] % 17,
            ],
            dtype=float,
        )

        osobina = np.concatenate(
            [
                normalizovano,
                razlike,
                fft,
                rezimi / np.asarray(
                    [7, 11, 13, 17],
                    dtype=float,
                ),
            ]
        )

        osobine.append(osobina)
        mete.append(
            niz[indeks]
            / BROJ_SVIH_KOMBINACIJA
        )

    return (
        np.asarray(osobine, dtype=np.float32),
        np.asarray(mete, dtype=np.float32),
    )


def prilagodi_regresore(
    niz: np.ndarray,
    razvoj_kraj: int,
) -> dict:
    kraj = min(
        razvoj_kraj,
        ML_UZORAK,
    )

    x, y = napravi_ml_podatke(
        niz,
        0,
        kraj,
    )

    podela = int(len(x) * 0.80)

    scaler = StandardScaler()
    x_trening = scaler.fit_transform(
        x[:podela]
    )

    x_validacija = scaler.transform(
        x[podela:]
    )

    extra_trees = ExtraTreesRegressor(
        n_estimators=200,
        min_samples_leaf=2,
        max_features=0.70,
        n_jobs=-1,
        random_state=SEED,
    )

    gradient_boosting = HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.05,
        max_leaf_nodes=63,
        l2_regularization=1.0,
        random_state=SEED,
    )

    extra_trees.fit(
        x_trening,
        y[:podela],
    )

    gradient_boosting.fit(
        x_trening,
        y[:podela],
    )

    extra_pred = extra_trees.predict(
        x_validacija
    )

    gradient_pred = gradient_boosting.predict(
        x_validacija
    )

    return {
        "x_validacija": x_validacija,
        "y_validacija": y[podela:],
        "extra_pred": extra_pred,
        "gradient_pred": gradient_pred,
        "scaler": scaler,
        "extra_trees": extra_trees,
        "gradient_boosting": gradient_boosting,
        "extra_mae": float(
            np.mean(
                np.abs(
                    extra_pred - y[podela:]
                )
            )
        ),
        "gradient_mae": float(
            np.mean(
                np.abs(
                    gradient_pred - y[podela:]
                )
            )
        ),
    }


# =============================================================================
# VREMENSKI TRANSFORMER
# =============================================================================

class VremenskiTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.ulaz = nn.Linear(
            5,
            TRANSFORMER_DIMENZIJA,
        )

        sloj = nn.TransformerEncoderLayer(
            d_model=TRANSFORMER_DIMENZIJA,
            nhead=TRANSFORMER_GLAVE,
            dim_feedforward=TRANSFORMER_DIMENZIJA * 2,
            dropout=0.0,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            sloj,
            num_layers=TRANSFORMER_SLOJEVI,
        )

        self.izlaz = nn.Linear(
            TRANSFORMER_DIMENZIJA,
            1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ulaz(x)
        x = self.transformer(x)
        x = x[:, -1, :]
        return torch.sigmoid(
            self.izlaz(x)
        ).squeeze(-1)


def transformer_osobine(
    niz: np.ndarray,
    kraj: int,
) -> tuple[np.ndarray, np.ndarray]:
    niz = niz[:kraj]
    x = []
    y = []

    for indeks in range(
        DUZINA_SEKVENCE,
        len(niz),
    ):
        sekvenca = niz[
            indeks - DUZINA_SEKVENCE:
            indeks
        ].astype(np.float64)

        osobine = np.column_stack(
            [
                sekvenca / BROJ_SVIH_KOMBINACIJA,
                (sekvenca % 7) / 7.0,
                (sekvenca % 11) / 11.0,
                (sekvenca % 13) / 13.0,
                (sekvenca % 17) / 17.0,
            ]
        )

        x.append(osobine)
        y.append(
            niz[indeks]
            / BROJ_SVIH_KOMBINACIJA
        )

    return (
        np.asarray(x, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
    )


def prilagodi_transformer(
    niz: np.ndarray,
    razvoj_kraj: int,
) -> dict:
    kraj = min(
        razvoj_kraj,
        TRANSFORMER_UZORAK,
    )

    x, y = transformer_osobine(
        niz,
        kraj,
    )

    podela = int(len(x) * 0.80)

    model = VremenskiTransformer()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-4,
    )

    kriterijum = nn.MSELoss()

    x_trening = torch.from_numpy(
        x[:podela]
    )

    y_trening = torch.from_numpy(
        y[:podela]
    )

    model.train()

    for epoha in range(TRANSFORMER_EPOHE):
        redosled = torch.randperm(
            len(x_trening)
        )

        gubici = []

        for pocetak in range(
            0,
            len(x_trening),
            TRANSFORMER_BATCH,
        ):
            indeksi = redosled[
                pocetak:
                pocetak + TRANSFORMER_BATCH
            ]

            xb = x_trening[indeksi]
            yb = y_trening[indeksi]

            optimizer.zero_grad()

            predikcija = model(xb)
            gubitak = kriterijum(
                predikcija,
                yb,
            )

            gubitak.backward()
            optimizer.step()

            gubici.append(
                float(gubitak.detach())
            )

        print(
            f"  Transformer epoha "
            f"{epoha + 1}/{TRANSFORMER_EPOHE}"
            f" — gubitak {np.mean(gubici):.8f}"
        )

    model.eval()

    with torch.no_grad():
        validaciona_predikcija = model(
            torch.from_numpy(x[podela:])
        ).numpy()

    validacioni_mae = float(
        np.mean(
            np.abs(
                validaciona_predikcija
                - y[podela:]
            )
        )
    )

    return {
    "model": model,
    "mae": validacioni_mae,
    "y_validacija": y[podela:],
    "pred_validacija": validaciona_predikcija,
    }


# =============================================================================
# BAJESOVO PONDERISANJE KANDIDATA
# =============================================================================

def bajesove_tezine(
    greske: dict[str, float],
) -> dict[str, float]:
    """
    Pretvara validacione greške u posteriorne težine modela.

    Model sa nula grešaka dobija najveću verovatnoću, ali se svi
    kandidati prvo stvarno ocenjuju.
    """

    nazivi = list(greske)
    vrednosti = np.asarray(
        [greske[naziv] for naziv in nazivi],
        dtype=np.float64,
    )

    skala = max(
        float(np.median(vrednosti[vrednosti > 0]))
        if np.any(vrednosti > 0)
        else 1.0,
        EPS,
    )

    log_dokazi = (
        -vrednosti / skala
    )

    log_dokazi -= logsumexp(log_dokazi)
    tezine = np.exp(log_dokazi)

    return {
        naziv: float(tezina)
        for naziv, tezina in zip(
            nazivi,
            tezine,
        )
    }


def poslednja_ml_osobina(
    niz: np.ndarray,
) -> np.ndarray:
    istorija = niz[
        -DUZINA_SEKVENCE:
    ].astype(np.float64)

    normalizovano = (
        istorija / BROJ_SVIH_KOMBINACIJA
    )

    razlike = np.diff(normalizovano)

    fft = np.abs(
        np.fft.rfft(
            normalizovano
            - normalizovano.mean()
        )
    )

    rezimi = np.asarray(
        [
            istorija[-1] % 7,
            istorija[-1] % 11,
            istorija[-1] % 13,
            istorija[-1] % 17,
        ],
        dtype=float,
    )

    osobina = np.concatenate(
        [
            normalizovano,
            razlike,
            fft,
            rezimi / np.asarray(
                [7, 11, 13, 17],
                dtype=float,
            ),
        ]
    )

    return osobina.reshape(1, -1).astype(
        np.float32
    )


def poslednja_transformer_osobina(
    niz: np.ndarray,
) -> np.ndarray:
    sekvenca = niz[
        -DUZINA_SEKVENCE:
    ].astype(np.float64)

    osobine = np.column_stack(
        [
            sekvenca / BROJ_SVIH_KOMBINACIJA,
            (sekvenca % 7) / 7.0,
            (sekvenca % 11) / 11.0,
            (sekvenca % 13) / 13.0,
            (sekvenca % 17) / 17.0,
        ]
    )

    return osobine[
        np.newaxis,
        :,
        :,
    ].astype(np.float32)


def normalizovana_predikcija_u_rang(
    vrednost: float,
) -> int:
    rang = int(
        round(
            float(vrednost)
            * BROJ_SVIH_KOMBINACIJA
        )
    )

    return int(
        np.clip(
            rang,
            0,
            BROJ_SVIH_KOMBINACIJA - 1,
        )
    )


def gubitak_pogodaka(
    predikcije: np.ndarray,
    stvarne_vrednosti: np.ndarray,
) -> float:
    if len(stvarne_vrednosti) == 0:
        return 1.0

    gubici = []

    for predikcija, stvarno in zip(
        predikcije,
        stvarne_vrednosti,
    ):
        pred_rang = normalizovana_predikcija_u_rang(
            float(predikcija)
        )

        stvarni_rang = normalizovana_predikcija_u_rang(
            float(stvarno)
        )

        pred_brojevi = set(
            rang_u_kombinaciju(pred_rang)
        )

        stvarni_brojevi = set(
            rang_u_kombinaciju(stvarni_rang)
        )

        pogodaka = len(
            pred_brojevi & stvarni_brojevi
        )

        gubici.append(
            1.0 - pogodaka / 7.0
        )

    return float(np.mean(gubici))


def sastavi_next(
    kandidati: dict[str, int],
    tezine: dict[str, float],
) -> list[int]:
    skorovi = np.zeros(
        BROJ_KUGLICA + 1,
        dtype=np.float64,
    )

    for naziv, rang in kandidati.items():
        kombinacija = rang_u_kombinaciju(
            int(rang)
        )

        tezina = float(
            tezine.get(naziv, 0.0)
        )

        for broj in kombinacija:
            skorovi[broj] += tezina

    poredak = sorted(
        range(1, BROJ_KUGLICA + 1),
        key=lambda broj: (
            -skorovi[broj],
            broj,
        ),
    )

    return sorted(
        poredak[:BROJEVA_U_KOMBINACIJI]
    )



# =============================================================================
# NESTED WALK-FORWARD I HOLDOUT
# =============================================================================

@njit
def proveri_modularni_model(
    niz: np.ndarray,
    a: np.ndarray,
    c: np.ndarray,
    broj_rezima: int,
    razvoj_kraj: int,
    validacija_kraj: int,
    vidljivi_kraj: int,
) -> tuple[int, int, int]:
    razvojne_greske = 0
    validacione_greske = 0
    holdout_greske = 0

    for indeks in range(1, vidljivi_kraj):
        prethodno = niz[indeks - 1]
        rezim = prethodno % broj_rezima

        predikcija = (
            a[rezim] * prethodno
            + c[rezim]
        ) % BROJ_SVIH_KOMBINACIJA

        stvarno = niz[indeks]

        if predikcija != stvarno:
            if indeks < razvoj_kraj:
                razvojne_greske += 1
            elif indeks < validacija_kraj:
                validacione_greske += 1
            else:
                holdout_greske += 1

    return (
        razvojne_greske,
        validacione_greske,
        holdout_greske,
    )


def izracunaj_pogotke_rangova(
    predikcije: np.ndarray,
    stvarne_vrednosti: np.ndarray,
) -> np.ndarray:
    pogoci = np.zeros(
        len(stvarne_vrednosti),
        dtype=np.int64,
    )

    for indeks, (predikcija, stvarno) in enumerate(
        zip(predikcije, stvarne_vrednosti)
    ):
        pred_rang = normalizovana_predikcija_u_rang(
            float(predikcija)
        )

        stvarni_rang = normalizovana_predikcija_u_rang(
            float(stvarno)
        )

        pred_brojevi = set(
            rang_u_kombinaciju(pred_rang)
        )

        stvarni_brojevi = set(
            rang_u_kombinaciju(stvarni_rang)
        )

        pogoci[indeks] = len(
            pred_brojevi & stvarni_brojevi
        )

    return pogoci


def nested_walk_forward(
    niz: np.ndarray,
    validacija_kraj: int,
) -> dict:
    foldovi = (
        (0.35, 0.45),
        (0.45, 0.55),
        (0.55, 0.65),
        (0.65, 0.75),
        (0.75, 0.80),
    )

    svi_pogoci = []
    greske_po_foldu = []
    fold_rezultati = []

    dostupni_redovi = min(
        validacija_kraj,
        len(niz),
    )

    for redni, (
        udeo_obuke,
        udeo_provere,
    ) in enumerate(foldovi, start=1):
        kraj_obuke = max(
            DUZINA_SEKVENCE + 20,
            int(dostupni_redovi * udeo_obuke),
        )

        kraj_provere = min(
            dostupni_redovi,
            int(dostupni_redovi * udeo_provere),
        )

        x_obuka, y_obuka = napravi_ml_podatke(
            niz,
            0,
            kraj_obuke,
        )

        x_provera, y_provera = napravi_ml_podatke(
            niz,
            kraj_obuke,
            kraj_provere,
        )

        if (
            len(x_obuka) < 30
            or len(x_provera) == 0
        ):
            continue

        scaler = StandardScaler()

        x_obuka = scaler.fit_transform(
            x_obuka
        )

        x_provera = scaler.transform(
            x_provera
        )

        extra_trees = ExtraTreesRegressor(
            n_estimators=200,
            min_samples_leaf=2,
            max_features=0.70,
            n_jobs=-1,
            random_state=SEED + redni,
        )

        gradient_boosting = HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=63,
            l2_regularization=1.0,
            random_state=SEED + redni,
        )

        extra_trees.fit(
            x_obuka,
            y_obuka,
        )

        gradient_boosting.fit(
            x_obuka,
            y_obuka,
        )

        extra_pred = extra_trees.predict(
            x_provera
        )

        gradient_pred = gradient_boosting.predict(
            x_provera
        )

        extra_gubitak = gubitak_pogodaka(
            extra_pred,
            y_provera,
        )

        gradient_gubitak = gubitak_pogodaka(
            gradient_pred,
            y_provera,
        )

        lokalne_tezine = bajesove_tezine(
            {
                "Extra Trees": extra_gubitak,
                "Gradient boosting": gradient_gubitak,
            }
        )

        ansambl_pred = (
            lokalne_tezine["Extra Trees"]
            * extra_pred
            + lokalne_tezine["Gradient boosting"]
            * gradient_pred
        )

        pogoci = izracunaj_pogotke_rangova(
            ansambl_pred,
            y_provera,
        )

        potpuno_netacnih = int(
            np.sum(pogoci < 7)
        )

        prosek_pogodaka = float(
            np.mean(pogoci)
        )

        svi_pogoci.extend(
            pogoci.tolist()
        )

        greske_po_foldu.append(
            potpuno_netacnih
        )

        fold_rezultati.append(
            {
                "fold": redni,
                "obuka": kraj_obuke,
                "provera": len(y_provera),
                "prosek_pogodaka": prosek_pogodaka,
                "potpuno_tacnih": int(
                    np.sum(pogoci == 7)
                ),
                "greske": potpuno_netacnih,
            }
        )

        print(
            f"  Fold {redni}/{len(foldovi)}"
            f" — obuka {kraj_obuke:,}"
            f" — provera {len(y_provera):,}"
            f" — prosek pogodaka {prosek_pogodaka:.6f}"
            f" — potpuno tačnih {np.sum(pogoci == 7):,}"
        )

    svi_pogoci = np.asarray(
        svi_pogoci,
        dtype=np.int64,
    )

    return {
        "foldovi": fold_rezultati,
        "broj_provera": int(len(svi_pogoci)),
        "greske_po_foldu": greske_po_foldu,
        "ukupno_gresaka": int(
            np.sum(greske_po_foldu)
        ),
        "potpuno_tacnih": int(
            np.sum(svi_pogoci == 7)
        ) if len(svi_pogoci) else 0,
        "prosek_pogodaka": float(
            np.mean(svi_pogoci)
        ) if len(svi_pogoci) else float("nan"),
    }


def proveri_zakljucani_holdout(
    niz: np.ndarray,
    validacija_kraj: int,
    regresori: dict,
    transformer: dict,
    tezine: dict[str, float],
) -> dict:
    x_holdout, y_holdout = napravi_ml_podatke(
        niz,
        validacija_kraj,
        len(niz),
    )

    if len(x_holdout) == 0:
        return {
            "broj_provera": 0,
            "potpuno_tacnih": 0,
            "ukupno_gresaka": 0,
            "prosek_pogodaka": float("nan"),
            "raspodela_pogodaka": {
                broj: 0
                for broj in range(8)
            },
        }

    x_holdout_scaled = regresori[
        "scaler"
    ].transform(x_holdout)

    extra_pred = regresori[
        "extra_trees"
    ].predict(x_holdout_scaled)

    gradient_pred = regresori[
        "gradient_boosting"
    ].predict(x_holdout_scaled)

    transformer_x, transformer_y = transformer_osobine(
        niz,
        len(niz),
    )

    prvi_transformer_indeks = (
        validacija_kraj - DUZINA_SEKVENCE
    )

    prvi_transformer_indeks = max(
        0,
        prvi_transformer_indeks,
    )

    transformer_x = transformer_x[
        prvi_transformer_indeks:
    ]

    transformer_y = transformer_y[
        prvi_transformer_indeks:
    ]

    broj_zajednickih = min(
        len(y_holdout),
        len(transformer_y),
    )

    if broj_zajednickih == 0:
        return {
            "broj_provera": 0,
            "potpuno_tacnih": 0,
            "ukupno_gresaka": 0,
            "prosek_pogodaka": float("nan"),
            "raspodela_pogodaka": {
                broj: 0
                for broj in range(8)
            },
        }

    x_holdout = x_holdout[
        -broj_zajednickih:
    ]

    y_holdout = y_holdout[
        -broj_zajednickih:
    ]

    extra_pred = extra_pred[
        -broj_zajednickih:
    ]

    gradient_pred = gradient_pred[
        -broj_zajednickih:
    ]

    transformer_x = transformer_x[
        -broj_zajednickih:
    ]

    transformer["model"].eval()

    with torch.no_grad():
        transformer_pred = transformer[
            "model"
        ](
            torch.from_numpy(
                transformer_x
            )
        ).numpy()

    pogoci = []

    for indeks in range(broj_zajednickih):
        kandidati_reda = {
            "Extra Trees":
                normalizovana_predikcija_u_rang(
                    extra_pred[indeks]
                ),
            "Gradient boosting":
                normalizovana_predikcija_u_rang(
                    gradient_pred[indeks]
                ),
            "Vremenski Transformer":
                normalizovana_predikcija_u_rang(
                    transformer_pred[indeks]
                ),
        }

        aktivne_tezine = {
            naziv: tezine[naziv]
            for naziv in kandidati_reda
            if naziv in tezine
        }

        zbir_tezina = sum(
            aktivne_tezine.values()
        )

        if zbir_tezina <= 0:
            aktivne_tezine = {
                naziv: 1.0 / len(kandidati_reda)
                for naziv in kandidati_reda
            }
        else:
            aktivne_tezine = {
                naziv: vrednost / zbir_tezina
                for naziv, vrednost
                in aktivne_tezine.items()
            }

        pred_kombinacija = set(
            sastavi_next(
                kandidati_reda,
                aktivne_tezine,
            )
        )

        stvarni_rang = normalizovana_predikcija_u_rang(
            y_holdout[indeks]
        )

        stvarna_kombinacija = set(
            rang_u_kombinaciju(
                stvarni_rang
            )
        )

        pogoci.append(
            len(
                pred_kombinacija
                & stvarna_kombinacija
            )
        )

    pogoci = np.asarray(
        pogoci,
        dtype=np.int64,
    )

    raspodela = {
        broj: int(np.sum(pogoci == broj))
        for broj in range(8)
    }

    return {
        "broj_provera": int(len(pogoci)),
        "potpuno_tacnih": int(
            np.sum(pogoci == 7)
        ),
        "ukupno_gresaka": int(
            np.sum(pogoci < 7)
        ),
        "prosek_pogodaka": float(
            np.mean(pogoci)
        ),
        "raspodela_pogodaka": raspodela,
    }



# =============================================================================
# GLAVNI PROGRAM
# =============================================================================

def obradi_igru(
    naziv: str,
    putanja: str,
) -> dict:
    naslov(f"OBRADA: {naziv}")

    niz = ucitaj_csv(putanja)

    razvoj_kraj, validacija_kraj = granice_podele(
        len(niz)
    )

    print(f"CSV: {putanja}")
    print(f"Broj redova: {len(niz):,}")
    print("Prvi red se tretira kao najstariji.")
    print("Poslednji red se tretira kao najnoviji.")
    print(f"Razvojni deo: {razvoj_kraj:,}")
    print(
        f"Validacioni deo: "
        f"{validacija_kraj - razvoj_kraj:,}"
    )
    print(
        f"Zaključani holdout: "
        f"{len(niz) - validacija_kraj:,}"
    )

    naslov("1. AUTOMATSKO RANGIRANJE PORODICA GENERATORA")

    modularni_model, porodice = rangiraj_porodice_generatora(
        niz,
        razvoj_kraj,
        validacija_kraj,
    )

    if modularni_model is None:
        print(
            "Nije pronađena tačna modularna rekurencija."
        )
    else:
        print(
            f"Izabrani simboličko-modularni model: "
            f"{modularni_model.naziv}"
        )

        for rezultat in porodice:
            print(
                f"  {rezultat['model'].naziv:<38}"
                f" greške={rezultat['greske']:<8}"
                f" srednja greška="
                f"{rezultat['srednja_greska']:.6f}"
            )

    naslov("2. DETEKCIJA SKRIVENIH REŽIMA I HMM")

    rezimi = detektuj_rezime(
        niz,
        razvoj_kraj,
    )

    hmm = prilagodi_hmm(
        osobine=rezimi["osobine"],
        broj_rezima=rezimi["broj_rezima"],
    )

    poslednji_rezim = int(
        hmm["stanja"][-1]
    )

    print(
        f"Automatski izabran broj režima: "
        f"{rezimi['broj_rezima']}"
    )
    print(
        f"HMM pronađenih stanja: "
        f"{hmm['broj_stanja']}"
    )
    print(
        f"Poslednji režim: "
        f"{poslednji_rezim}"
    )

    naslov("3. K-MER, SUFFIX, MINIMIZER I PORAVNANJE")

    indeks_kraj = min(
        razvoj_kraj,
        len(niz),
        500_000,
    )

    kmer = napravi_kmer_indeks(
        niz,
        indeks_kraj,
    )

    minimizer = napravi_minimizer_indeks(
        kmer["simboli"]
    )

    upit = kmer["simboli"][
        -KMER_DUZINA:
    ]

    suffix_kandidati = suffix_pretraga(
        kmer["simboli"][:-1],
        upit,
    )

    slicni_regioni = pronadji_slicne_regione(
        niz,
        indeks_kraj,
    )

    print(f"k-mer čvorova: {len(kmer['indeks']):,}")
    print(f"Minimizer stavki: {len(minimizer):,}")
    print(f"Suffix kandidata: {len(suffix_kandidati):,}")
    print(
        f"Poravnatih sličnih regiona: "
        f"{len(slicni_regioni):,}"
    )

    naslov("4. SPEKTRALNA, AUTOKORELACIONA I ENTROPIJSKA ANALIZA")

    spektralno = spektralna_analiza(
        niz,
        min(
            razvoj_kraj,
            262_144,
        ),
    )

    print(
        f"Dominantni spektralni indeks: "
        f"{spektralno['dominantni_indeks']}"
    )
    print(
        f"Dominantna spektralna jačina: "
        f"{spektralno['dominantna_jacina']:.9f}"
    )
    print(
        f"Entropija: "
        f"{spektralno['entropija']:.9f}"
    )

    for pomeraj, vrednost in (
        spektralno["autokorelacije"].items()
    ):
        print(
            f"  lag {pomeraj:02d}: "
            f"{vrednost:+.9f}"
        )

    naslov("5. DE BRUIJN GRAF NASTAVAKA")

    de_bruijn = napravi_de_bruijn_graf(
        kmer["simboli"],
        red=4,
    )

    de_bruijn_next = de_bruijn_predikcija(
        de_bruijn,
        kmer["simboli"][-4:],
    )

    print(f"De Bruijn čvorova: {len(de_bruijn):,}")
    print(
        f"De Bruijn simbol nastavka: "
        f"{de_bruijn_next}"
    )

    naslov("6. GRADIENT BOOSTING I EXTRA TREES")

    regresori = prilagodi_regresore(
        niz,
        razvoj_kraj,
    )

    print(
        f"Extra Trees validacioni MAE: "
        f"{regresori['extra_mae']:.9f}"
    )
    print(
        f"Gradient boosting validacioni MAE: "
        f"{regresori['gradient_mae']:.9f}"
    )

    naslov("7. VREMENSKI TRANSFORMER")

    transformer = prilagodi_transformer(
        niz,
        razvoj_kraj,
    )

    print(
        f"Transformer validacioni MAE: "
        f"{transformer['mae']:.9f}"
    )

    naslov("8. BAJESOVO PONDERISANJE KANDIDATA")

    greske_modela = {
        "Extra Trees": gubitak_pogodaka(
            regresori["extra_pred"],
            regresori["y_validacija"],
        ),
        "Gradient boosting": gubitak_pogodaka(
            regresori["gradient_pred"],
            regresori["y_validacija"],
        ),
        "Vremenski Transformer": gubitak_pogodaka(
            transformer["pred_validacija"],
            transformer["y_validacija"],
        ),
    }

    poslednja_ml = regresori["scaler"].transform(
        poslednja_ml_osobina(niz)
    )

    kandidati = {
        "Extra Trees": normalizovana_predikcija_u_rang(
            regresori["extra_trees"].predict(
                poslednja_ml
            )[0]
        ),
        "Gradient boosting": normalizovana_predikcija_u_rang(
            regresori["gradient_boosting"].predict(
                poslednja_ml
            )[0]
        ),
    }

    poslednja_transformer = (
        poslednja_transformer_osobina(niz)
    )

    transformer["model"].eval()

    with torch.no_grad():
        transformer_predikcija = float(
            transformer["model"](
                torch.from_numpy(
                    poslednja_transformer
                )
            ).item()
        )

    kandidati["Vremenski Transformer"] = (
        normalizovana_predikcija_u_rang(
            transformer_predikcija
        )
    )

    if modularni_model is not None:
        modularni_x = niz[
            razvoj_kraj:validacija_kraj
        ]

        modularni_y = niz[
            razvoj_kraj + 1:validacija_kraj + 1
        ]

        if len(modularni_y) > 0:
            modularne_predikcije = (
                modularni_model.predvidi(
                    modularni_x
                )
                / BROJ_SVIH_KOMBINACIJA
            )

            modularne_mete = (
                modularni_y
                / BROJ_SVIH_KOMBINACIJA
            )

            greske_modela[
                "Simboličko-modularni"
            ] = gubitak_pogodaka(
                modularne_predikcije,
                modularne_mete,
            )

            kandidati[
                "Simboličko-modularni"
            ] = modularni_model.predvidi_jedan(
                int(niz[-1])
            )

    tezine = bajesove_tezine(
        greske_modela
    )

    for ime in sorted(
        tezine,
        key=tezine.get,
        reverse=True,
    ):
        print(
            f"  {ime:<28}"
            f" težina={tezine[ime]:.9f}"
            f" gubitak={greske_modela[ime]:.9f}"
        )

    next_kombinacija = sastavi_next(
        kandidati,
        tezine,
    )



    naslov("9. NESTED WALK-FORWARD VALIDACIJA")

    nested = nested_walk_forward(
        niz=niz,
        validacija_kraj=validacija_kraj,
    )

    print(
        f"Broj hronoloških provera: "
        f"{nested['broj_provera']:,}"
    )
    print(
        f"Prosečan broj pogodaka: "
        f"{nested['prosek_pogodaka']:.6f}"
    )
    print(
        f"Potpuno tačnih prelaza: "
        f"{nested['potpuno_tacnih']:,}"
    )
    print(
        f"Netačnih prelaza: "
        f"{nested['ukupno_gresaka']:,}"
    )

    naslov("10. ZAKLJUČANI ZAVRŠNI HOLDOUT")

    holdout = proveri_zakljucani_holdout(
        niz=niz,
        validacija_kraj=validacija_kraj,
        regresori=regresori,
        transformer=transformer,
        tezine=tezine,
    )

    print(
        f"Broj holdout provera: "
        f"{holdout['broj_provera']:,}"
    )
    print(
        f"Prosečan broj pogodaka: "
        f"{holdout['prosek_pogodaka']:.6f}"
    )
    print(
        f"Potpuno tačnih prelaza: "
        f"{holdout['potpuno_tacnih']:,}"
    )
    print(
        f"Netačnih prelaza: "
        f"{holdout['ukupno_gresaka']:,}"
    )
    print("Raspodela pogodaka:")

    for broj, koliko in (
        holdout["raspodela_pogodaka"].items()
    ):
        print(
            f"  {broj} pogodaka: "
            f"{koliko:,}"
        )

    naslov("KONTROLNA LISTA", znak="#")

    status(
        "Automatsko rangiranje generatora",
        len(porodice) > 0,
    )
    status(
        "Simbolička regresija i modularne rekurencije",
        modularni_model is not None,
    )
    status(
        "Detekcija skrivenih režima",
        rezimi["broj_rezima"] > 0,
        f"režima={rezimi['broj_rezima']}",
    )
    status(
        "Hidden Markov / switching-state model",
        hmm["broj_stanja"] > 0,
        f"stanja={hmm['broj_stanja']}",
    )
    status(
        "k-mer, suffix i minimizer pretraga",
        (
            len(kmer["indeks"]) > 0
            and len(minimizer) > 0
        ),
    )
    status(
        "Sekvencijalno poravnanje",
        len(slicni_regioni) > 0,
    )
    status(
        "Spektralna, autokorelaciona i entropijska analiza",
        np.isfinite(
            spektralno["entropija"]
        ),
    )
    status(
        "De Bruijn graf nastavaka",
        len(de_bruijn) > 0,
    )
    status(
        "Gradient boosting i Extra Trees ansambl",
        (
            np.isfinite(
                regresori["extra_mae"]
            )
            and np.isfinite(
                regresori["gradient_mae"]
            )
        ),
    )
    status(
        "Vremenski Transformer",
        np.isfinite(
            transformer["mae"]
        ),
    )
    status(
        "Bajesovo ponderisanje kandidata",
        abs(
            sum(tezine.values()) - 1.0
        ) < 1e-9,
    )
    status(
        "Nested walk-forward validacija",
        nested["broj_provera"] > 0,
        (
            f"provera={nested['broj_provera']}, "
            f"prosek={nested['prosek_pogodaka']:.4f}, "
            f"tačnih={nested['potpuno_tacnih']}"
        ),
    )
    status(
        "Zaključani završni holdout",
        holdout["broj_provera"] > 0,
        (
            f"provera={holdout['broj_provera']}, "
            f"prosek={holdout['prosek_pogodaka']:.4f}, "
            f"tačnih={holdout['potpuno_tacnih']}"
        ),
    )
    status(
        "Jedna NEXT predikcija",
        len(next_kombinacija) == 7,
    )







    naslov("KONAČNA NEXT PREDIKCIJA", znak="#")

    zakljucani_next_rang = kombinacija_u_rang(
        next_kombinacija
    )

    broj_razvojnih = razvoj_kraj
    broj_validacionih = (
        validacija_kraj - razvoj_kraj
    )
    broj_holdout = (
        len(niz) - validacija_kraj
    )

    naslov(
        f"KONAČNI REZULTAT — {naziv}",
        znak="#",
    )

    print(
        f"Broj kombinacija u CSV-u:          "
        f"{len(niz):,}"
    )
    print(
        f"Razvojni skup:                     "
        f"{broj_razvojnih:,}"
    )
    print(
        f"Zaključana validacija:             "
        f"{broj_validacionih:,}"
    )
    print(
        f"Završni holdout:                   "
        f"{broj_holdout:,}"
    )

    print()

    print(
        f"Nested walk-forward provera:       "
        f"{nested['broj_provera']:,}"
    )
    print(
        f"Greške u walk-forward proveri:     "
        f"{nested['ukupno_gresaka']:,}"
    )
    print(
        f"Potpuno tačni walk-forward prelazi:"
        f" {nested['potpuno_tacnih']:,}"
    )
    print(
        f"Prosečan broj pogodaka:            "
        f"{nested['prosek_pogodaka']:.6f}"
    )

    print()

    print(
        f"Zaključanih holdout provera:       "
        f"{holdout['broj_provera']:,}"
    )
    print(
        f"Greške na završnom holdoutu:       "
        f"{holdout['ukupno_gresaka']:,}"
    )
    print(
        f"Potpuno tačni holdout prelazi:     "
        f"{holdout['potpuno_tacnih']:,}"
    )
    print(
        f"Holdout prosek pogodaka:           "
        f"{holdout['prosek_pogodaka']:.6f}"
    )

    print()

    print(
        f"Poslednji režim:                   "
        f"{poslednji_rezim}"
    )
    print(
        f"Zaključani NEXT rang:              "
        f"{zakljucani_next_rang:,}"
    )
    print(
        f"NEXT:                              "
        f"{formatiraj_kombinaciju(next_kombinacija)}"
    )

    return {
        "naziv": naziv,
        "broj_redova": len(niz),
        "istorijskih_prelaza": len(niz) - 1,
        "poslednji_rezim": poslednji_rezim,
        "nested": nested,
        "holdout": holdout,
        "next": next_kombinacija,
    }


def main() -> None:
    pocetak_programa = time.time()

    naslov(
        "LOTO 7/39 — NAJJAČI SISTEM PREPOZNAVANJA "
        "SKRIVENE ZAKONITOSTI"
    )

    print(f"Seed: {SEED}")
    print(
        f"Ukupno mogućih kombinacija: "
        f"{BROJ_SVIH_KOMBINACIJA:,}"
    )

    loto = obradi_igru(
        "Loto",
        LOTO_CSV,
    )

    naslov("KONAČNA NEXT PREDIKCIJA", znak="#")

    print(
        f"Loto: "
        f"{formatiraj_kombinaciju(loto['next'])}"
    )



    print(
        f"\nUkupno vreme: "
        f"{time.time() - pocetak_programa:.2f} sekundi"
    )


if __name__ == "__main__":
    main()



"""
==============================================================================
LOTO 7/39 — NAJJAČI SISTEM PREPOZNAVANJA SKRIVENE ZAKONITOSTI
==============================================================================
Seed: 39
Ukupno mogućih kombinacija: 15,380,937

==============================================================================
OBRADA: Loto
==============================================================================
CSV: /Users/4c/Desktop/GHQ/data/loto7_4682_k72.csv
Broj redova: 4,682
Prvi red se tretira kao najstariji.
Poslednji red se tretira kao najnoviji.
Razvojni deo: 2,809
Validacioni deo: 936
Zaključani holdout: 937

==============================================================================
1. AUTOMATSKO RANGIRANJE PORODICA GENERATORA
==============================================================================
Nije pronađena tačna modularna rekurencija.

==============================================================================
2. DETEKCIJA SKRIVENIH REŽIMA I HMM
==============================================================================
Automatski izabran broj režima: 9
HMM pronađenih stanja: 9
Poslednji režim: 6

==============================================================================
3. K-MER, SUFFIX, MINIMIZER I PORAVNANJE
==============================================================================
k-mer čvorova: 2,805
Minimizer stavki: 928
Suffix kandidata: 1
Poravnatih sličnih regiona: 100

==============================================================================
4. SPEKTRALNA, AUTOKORELACIONA I ENTROPIJSKA ANALIZA
==============================================================================
Dominantni spektralni indeks: 216
Dominantna spektralna jačina: 0.014884790
Entropija: 10.854976742
  lag 01: +0.007745624
  lag 02: +0.017246036
  lag 03: -0.033725789
  lag 05: +0.005295297
  lag 07: +0.021020699
  lag 11: +0.005477659
  lag 13: -0.007947843
  lag 17: -0.000885541
  lag 23: +0.007266824
  lag 31: -0.025879130

==============================================================================
5. DE BRUIJN GRAF NASTAVAKA
==============================================================================
De Bruijn čvorova: 2,805
De Bruijn simbol nastavka: None

==============================================================================
6. GRADIENT BOOSTING I EXTRA TREES
==============================================================================
Extra Trees validacioni MAE: 0.261729899
Gradient boosting validacioni MAE: 0.266343577

==============================================================================
7. VREMENSKI TRANSFORMER
==============================================================================
  Transformer epoha 1/4 — gubitak 0.10633416
  Transformer epoha 2/4 — gubitak 0.09489754
  Transformer epoha 3/4 — gubitak 0.08467993
  Transformer epoha 4/4 — gubitak 0.08625122
Transformer validacioni MAE: 0.264972270

==============================================================================
8. BAJESOVO PONDERISANJE KANDIDATA
==============================================================================
  Vremenski Transformer        težina=0.334778777 gubitak=0.817927171
  Extra Trees                  težina=0.332713523 gubitak=0.823020117
  Gradient boosting            težina=0.332507700 gubitak=0.823529412

==============================================================================
9. NESTED WALK-FORWARD VALIDACIJA
==============================================================================
  Fold 1/5 — obuka 1,310 — provera 375 — prosek pogodaka 1.290667 — potpuno tačnih 0
  Fold 2/5 — obuka 1,685 — provera 374 — prosek pogodaka 1.192513 — potpuno tačnih 0
  Fold 3/5 — obuka 2,059 — provera 375 — prosek pogodaka 1.304000 — potpuno tačnih 0
  Fold 4/5 — obuka 2,434 — provera 374 — prosek pogodaka 1.275401 — potpuno tačnih 0
  Fold 5/5 — obuka 2,808 — provera 188 — prosek pogodaka 1.335106 — potpuno tačnih 0
Broj hronoloških provera: 1,686
Prosečan broj pogodaka: 1.273428
Potpuno tačnih prelaza: 0
Netačnih prelaza: 1,686

==============================================================================
10. ZAKLJUČANI ZAVRŠNI HOLDOUT
==============================================================================
Broj holdout provera: 937
Prosečan broj pogodaka: 1.277481
Potpuno tačnih prelaza: 0
Netačnih prelaza: 937
Raspodela pogodaka:
  0 pogodaka: 175
  1 pogodaka: 410
  2 pogodaka: 280
  3 pogodaka: 62
  4 pogodaka: 9
  5 pogodaka: 1
  6 pogodaka: 0
  7 pogodaka: 0

##############################################################################
KONTROLNA LISTA
##############################################################################
Automatsko rangiranje generatora                 NIJE PROŠLO
Simbolička regresija i modularne rekurencije     NIJE PROŠLO
Detekcija skrivenih režima                       PROŠLO        režima=9
Hidden Markov / switching-state model            PROŠLO        stanja=9
k-mer, suffix i minimizer pretraga               PROŠLO
Sekvencijalno poravnanje                         PROŠLO
Spektralna, autokorelaciona i entropijska analiza PROŠLO
De Bruijn graf nastavaka                         PROŠLO
Gradient boosting i Extra Trees ansambl          PROŠLO
Vremenski Transformer                            PROŠLO
Bajesovo ponderisanje kandidata                  PROŠLO
Nested walk-forward validacija                   PROŠLO        provera=1686, prosek=1.2734, tačnih=0
Zaključani završni holdout                       PROŠLO        provera=937, prosek=1.2775, tačnih=0
Jedna NEXT predikcija                            PROŠLO

##############################################################################
KONAČNA NEXT PREDIKCIJA
##############################################################################

##############################################################################
KONAČNI REZULTAT — Loto
##############################################################################
Broj kombinacija u CSV-u:          4,682
Razvojni skup:                     2,809
Zaključana validacija:             936
Završni holdout:                   937

Nested walk-forward provera:       1,686
Greške u walk-forward proveri:     1,686
Potpuno tačni walk-forward prelazi: 0
Prosečan broj pogodaka:            1.273428

Zaključanih holdout provera:       937
Greške na završnom holdoutu:       937
Potpuno tačni holdout prelazi:     0
Holdout prosek pogodaka:           1.277481

Poslednji režim:                   6
Zaključani NEXT rang:              5,384,717
NEXT:                              03, 04, 18, 19, 28, 33, 34

##############################################################################
KONAČNA NEXT PREDIKCIJA
##############################################################################
Loto: 03, 04, 18, 19, 28, 33, 34

Ukupno vreme: 22.20 sekundi
"""
